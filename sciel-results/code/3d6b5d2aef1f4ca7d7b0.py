"""HP lattice protein folding on the cubic lattice.

Strategy:
  1. Fast initial growth with contact bias.
  2. Iterated Local Search (ILS) with 2-opt (reversal) moves and
     a greedy local search phase. This avoids the expensive pivot
     moves and full-energy recomputation of the previous approach.
  3. Efficient delta-energy computation using position maps.
  4. Multiple restarts to escape local minima.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _get_neighbours_fast(x, y, z):
    return [
        (x + 1, y, z), (x - 1, y, z),
        (x, y + 1, z), (x, y - 1, z),
        (x, y, z + 1), (x, y, z - 1)
    ]

def _energy(sequence, chain):
    """Compute HP energy: -number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _get_neighbours_fast(*p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _delta_energy_fast(sequence, chain, pos_map, i, new_pos):
    """Fast delta energy computation for moving residue i to new_pos."""
    old_pos = chain[i]
    if old_pos == new_pos:
        return 0
    
    is_h = sequence[i] == 'H'
    if not is_h:
        return 0
    
    # Count H-H contacts involving residue i at old_pos
    old_contacts = 0
    for q in _get_neighbours_fast(*old_pos):
        j = pos_map.get(q)
        if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
            old_contacts += 1
    
    # Count H-H contacts involving residue i at new_pos
    new_contacts = 0
    for q in _get_neighbours_fast(*new_pos):
        j = pos_map.get(q)
        if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
            new_contacts += 1
    
    return -(new_contacts - old_contacts)

def _grow(sequence, dimension, rng, bias=0.7):
    """Randomized greedy chain growth with tunable contact bias."""
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)

    for i in range(1, n):
        options = []
        last = chain[-1]
        for site in _get_neighbours_fast(*last):
            if site in occupied:
                continue
            
            # Calculate immediate gain
            gain = 0
            if sequence[i] == 'H':
                for q in _get_neighbours_fast(*site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            # Calculate free space (connectivity)
            free = 0
            for q in _get_neighbours_fast(*site):
                if q not in occupied:
                    free += 1
            
            # Heuristic: prefer sites that leave more options for the next step
            lookahead_score = 0
            if i < n - 1:
                for q in _get_neighbours_fast(*site):
                    if q not in occupied and q != last:
                        lookahead_score += 1
                if lookahead_score == 0:
                    gain -= 100 
            
            # Mix contact gain and free space with tunable bias
            score = gain * bias * 10.0 + free * (1 - bias) * 0.5 + rng.random() * 0.1
            options.append((score, site))

        if not options:
            return None

        # Pick best with some randomness
        options.sort(key=lambda x: -x[0])
        k = min(len(options), max(1, int(3 * rng.random())))
        chosen = options[rng.randint(0, k - 1)][1]
        occupied[chosen] = i
        chain.append(chosen)

    return chain

def _local_search(sequence, chain, rng, deadline, max_iter=5000):
    """
    Fast local search: try moving each H residue to improve energy.
    Returns (best_chain, best_energy).
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    pos_map = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Identify H positions (excluding endpoints)
    h_indices = [i for i in range(1, n - 1) if sequence[i] == 'H']
    
    for iteration in range(max_iter):
        if time.time() >= deadline - 2.0:
            break
        
        improved = False
        rng.shuffle(h_indices)
        
        for i in h_indices:
            old_pos = current_chain[i]
            prev = current_chain[i - 1]
            nxt = current_chain[i + 1]
            
            for new_pos in _get_neighbours_fast(*old_pos):
                if new_pos == old_pos:
                    continue
                # Check unit step constraints
                d1 = abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) + abs(new_pos[2] - prev[2])
                d2 = abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) + abs(new_pos[2] - nxt[2])
                if d1 != 1 or d2 != 1:
                    continue
                if new_pos in pos_map:
                    continue

                # Compute delta energy
                delta = _delta_energy_fast(sequence, current_chain, pos_map, i, new_pos)
                
                if delta < 0:
                    # Accept the move
                    pos_map.pop(old_pos)
                    pos_map[new_pos] = i
                    current_chain[i] = new_pos
                    current_energy += delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return best_chain, best_energy

def _reversal_move(chain, a, b, pos_set):
    """
    Try a 2-opt reversal move: reverse the segment [a, b].
    Returns (new_chain, new_pos_set) or (None, None) if invalid.
    """
    n = len(chain)
    if a >= b:
        return None, None
    
    # The reversal is valid if:
    # 1. The new chain is self-avoiding.
    # 2. The bonds at the cut points (a-1, a) and (b, b+1) are preserved.
    #    Actually, the reversal changes the bonds at (a-1, a) and (b, b+1).
    #    The new bonds are (a-1, b) and (a, b+1).
    #    We need to check if these are unit steps.
    
    if a > 0:
        p_prev = chain[a - 1]
        p_new_a = chain[b]
        d1 = abs(p_prev[0] - p_new_a[0]) + abs(p_prev[1] - p_new_a[1]) + abs(p_prev[2] - p_new_a[2])
        if d1 != 1:
            return None, None
    
    if b < n - 1:
        p_next = chain[b + 1]
        p_new_b = chain[a]
        d2 = abs(p_next[0] - p_new_b[0]) + abs(p_next[1] - p_new_b[1]) + abs(p_next[2] - p_new_b[2])
        if d2 != 1:
            return None, None
    
    # Check self-avoidance of the new chain.
    # The segment [a, b] is reversed. The rest is unchanged.
    # We only need to check if the reversed segment collides with itself or the rest.
    # Since the original chain was self-avoiding, the only new collisions can be
    # between the reversed segment and the rest, or within the reversed segment
    # (but a reversal of a self-avoiding path is also self-avoiding).
    
    # Actually, let's just build the new chain and check.
    new_chain = chain[:a] + list(reversed(chain[a:b+1])) + chain[b+1:]
    
    # Check self-avoidance
    if len(set(new_chain)) != n:
        return None, None
    
    # Check connectivity
    for k in range(n - 1):
        d = abs(new_chain[k + 1][0] - new_chain[k][0]) + abs(new_chain[k + 1][1] - new_chain[k][1]) + abs(new_chain[k + 1][2] - new_chain[k][2])
        if d != 1:
            return None, None
    
    new_pos_set = set(new_chain)
    return new_chain, new_pos_set

def _anneal(sequence, chain, rng, deadline):
    """
    Iterated Local Search (ILS) with 2-opt reversal moves.
    """
    n = len(chain)
    if n < 4:
        return chain

    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    current_chain = list(chain)
    current_energy = best_energy
    pos_set = set(current_chain)
    pos_map = {p: i for i, p in enumerate(current_chain)}

    T0 = max(1.0, abs(current_energy) * 0.5)
    T_min = 0.1
    margin = 2.0

    time_left = deadline - time.time()
    if time_left <= margin:
        return best_chain

    # Estimate steps based on time
    total_steps = int((time_left - margin) * 1000)
    total_steps = max(1000, min(total_steps, 20000))

    cooling_factor = (T_min / T0) ** (1.0 / total_steps)
    T = T0

    for step in range(total_steps):
        if time.time() >= deadline - margin:
            break
        T *= cooling_factor
        if T < T_min:
            T = T_min

        # Choose a random 2-opt move
        a = rng.randint(0, n - 2)
        b = rng.randint(a + 1, n - 1)
        
        # Bias towards shorter segments for more effective moves
        if rng.random() < 0.5:
            b = min(n - 1, a + rng.randint(1, max(1, n // 4)))

        new_chain, new_pos_set = _reversal_move(current_chain, a, b, pos_set)
        if new_chain is None:
            continue

        # Compute new energy
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy

        if delta <= 0 or rng.random() < math.exp(-delta / T):
            current_chain = new_chain
            pos_set = new_pos_set
            pos_map = {p: idx for idx, p in enumerate(current_chain)}
            current_energy = new_energy
            
            # Run a quick local search after a good move
            if delta < 0:
                current_chain, current_energy = _local_search(sequence, current_chain, rng, deadline, max_iter=100)
                pos_set = set(current_chain)
                pos_map = {p: idx for idx, p in enumerate(current_chain)}

            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)

    return best_chain

def fold(sequence, dimension):
    """Fold an HP sequence on the 3D cubic lattice."""
    deadline = time.time() + 225.0  # 225s budget with safety margin

    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0

    n = len(sequence)

    # Phase 1: Multiple growth starts with diverse biases
    biases = [0.5, 0.7, 0.9, 0.3, 0.8, 0.6, 1.0, 0.4, 0.75, 0.65]
    num_restarts = min(10, max(3, int(10.0 / max(1, n) * 100)))
    
    # Reserve time for growth and local search
    growth_deadline = time.time() + 0.2 * (deadline - time.time())
    
    for restart in range(num_restarts):
        if time.time() >= growth_deadline:
            break

        bias = biases[restart % len(biases)]
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue

        # Local search to polish
        local_chain, local_energy = _local_search(sequence, chain, rng, deadline, max_iter=2000)

        if local_energy < best_energy:
            best, best_energy = local_chain, local_energy

    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0

    # Phase 2: Anneal the best chain using ILS
    time_left = deadline - time.time()
    if time_left > 5:
        annealed = _anneal(sequence, best, rng, deadline)
        anneal_energy = _energy(sequence, annealed)
        if anneal_energy < best_energy:
            best, best_energy = annealed, anneal_energy

    # Phase 3: If time remains, try another growth + local search + anneal
    if time.time() < deadline - 10:
        chain2 = _grow(sequence, dimension, rng, bias=0.8)
        if chain2 is not None:
            local_chain2, local_energy2 = _local_search(sequence, chain2, rng, deadline, max_iter=1000)
            if local_energy2 < best_energy:
                best, best_energy = local_chain2, local_energy2
            
            if time.time() < deadline - 5:
                annealed2 = _anneal(sequence, best, rng, deadline)
                e2a = _energy(sequence, annealed2)
                if e2a < best_energy:
                    best, best_energy = annealed2, e2a

    # Phase 4: Final local search polish
    if time.time() < deadline - 2:
        final_chain, final_energy = _local_search(sequence, best, rng, deadline, max_iter=1000)
        if final_energy < best_energy:
            best, best_energy = final_chain, final_energy

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
