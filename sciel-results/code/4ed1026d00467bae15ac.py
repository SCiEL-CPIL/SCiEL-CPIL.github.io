"""HP lattice protein folding on the cubic lattice.

Strategy:
  1. Multi-start growth with diverse bias parameters and lookahead.
  2. Simulated annealing with Flip, Pivot, and Reversal moves.
  3. Efficient delta-energy computation using position maps.
  4. Multiple annealing cycles with restarts to escape local minima.
  5. Contact-map local search for final polishing.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

# Pre-computed neighbor offsets for 3D cubic lattice
_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _get_neighbours_fast(x, y, z):
    """Fast neighbor generation using precomputed offsets."""
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
    """Fast delta energy computation using precomputed position map."""
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

def _grow(sequence, dimension, rng, bias=0.7, lookahead=1):
    """Randomized greedy chain growth with tunable contact bias and lookahead."""
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
            
            # Lookahead: check if next residue can be placed
            if lookahead > 0 and i < n - 1:
                lookahead_score = 0
                # Heuristic: prefer sites that leave more options for the next step
                # This is a simplified lookahead
                for q in _get_neighbours_fast(*site):
                    if q not in occupied and q != last:
                        lookahead_score += 1
                # If no options for next step, penalize heavily
                if lookahead_score == 0:
                    gain -= 100 
            
            # Mix contact gain and free space with tunable bias
            # Higher bias favors contacts, lower favors free space
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

def _apply_pivot(chain, i, axis, sign):
    """
    Apply a pivot move to the chain at index i.
    Rotates the sub-chain from i+1 to end around the pivot at i.
    Returns the new chain or None if invalid (collision).
    """
    n = len(chain)
    if i >= n - 1:
        return chain
    
    pivot = chain[i]
    # The sub-chain to rotate
    sub_chain = chain[i+1:]
    
    # Precompute relative positions
    rel_positions = []
    for pos in sub_chain:
        rel = (pos[0] - pivot[0], pos[1] - pivot[1], pos[2] - pivot[2])
        rel_positions.append(rel)
    
    # Apply rotation
    new_rel_positions = []
    for rel in rel_positions:
        x, y, z = rel
        if axis == 0: # x-axis
            if sign == 1: # CW: (x, y, z) -> (x, z, -y)
                new_rel = (x, z, -y)
            else: # CCW: (x, y, z) -> (x, -z, y)
                new_rel = (x, -z, y)
        elif axis == 1: # y-axis
            if sign == 1: # CW: (x, y, z) -> (z, y, -x)
                new_rel = (z, y, -x)
            else: # CCW: (x, y, z) -> (-z, y, x)
                new_rel = (-z, y, x)
        else: # z-axis
            if sign == 1: # CW: (x, y, z) -> (y, -x, z)
                new_rel = (y, -x, z)
            else: # CCW: (x, y, z) -> (-y, x, z)
                new_rel = (-y, x, z)
        new_rel_positions.append(new_rel)
    
    # Convert back to absolute positions
    new_sub_chain = []
    for rel in new_rel_positions:
        new_pos = (pivot[0] + rel[0], pivot[1] + rel[1], pivot[2] + rel[2])
        new_sub_chain.append(new_pos)
    
    new_chain = chain[:i+1] + new_sub_chain
    
    # Check for self-avoidance
    # 1. Check against fixed part (0..i)
    fixed_set = set(chain[:i+1])
    for pos in new_sub_chain:
        if pos in fixed_set:
            return None
            
    # 2. Check within new sub-chain
    if len(set(new_sub_chain)) != len(new_sub_chain):
        return None
        
    return new_chain

def _local_search_contact_map(sequence, chain, rng, max_iter=5000):
    """
    Contact-map-based local search: iteratively try to improve H-H contacts
    by moving H residues to positions that create new contacts.
    """
    n = len(chain)
    if n < 4:
        return chain
    
    current_chain = list(chain)
    pos_map = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Identify H positions
    h_indices = [i for i in range(n) if sequence[i] == "H"]
    
    for iteration in range(max_iter):
        improved = False
        
        # Try moving each H residue to a better position
        for i in h_indices:
            if i == 0 or i == n - 1:
                continue  # Skip endpoints
            
            old_pos = current_chain[i]
            prev = current_chain[i - 1]
            nxt = current_chain[i + 1]
            
            # Try all valid neighbor positions
            for new_pos in _get_neighbours_fast(*old_pos):
                # Check unit steps
                d1 = abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) + abs(new_pos[2] - prev[2])
                d2 = abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) + abs(new_pos[2] - nxt[2])
                if d1 != 1 or d2 != 1:
                    continue
                
                # Check self-avoidance
                if new_pos in pos_map:
                    continue
                
                # Compute delta
                delta = _delta_energy_fast(sequence, current_chain, pos_map, i, new_pos)
                
                if delta < 0:  # Energy improves
                    # Accept immediately
                    pos_map.pop(old_pos)
                    pos_map[new_pos] = i
                    current_chain[i] = new_pos
                    current_energy += delta
                    improved = True
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    break  # Move to next H residue
        
        if not improved:
            break
    
    return best_chain

def _anneal(sequence, chain, dimension, rng, deadline):
    """Simulated annealing with Flip, Pivot, and Reversal moves."""
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
    T_min = 0.01
    margin = 5.0

    time_left = deadline - time.time()
    if time_left <= margin:
        return best_chain

    # Estimate steps based on time
    # Flip moves are fast, Pivot moves are slower
    total_steps = int((time_left - margin) * 2000)
    total_steps = max(1000, min(total_steps, 50000))

    cooling_factor = (T_min / T0) ** (1.0 / total_steps)
    T = T0

    for step in range(total_steps):
        if time.time() >= deadline - margin:
            break
        T *= cooling_factor
        if T < T_min:
            T = T_min

        move_type = rng.random()

        if move_type < 0.5:
            # Flip: move one residue to a random valid neighbor
            i = rng.randint(1, n - 2)
            old_pos = current_chain[i]
            prev = current_chain[i - 1]
            nxt = current_chain[i + 1]

            valid_options = []
            for new_pos in _get_neighbours_fast(*old_pos):
                d1 = abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) + abs(new_pos[2] - prev[2])
                d2 = abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) + abs(new_pos[2] - nxt[2])
                if d1 != 1 or d2 != 1:
                    continue
                if new_pos == old_pos:
                    continue
                if new_pos in pos_set:
                    continue
                valid_options.append(new_pos)

            if not valid_options:
                continue

            new_pos = rng.choice(valid_options)

            # Compute delta energy efficiently
            delta = _delta_energy_fast(sequence, current_chain, pos_map, i, new_pos)

            if delta <= 0 or rng.random() < math.exp(-delta / T):
                pos_set.discard(old_pos)
                pos_set.add(new_pos)
                pos_map.pop(old_pos)
                pos_map[new_pos] = i
                current_chain[i] = new_pos
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)

        elif move_type < 0.8:
            # Pivot move
            i = rng.randint(1, n - 3)
            axis = rng.randint(0, 2)
            sign = rng.choice([1, -1])
            
            new_chain = _apply_pivot(current_chain, i, axis, sign)
            if new_chain is None:
                continue
                
            # Check connectivity at the pivot
            p_i = new_chain[i]
            p_ip1 = new_chain[i+1]
            dist = abs(p_i[0]-p_ip1[0]) + abs(p_i[1]-p_ip1[1]) + abs(p_i[2]-p_ip1[2])
            if dist != 1:
                continue
            
            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy

            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_chain = new_chain
                pos_set = set(current_chain)
                pos_map = {p: idx for idx, p in enumerate(current_chain)}
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)

        else:
            # Reversal
            a = rng.randint(0, n - 2)
            b = rng.randint(a + 1, n - 1)

            new_chain = list(current_chain)
            new_chain[a:b + 1] = reversed(new_chain[a:b + 1])

            if len(set(new_chain)) != n:
                continue

            valid = True
            for k in range(n - 1):
                d = abs(new_chain[k + 1][0] - new_chain[k][0]) + abs(new_chain[k + 1][1] - new_chain[k][1]) + abs(new_chain[k + 1][2] - new_chain[k][2])
                if d != 1:
                    valid = False
                    break
            if not valid:
                continue

            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy

            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_chain = new_chain
                pos_set = set(current_chain)
                pos_map = {p: idx for idx, p in enumerate(current_chain)}
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)

    return best_chain

def fold(sequence, dimension):
    """Fold an HP sequence on the 3D cubic lattice."""
    deadline = time.time() + 230.0  # 230s budget with safety margin

    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0

    n = len(sequence)

    # Phase 1: Multiple growth starts with diverse biases
    biases = [0.5, 0.7, 0.9, 0.3, 0.8, 0.6, 1.0, 0.4, 0.75, 0.65]
    # Allocate time for growth and local search
    # If n is small, we can do more restarts.
    num_restarts = min(20, max(5, int(30.0 / max(1, n) * 150)))
    
    # Reserve time for annealing
    # Let's say we spend 30% of time on growth/local search, 70% on annealing
    growth_deadline = time.time() + 0.3 * (deadline - time.time())
    
    for restart in range(num_restarts):
        if time.time() >= growth_deadline:
            break

        bias = biases[restart % len(biases)]
        chain = _grow(sequence, dimension, rng, bias=bias, lookahead=1)
        if chain is None:
            continue

        # Local search to polish using contact map
        local_chain = _local_search_contact_map(sequence, chain, rng, max_iter=2000)
        local_energy = _energy(sequence, local_chain)

        if local_energy < best_energy:
            best, best_energy = local_chain, local_energy

    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0

    # Phase 2: Anneal the best chain
    time_left = deadline - time.time()
    if time_left > 10:
        annealed = _anneal(sequence, best, dimension, rng, deadline)
        anneal_energy = _energy(sequence, annealed)
        if anneal_energy < best_energy:
            best, best_energy = annealed, anneal_energy

    # Phase 3: Final local search polish using contact map
    if time.time() < deadline - 5:
        final_chain = _local_search_contact_map(sequence, best, rng, max_iter=5000)
        final_energy = _energy(sequence, final_chain)
        if final_energy < best_energy:
            best, best_energy = final_chain, final_energy

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
