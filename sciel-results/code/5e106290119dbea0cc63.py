"""HP lattice protein folding with efficient local search and simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _neighbours(point):
    """Yield the 4 neighbours of a point on the square lattice."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))

def _energy(sequence, chain):
    """Compute the HP contact energy of a chain conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, rng, temperature=1.0):
    """Randomized greedy chain growth with temperature-based selection."""
    n = len(sequence)
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Calculate gain: number of new H-H contacts
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1
            # Count free neighbours
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Score with randomness
            score = gain * 5.0 + free * 0.1 - rng.random() * temperature
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Boltzmann-like selection for diversity
        if temperature > 0.1 and len(candidates) > 1:
            max_score = max(s for s, _ in candidates)
            weights = [math.exp((s - max_score) / (temperature * 0.5 + 0.01)) for s, _ in candidates]
            total = sum(weights)
            r = rng.random() * total
            cum = 0
            selected = candidates[-1][1]
            for (s, site), w in zip(candidates, weights):
                cum += w
                if r <= cum:
                    selected = site
                    break
        else:
            selected = max(candidates)[1]
        
        chain.append(selected)
        occupied[selected] = i
    
    return chain

def _build_occupied_set(chain):
    """Build a set of occupied positions from a chain."""
    return set(chain)

def _try_single_move(sequence, chain, occupied, i, new_pos):
    """Try moving residue i to new_pos. Returns (success, new_chain, new_occupied, energy_delta)."""
    n = len(sequence)
    old_pos = chain[i]
    
    # Check adjacency to neighbors
    if i > 0:
        if abs(new_pos[0] - chain[i-1][0]) + abs(new_pos[1] - chain[i-1][1]) != 1:
            return None
    if i < n - 1:
        if abs(new_pos[0] - chain[i+1][0]) + abs(new_pos[1] - chain[i+1][1]) != 1:
            return None
    
    # Check self-avoidance: new_pos must not be occupied (except by itself)
    if new_pos in occupied and new_pos != old_pos:
        return None
    
    # Compute energy delta
    old_energy = 0
    new_energy = 0
    
    # Remove old contacts involving residue i
    if sequence[i] == 'H':
        for q in _neighbours(old_pos):
            j = None
            for k, p in enumerate(chain):
                if p == q:
                    j = k
                    break
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                old_energy += 1
    
    # Add new contacts involving residue i at new_pos
    if sequence[i] == 'H':
        for q in _neighbours(new_pos):
            j = None
            for k, p in enumerate(chain):
                if p == q:
                    j = k
                    break
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                new_energy += 1
    
    energy_delta = -(new_energy - old_energy)
    
    new_chain = list(chain)
    new_chain[i] = new_pos
    new_occupied = occupied - {old_pos} | {new_pos}
    
    return (new_chain, new_occupied, energy_delta)

def _local_search_sa(sequence, chain, max_iters=2000, rng=None, time_limit=15.0):
    """Simulated annealing local search with single-residue moves."""
    if rng is None:
        rng = random.Random()
    
    n = len(sequence)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_occupied = _build_occupied_set(current_chain)
    current_energy = _energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    t0 = time.time()
    t_initial = 5.0
    t_final = 0.01
    cooling_rate = (t_final / t_initial) ** (1.0 / max(1, max_iters - 1))
    
    for it in range(max_iters):
        if time.time() - t0 > time_limit:
            break
        
        temperature = t_initial * (cooling_rate ** it)
        
        # Pick a random residue to move
        i = rng.randint(1, n - 2)
        old_pos = current_chain[i]
        
        # Try all 4 possible moves
        best_move = None
        best_delta = float('inf')
        
        for new_pos in _neighbours(old_pos):
            if new_pos == old_pos:
                continue
            result = _try_single_move(sequence, current_chain, current_occupied, i, new_pos)
            if result is None:
                continue
            _, _, delta = result
            if delta < best_delta:
                best_delta = delta
                best_move = result
        
        if best_move is None:
            continue
        
        new_chain, new_occupied, delta = best_move
        new_energy = current_energy + delta
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or (temperature > 0.001 and rng.random() < math.exp(-max(0, -delta) / max(temperature, 0.001))):
            current_chain = new_chain
            current_occupied = new_occupied
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
    
    return best_chain, best_energy

def _segment_reversal(sequence, chain, i, j, rng):
    """Try reversing the segment from i to j. Returns new chain or None."""
    n = len(sequence)
    if i >= j or i < 1 or j > n - 2:
        return None
    
    new_chain = list(chain)
    # Reverse positions i..j
    for k in range((j - i + 1) // 2):
        new_chain[i + k], new_chain[j - k] = new_chain[j - k], new_chain[i + k]
    
    # Check self-avoidance
    seen = set()
    for p in new_chain:
        if p in seen:
            return None
        seen.add(p)
    
    # Check unit steps
    for k in range(n - 1):
        if abs(new_chain[k][0] - new_chain[k+1][0]) + abs(new_chain[k][1] - new_chain[k+1][1]) != 1:
            return None
    
    return new_chain

def _polish(sequence, chain, rng, time_limit=5.0):
    """Final polish: deterministic hill-climbing on single moves."""
    n = len(sequence)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_occupied = _build_occupied_set(current_chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    t0 = time.time()
    improved = True
    while improved and time.time() - t0 < time_limit:
        improved = False
        for i in range(1, n - 1):
            if time.time() - t0 > time_limit:
                break
            old_pos = current_chain[i]
            for new_pos in _neighbours(old_pos):
                result = _try_single_move(sequence, current_chain, current_occupied, i, new_pos)
                if result is None:
                    continue
                new_chain, new_occupied, delta = result
                if delta < 0:
                    current_chain = new_chain
                    current_occupied = new_occupied
                    current_energy += delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    improved = True
                    break
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation."""
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin before 180s limit
    n = len(sequence)
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Multiple greedy growths with varying temperatures
    grows_done = 0
    while grows_done < 5000:
        if time.time() > deadline - 30.0:
            break
        
        temperature = 0.3 + rng.random() * 1.5
        chain = _greedy_grow(sequence, rng, temperature)
        if chain is not None:
            grows_done += 1
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
    
    # Phase 2: Local search on best chain
    if best_chain is not None and time.time() < deadline - 10.0:
        remaining = deadline - 10.0 - time.time()
        improved_chain, improved_energy = _local_search_sa(
            sequence, best_chain, max_iters=3000, rng=rng, time_limit=min(remaining, 20.0)
        )
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    # Phase 3: Polish
    if best_chain is not None and time.time() < deadline - 5.0:
        remaining = deadline - 5.0 - time.time()
        polished_chain, polished_energy = _polish(
            sequence, best_chain, rng, time_limit=min(remaining, 5.0)
        )
        if polished_energy < best_energy:
            best_chain = polished_chain
            best_energy = polished_energy
    
    # Fallback
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
