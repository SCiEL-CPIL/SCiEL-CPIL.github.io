"""HP lattice protein folding: diverse greedy growth + simulated annealing with reversals and pivots."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 115.0


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbours(point):
    x, y = point
    for dx, dy in _DELTAS:
        yield (x + dx, y + dy)


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    if len(chain) < 2:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts // 2


def _grow(sequence, dimension, rng, bias_hh, bias_free):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Score: weighted sum of contacts and free space, plus noise
            score = bias_hh * gain + bias_free * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        # Pick the best option
        best_score, best_site = max(options)
        occupied[best_site] = i
        chain.append(best_site)
    return chain


def _is_valid_chain(chain):
    """Check self-avoidance."""
    return len(set(chain)) == len(chain)


def _apply_reversal(chain, i, j):
    """
    Attempt a reversal move. Returns new chain if valid, else None.
    i, j are indices. 0 <= i < j < len(chain).
    """
    n = len(chain)
    if i >= j:
        return None
    
    # Check boundary conditions
    # Left cut: chain[i-1] must be adjacent to chain[j]
    if i > 0:
        p_prev = chain[i - 1]
        p_end = chain[j]
        if abs(p_prev[0] - p_end[0]) + abs(p_prev[1] - p_end[1]) != 1:
            return None
            
    # Right cut: chain[i] must be adjacent to chain[j+1]
    if j < n - 1:
        p_start = chain[i]
        p_next = chain[j + 1]
        if abs(p_start[0] - p_next[0]) + abs(p_start[1] - p_next[1]) != 1:
            return None
            
    # Construct new chain
    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
    
    # Check self-avoidance
    if not _is_valid_chain(new_chain):
        return None
        
    return new_chain


def _apply_pivot(chain, i, angle):
    """
    Attempt a pivot move. Rotate tail chain[i+1:] around chain[i] by 'angle' (1, 2, or 3 * 90 deg).
    Returns new chain if valid, else None.
    """
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return None
        
    pivot = chain[i]
    tail = chain[i+1:]
    prefix = chain[:i+1]
    
    # Rotate tail
    rotated_tail = []
    for p in tail:
        x, y = p[0] - pivot[0], p[1] - pivot[1]
        for _ in range(angle):
            x, y = -y, x
        rotated_tail.append((pivot[0] + x, pivot[1] + y))
        
    new_chain = prefix + rotated_tail
    
    # Check self-avoidance
    if not _is_valid_chain(new_chain):
        return None
        
    return new_chain


def _anneal(sequence, initial_chain, time_limit, rng):
    """
    Simulated Annealing starting from initial_chain.
    Returns best_chain, best_energy.
    """
    n = len(initial_chain)
    current_chain = initial_chain[:]
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy
    
    T = 1.0
    cooling_rate = 0.9995
    
    deadline = time.time() + time_limit
    
    while time.time() < deadline:
        if _remaining_time() < 1.0:
            break
            
        # Choose move type: 50% Reversal, 50% Pivot
        if rng.random() < 0.5:
            # Reversal
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            new_chain = _apply_reversal(current_chain, i, j)
            if new_chain is None:
                continue
        else:
            # Pivot
            i = rng.randint(1, n - 2)
            angle = rng.choice([1, 2, 3])
            new_chain = _apply_pivot(current_chain, i, angle)
            if new_chain is None:
                continue
        
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Metropolis criterion
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 0.01)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
                
        T *= cooling_rate
        if T < 0.01:
            # Reheat if stuck? Or just break.
            # Let's just break and rely on multiple restarts
            break
            
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    if n < 2:
        return [[0, 0]] * n
        
    best_chain = None
    best_energy = 0 # Energy is negative, so 0 is worse than any contact
    
    # Phase 1: Generate diverse initial chains
    # Use different bias parameters to explore different regions of the solution space
    bias_configs = [
        (1.0, 0.1),  # Strong HH bias
        (0.5, 0.5),  # Balanced
        (0.1, 1.0),  # Strong Free space bias (less compact, maybe easier to fold later?)
        (2.0, 0.0),  # Very strong HH bias
    ]
    
    initial_chains = []
    time_for_growth = 2.0 # Spend 2s on growth
    
    # Distribute growth time
    growths_per_config = max(1, int(time_for_growth / len(bias_configs) / 0.01)) # Rough estimate
    
    for bias_hh, bias_free in bias_configs:
        if _remaining_time() < 5.0:
            break
        for _ in range(growths_per_config):
            if _remaining_time() < 5.0:
                break
            chain = _grow(sequence, dimension, rng, bias_hh, bias_free)
            if chain is not None:
                initial_chains.append(chain)
                
    # If no chains generated, fallback to simple straight line (should not happen)
    if not initial_chains:
        initial_chains.append([(i, 0) for i in range(n)])
        
    # Evaluate and sort initial chains
    scored_chains = []
    for chain in initial_chains:
        e = _energy(sequence, chain)
        scored_chains.append((e, chain))
        
    # Sort by energy (most negative first)
    scored_chains.sort(key=lambda x: x[0])
    
    # Keep top K chains for annealing
    top_k = 5
    candidates = [c[1] for c in scored_chains[:top_k]]
    
    # Phase 2: Anneal each candidate
    # Allocate remaining time to annealing
    total_time_remaining = _remaining_time()
    time_per_candidate = total_time_remaining / len(candidates)
    
    for chain in candidates:
        if _remaining_time() < 1.0:
            break
            
        # Ensure we don't exceed global deadline
        alloc_time = min(time_per_candidate, _remaining_time() - 0.5)
        if alloc_time < 0.1:
            continue
            
        best_c, best_e = _anneal(sequence, chain, alloc_time, rng)
        
        if best_e < best_energy:
            best_energy = best_e
            best_chain = best_c
            
    # Fallback if something went wrong
    if best_chain is None:
        if initial_chains:
            best_chain = initial_chains[0]
        else:
            best_chain = [(i, 0) for i in range(n)]
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
