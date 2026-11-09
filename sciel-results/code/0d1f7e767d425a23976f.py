"""HP lattice protein folding: Compact greedy growth + Two-phase Simulated Annealing (Standard + Polish)."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
# Conservative deadline to ensure we return within limits. 
# Task says 120s CPU for all sequences. Evaluator allows 180s wall clock.
# We have 10 sequences. If we spend 11s each, that's 110s total.
# We will manage time per sequence inside fold().
_GLOBAL_DEADLINE = _GLOBAL_START + 115.0


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbours(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    n = len(chain)
    if n < 2:
        return 0
    
    where = {p: i for i, p in enumerate(chain)}
    h_positions = set()
    for i, c in enumerate(sequence):
        if c == "H":
            h_positions.add(i)
    
    if not h_positions:
        return 0

    contacts = 0
    # Iterate only over H positions to save time
    for i in h_positions:
        p = chain[i]
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and abs(i - j) > 1 and j in h_positions:
                contacts += 1
    return -contacts // 2


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


def _grow_compact(sequence, rng, compactness, contact_bias):
    """
    Greedy chain growth with a compactness parameter.
    Higher compactness biases toward filling space densely (fewer free neighbors).
    contact_bias biases toward making H-H contacts.
    Returns a valid self-avoiding chain or None.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        current = chain[-1]
        options = []
        for site in _neighbours(current):
            if site in occupied:
                continue
            
            score = 0.0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            score += contact_bias
                        elif j < i - 1:
                            score += contact_bias * 0.1 # Slight preference for any contact if H
                            
            # Compactness: prefer sites with fewer free neighbors (more enclosed)
            free_count = sum(1 for q in _neighbours(site) if q not in occupied)
            score -= compactness * free_count * 0.5
            
            # Add small noise for diversity
            score += rng.random() * 0.5
            
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option
        options.sort(key=lambda x: x[0], reverse=True)
        best_site = options[0][1]
        occupied[best_site] = i
        chain.append(best_site)
    
    return chain


def _anneal(sequence, initial_chain, time_limit, rng, polish=False):
    """
    Simulated Annealing starting from initial_chain.
    If polish=True, uses a tighter cooling schedule and shorter moves for fine-tuning.
    Returns best_chain, best_energy.
    """
    n = len(initial_chain)
    if n < 3:
        return initial_chain[:], _energy(sequence, initial_chain)
    
    current_chain = initial_chain[:]
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy
    
    if polish:
        T = 0.5
        cooling_rate = 0.998
        max_stall = 50
    else:
        T = 1.5
        cooling_rate = 0.9995
        max_stall = 150

    deadline = time.time() + time_limit
    
    stall_counter = 0
    
    while time.time() < deadline:
        if _remaining_time() < 1.0:
            break
            
        # Choose move type
        if polish:
            # For polish: prefer short reversals and small pivots near the end
            r = rng.random()
            if r < 0.6:
                # Short reversal: length 2-6
                seg_len = rng.randint(2, min(6, n - 1))
                i = rng.randint(0, n - 1 - seg_len)
                j = i + seg_len
                new_chain = _apply_reversal(current_chain, i, j)
                if new_chain is None:
                    continue
            else:
                # Small pivot near the end (last 15 residues or 10% of chain)
                pivot_range = max(1, min(15, n // 10))
                i = rng.randint(max(1, n - pivot_range), n - 2)
                angle = rng.choice([1, 3])
                new_chain = _apply_pivot(current_chain, i, angle)
                if new_chain is None:
                    continue
        else:
            # Standard mode: 50% Reversal, 50% Pivot
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
            stall_counter = 0
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
        else:
            stall_counter += 1
            
        T *= cooling_rate
        if T < 0.01:
            # Reheat if stuck? Or just break.
            # In polish mode, we want to converge, so break.
            # In standard mode, we might reheat if we have time and are stuck.
            if polish:
                break
            else:
                if stall_counter > 100 and rng.random() < 0.2:
                    T = 1.0 # Reheat
                else:
                    break
                    
        if stall_counter > max_stall:
            # Try a larger perturbation or break
            if polish:
                break
            else:
                # Occasionally try a random restart of the tail? 
                # For now, just continue with lower T
                pass
                
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
    # (contact_bias, compactness)
    bias_configs = [
        (1.0, 0.2),  # Balanced
        (1.5, 0.1),  # Contact focused
        (0.5, 0.8),  # Compactness focused
        (2.0, 0.3),  # Strong contact + some compactness
    ]
    
    initial_chains = []
    
    # Allocate time for growth: ~1.5 seconds per sequence
    growth_deadline = time.time() + 1.5
    
    for contact_bias, compactness in bias_configs:
        if time.time() > growth_deadline:
            break
        # Try a few growths per config
        for _ in range(3):
            if time.time() > growth_deadline:
                break
            chain = _grow_compact(sequence, rng, compactness, contact_bias)
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
    top_k = 4
    candidates = [c[1] for c in scored_chains[:top_k]]
    
    # Phase 2: Anneal each candidate
    # Allocate remaining time to annealing
    # Total time budget per sequence is roughly 11s (115s / 10 seqs)
    # We spent ~1.5s on growth. Remaining ~9.5s.
    # Split into Standard SA and Polish SA.
    # Standard SA: 60% of remaining
    # Polish SA: 40% of remaining
    
    time_remaining = _remaining_time()
    # Ensure we don't use more than our fair share
    # If this is the first sequence, we have plenty of time. If it's the last, we have less.
    # Simple heuristic: use min(time_remaining, 9.0) as our local budget to be safe.
    local_budget = min(time_remaining, 9.0)
    
    time_per_standard = local_budget * 0.6 / len(candidates)
    time_polish = local_budget * 0.4
    
    # Run Standard SA on top candidates
    for chain in candidates:
        if _remaining_time() < 1.0:
            break
            
        # Ensure we don't exceed local budget
        alloc_time = min(time_per_standard, _remaining_time() - 0.5)
        if alloc_time < 0.1:
            continue
            
        best_c, best_e = _anneal(sequence, chain, alloc_time, rng, polish=False)
        
        if best_e < best_energy:
            best_energy = best_e
            best_chain = best_c
            
    # Phase 3: Polish the best result found
    if best_chain is not None and _remaining_time() > 1.0:
        # Run Polish SA on the best chain
        # Use a fixed fraction of the polish time, or just the remaining time minus safety margin
        polish_time = min(time_polish, _remaining_time() - 0.5)
        if polish_time > 0.1:
            best_c, best_e = _anneal(sequence, best_chain, polish_time, rng, polish=True)
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
