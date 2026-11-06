"""
HP lattice protein folding using optimized Monte Carlo search.
Implements Kink moves and Tail Regeneration (Pivot) moves with Simulated Annealing.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Safety margin: 210s limit, leave 10s buffer -> 200s usable
    return 200.0 - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """Calculate energy: -number of H-H contacts (non-bonded neighbors)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(sequence)
    is_h = [c == 'H' for c in sequence]
    for i, p in enumerate(chain):
        if is_h[i]:
            # Check 4 neighbors
            x, y = p
            # Right
            q = (x + 1, y)
            j = where.get(q)
            if j is not None and j > i + 1 and is_h[j]:
                contacts += 1
            # Left
            q = (x - 1, y)
            j = where.get(q)
            if j is not None and j > i + 1 and is_h[j]:
                contacts += 1
            # Up
            q = (x, y + 1)
            j = where.get(q)
            if j is not None and j > i + 1 and is_h[j]:
                contacts += 1
            # Down
            q = (x, y - 1)
            j = where.get(q)
            if j is not None and j > i + 1 and is_h[j]:
                contacts += 1
    return -contacts

def _generate_random_saw(n, rng, max_attempts=50):
    """Generate a random self-avoiding walk of length n on the square lattice."""
    for _ in range(max_attempts):
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        success = True
        for i in range(1, n):
            neighbors = list(_neighbours(chain[-1]))
            rng.shuffle(neighbors)
            placed = False
            for site in neighbors:
                if site not in occupied:
                    chain.append(site)
                    occupied[site] = i
                    placed = True
                    break
            if not placed:
                success = False
                break
        if success and len(chain) == n:
            return chain
    return None

def _kink_move(chain, rng):
    """
    Attempt a kink move: move two consecutive residues i and i+1.
    Returns new_chain or None.
    """
    n = len(chain)
    if n < 4:
        return None
    
    # Pick i such that i-1 and i+2 exist
    i = rng.randint(1, n - 3)
    
    p_prev = chain[i - 1]
    p_curr = chain[i]
    p_next = chain[i + 1]
    p_after = chain[i + 2]
    
    # New positions must be neighbors of p_prev and p_after
    # and not occupied by any other part of the chain (except the ones being moved)
    
    # Get all neighbors of p_prev
    new_p_curr_candidates = list(_neighbours(p_prev))
    
    for new_p_curr in new_p_curr_candidates:
        # Must not be p_after (would make length 0 or overlap)
        if new_p_curr == p_after:
            continue
        # Must not be p_curr (no move)
        if new_p_curr == p_curr:
            continue
            
        # Get neighbors of new_p_curr
        new_p_next_candidates = list(_neighbours(new_p_curr))
        
        for new_p_next in new_p_next_candidates:
            # Must be adjacent to p_after
            if abs(new_p_next[0] - p_after[0]) + abs(new_p_next[1] - p_after[1]) != 1:
                continue
            # Must not be p_curr
            if new_p_next == p_curr:
                continue
            # Must not be p_next (no move)
            if new_p_next == p_next:
                continue
            
            # Check self-avoidance
            # The rest of the chain is chain[:i] + chain[i+2:]
            # We need to check if new_p_curr or new_p_next are in that set
            # Optimization: use a set for occupied positions
            # For speed, we can just check against the list if n is small, but set is better.
            # However, creating a set every move is O(N). 
            # Let's try to avoid full set creation if possible or accept it for simplicity/robustness.
            # Given N<=100, set creation is fast.
            
            occupied = set(chain)
            occupied.discard(p_curr)
            occupied.discard(p_next)
            
            if new_p_curr in occupied or new_p_next in occupied:
                continue
                
            # Valid move
            new_chain = chain.copy()
            new_chain[i] = new_p_curr
            new_chain[i + 1] = new_p_next
            return new_chain
            
    return None

def _tail_regeneration(chain, rng, max_steps=100):
    """
    Attempt a tail regeneration move (pivot at i).
    Regenerates chain[i+1:] as a random SAW avoiding chain[:i+1].
    """
    n = len(chain)
    if n < 5:
        return None
    
    # Choose pivot index i
    # Bias towards earlier pivots for larger rearrangements? 
    # Or random. Let's pick random i in [0, n-5] to leave some tail.
    i = rng.randint(0, n - 5)
    
    pivot_pos = chain[i]
    fixed_head = set(chain[:i + 1])
    
    # Generate new tail starting from pivot_pos
    new_tail = []
    current = pivot_pos
    
    # The first step of the new tail must be a neighbor of pivot_pos
    # and not in fixed_head.
    # We also need to ensure we can complete the tail of length n - i - 1.
    
    # Randomly try to generate the rest of the chain
    remaining_len = n - i - 1
    if remaining_len == 0:
        return None
        
    # Try a few random paths for the tail
    for _ in range(max_steps):
        # Shuffle neighbors of pivot
        neighbors = list(_neighbours(pivot_pos))
        rng.shuffle(neighbors)
        
        valid_path = True
        path = []
        occupied = set(fixed_head)
        
        for step in range(remaining_len):
            # Get neighbors of current
            if step == 0:
                opts = neighbors
            else:
                opts = list(_neighbours(current))
                rng.shuffle(opts)
            
            placed = False
            for nb in opts:
                if nb in occupied:
                    continue
                # If this is the last step, we don't have a constraint to connect to anything else,
                # just need to place it.
                path.append(nb)
                occupied.add(nb)
                current = nb
                placed = True
                break
            
            if not placed:
                valid_path = False
                break
        
        if valid_path and len(path) == remaining_len:
            new_chain = chain[:i + 1] + path
            # Check if it's actually different
            if new_chain != chain:
                return new_chain
            
    return None

def _local_search(sequence, chain, rng, max_iters, start_temp=5.0, end_temp=0.1):
    """
    Simulated Annealing local search.
    """
    n = len(chain)
    current = list(chain)
    current_energy = _energy(sequence, current)
    
    # Precompute is_h
    is_h = [c == 'H' for c in sequence]
    
    for it in range(max_iters):
        if _time_remaining() < 5:
            break
        
        # Annealing schedule
        progress = it / max(1, max_iters)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Choose move type
        # Kinks are cheap, Tail Regen is expensive but powerful
        if rng.random() < 0.6:
            new_chain = _kink_move(current, rng)
        else:
            new_chain = _tail_regeneration(current, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta/temp)
        if delta < 0:
            accept = True
        else:
            if temp > 0.01:
                prob = math.exp(-delta / temp)
                accept = rng.random() < prob
            else:
                accept = False
        
        if accept:
            current = new_chain
            current_energy = new_energy
    
    return current

def fold(sequence, dimension):
    """
    Fold the sequence using fragment-based Monte Carlo search with
    random SAW initialization and segment moves.
    """
    _init_time()
    n = len(sequence)
    
    # Determine search parameters based on sequence length
    # Longer sequences need more time per restart but fewer restarts
    if n <= 20:
        num_restarts = 100
        iters_per_restart = 2000
    elif n <= 50:
        num_restarts = 50
        iters_per_restart = 1000
    else:
        num_restarts = 30
        iters_per_restart = 500
    
    best_chain = None
    best_energy = 0  # 0 is neutral, we want negative (lower is better)
    
    for restart in range(num_restarts):
        if _time_remaining() < 10:
            break
        
        rng = random.Random(12345 + len(sequence) + restart * 7919)
        
        # Initialize with random SAW
        chain = _generate_random_saw(n, rng)
        if chain is None:
            # Fallback: simple greedy growth
            chain = [(0, 0)]
            occupied = {(0, 0): 0}
            for i in range(1, n):
                neighbors = list(_neighbours(chain[-1]))
                rng.shuffle(neighbors)
                placed = False
                for site in neighbors:
                    if site not in occupied:
                        chain.append(site)
                        occupied[site] = i
                        placed = True
                        break
                if not placed:
                    # Try to backtrack? For simplicity, just break and use whatever we have?
                    # No, we need a valid chain. If greedy fails, try another restart.
                    break
            if len(chain) < n:
                continue
        
        # Refine with local search
        refined_chain = _local_search(sequence, chain, rng, max_iters=iters_per_restart)
        
        energy = _energy(sequence, refined_chain)
        if best_chain is None or energy < best_energy:
            best_chain = refined_chain
            best_energy = energy
    
    if best_chain is None:
        # Final fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
