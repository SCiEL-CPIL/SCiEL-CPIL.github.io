import random
import time
import math

# Global time management
_START_TIME = None

def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    # 180s limit + safety margin. Let's aim for 170s total usage.
    return 170.0 - (time.time() - _START_TIME)

def _neighbors(p):
    """Yield 4 neighbors of a 2D point."""
    return ((p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1))

def _initial_saw(sequence, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    # Start at 0,0
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        # Get valid neighbors
        candidates = []
        for nb in _neighbors(last):
            if nb not in occupied:
                candidates.append(nb)
        
        if not candidates:
            # Dead end
            return None
        
        # Randomly pick a candidate
        nb = rng.choice(candidates)
        chain.append(nb)
        occupied[nb] = i
        
    return chain

def _energy(sequence, chain):
    """Calculate energy of a conformation."""
    n = len(sequence)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    
    # Only H residues contribute
    h_positions = [i for i in range(n) if sequence[i] == 'H']
    
    for i in h_positions:
        p = chain[i]
        for nb in _neighbors(p):
            j = pos_to_idx.get(nb)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
                
    return -contacts

def _local_search(sequence, initial_chain, time_budget, rng):
    """
    Perform Simulated Annealing local search using pivot moves.
    Returns the best chain found.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n < 3:
        return list(initial_chain)
        
    # State
    chain = list(initial_chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Precompute H indices for potential optimization (not used in current simple energy calc)
    
    start_time = time.time()
    iterations = 0
    # We will run until time budget is exhausted
    # Estimate: ~10000-50000 iters/sec depending on N.
    
    start_temp = 5.0
    end_temp = 0.1
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
            
        iterations += 1
        
        # Cool down based on time elapsed to be robust against speed variations
        # But we don't know total time ahead. 
        # Let's use a fixed number of "cooling steps" or just linear time based.
        # If we assume we run for ~time_budget, we can cool linearly.
        # However, if we break early, we are still hot.
        # Better: Cool based on iterations, but cap iterations.
        # Or, just use a constant low temp after a while?
        # Standard SA: T = T0 * exp(-alpha * t)
        
        # Let's use a simple linear cooling based on a assumed max iterations.
        # If we run 100k iterations in 10s, that's 10k/s.
        # Let's just use a fixed temp that decreases slowly.
        
        # Actually, for HP models, a simple "random walk" that accepts downhill and 
        # occasionally uphill works well.
        # Let's use T = 1.0 constant? Or decreasing?
        # Decreasing is better.
        
        # Let's assume max 200k iterations.
        max_iters_est = 200000
        if iterations > max_iters_est:
            break
            
        progress = iterations / max_iters_est
        temp = start_temp - (start_temp - end_temp) * progress
        if temp < end_temp:
            temp = end_temp
            
        # Perform a pivot move
        k = rng.randint(1, n - 1)
        pivot = chain[k - 1]
        old_tail_start = chain[k]
        
        # Find valid new positions for chain[k]
        # Must be neighbor of pivot, not old_tail_start
        # And not in head (0..k-2)
        
        candidates = []
        # Optimization: Precompute head set? 
        # Head is chain[0:k-1].
        # Checking collision is O(k).
        
        for nb in _neighbors(pivot):
            if nb == old_tail_start:
                continue
            # Check collision with head
            collision = False
            # Only check head 0..k-2
            for i in range(k - 1):
                if chain[i] == nb:
                    collision = True
                    break
            if not collision:
                candidates.append(nb)
                
        if not candidates:
            continue
            
        new_tail_start = rng.choice(candidates)
        
        # Calculate delta
        delta_x = new_tail_start[0] - old_tail_start[0]
        delta_y = new_tail_start[1] - old_tail_start[1]
        
        # Check if entire tail collides with head
        valid = True
        new_tail = [None] * (n - k)
        
        for i in range(k, n):
            old_pos = chain[i]
            new_pos = (old_pos[0] + delta_x, old_pos[1] + delta_y)
            new_tail[i - k] = new_pos
            
            # Check collision with head for i > k
            # (i == k is already checked via candidates)
            if i > k:
                for j in range(k - 1):
                    if chain[j] == new_pos:
                        valid = False
                        break
                if not valid:
                    break
                    
        if not valid:
            continue
            
        # Calculate Energy
        # Construct new chain
        # new_chain = chain[:k] + new_tail
        
        # To avoid list concatenation cost, we can calculate energy on the fly?
        # No, _energy expects a list.
        # List concatenation is O(N).
        new_chain = chain[:k] + new_tail
        
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Metropolis Accept
        if delta_e < 0:
            accept = True
        else:
            if temp > 0:
                # Avoid overflow/underflow in exp
                if delta_e / temp < 700: 
                    accept = (rng.random() < math.exp(-delta_e / temp))
                else:
                    accept = False
            else:
                accept = False
                
        if accept:
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)

    return best_chain

def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    if dimension != 2:
        raise ValueError("Only dimension 2 is supported")
        
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
        
    rng = random.Random(12345 + len(sequence))
    
    # Time budget per sequence
    # Total limit 210s. 10 sequences.
    # Allocate 15s per sequence.
    time_per_seq = 15.0
    
    best_chain = None
    best_energy = 0 # Worst case
    
    start_time = time.time()
    
    # Restart loop
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_per_seq - 1.0:
            break
            
        # Generate initial SAW
        init_chain = None
        for _ in range(20):
            init_chain = _initial_saw(sequence, rng)
            if init_chain:
                break
        
        if not init_chain:
            init_chain = [(i, 0) for i in range(n)]
            
        # Run local search
        run_time = time_per_seq - (time.time() - start_time) - 0.5
        if run_time < 1.0:
            break
            
        result_chain = _local_search(sequence, init_chain, run_time, rng)
        
        if result_chain:
            e = _energy(sequence, result_chain)
            if best_chain is None or e < best_energy:
                best_chain = result_chain
                best_energy = e
                
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
