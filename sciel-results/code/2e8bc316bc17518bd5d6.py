"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(order, P):
    """
    Compute makespan for a given job order on processing_times matrix P.
    Optimized for speed by minimizing Python overhead in the inner loop.
    """
    if len(order) == 0:
        return 0.0
    
    n_machines = P.shape[1]
    n_jobs = len(order)
    
    # c_machines[m] holds the completion time of the last job processed on machine m
    # We initialize to 0
    c_machines = [0.0] * n_machines
    
    last_completion = 0.0
    
    # Pre-extract rows for faster access? P[j] is usually fast enough, 
    # but let's ensure we are using local variables.
    for j in order:
        row = P[j]
        c_curr = 0.0
        # Loop over machines
        # c_curr = max(c_machines[m], c_curr) + row[m]
        for m in range(n_machines):
            if c_machines[m] > c_curr:
                c_curr = c_machines[m]
            c_curr += row[m]
            c_machines[m] = c_curr
        last_completion = c_curr
        
    return last_completion

def _neh_order(processing_times, rng=None):
    """
    NEH heuristic: order jobs by decreasing total processing time.
    If rng is provided, add slight noise to break ties or perturb order for diversity.
    """
    totals = processing_times.sum(axis=1)
    
    # Add small noise if rng is provided to create diverse initial solutions
    if rng is not None:
        noise = rng.uniform(0, 0.1 * np.max(totals), size=len(totals))
        totals = totals + noise
        
    return list(np.argsort(-totals))

def _apply_vnd(processing_times, order, time_limit=None, deadline=None):
    """
    Variable Neighborhood Descent.
    Neighborhoods:
    1. Or-opt moves (move a block of size 1, 2, or 3 to a different position)
    2. Adjacent swaps
    """
    n_jobs = len(order)
    if n_jobs <= 1:
        return order
    
    current_order = list(order)
    current_ms = _makespan(current_order, processing_times)
    
    improved = True
    start_time = time.time()
    
    while improved:
        if time_limit and (time.time() - start_time) > time_limit:
            break
        if deadline and time.time() > deadline:
            break
            
        improved = False
        n_jobs = len(current_order)
        
        # Neighborhood 1: Or-opt moves
        # Move a block of size k (1, 2, 3) from position i to position j
        for k in [1, 2, 3]:
            if n_jobs < k:
                continue
            for i in range(n_jobs - k + 1):
                if deadline and time.time() > deadline:
                    return current_order
                # Extract block
                block = current_order[i:i+k]
                # Remove block
                remaining = current_order[:i] + current_order[i+k:]
                # Try inserting block at all positions in remaining
                for j in range(len(remaining) + 1):
                    # Skip if it's the same position (i) or adjacent in a way that results in same order
                    # Inserting at j in remaining corresponds to:
                    # If j <= i: new pos is j
                    # If j > i: new pos is j + k (because we removed k elements before j if j was after i? No, remaining is shorter)
                    # Actually, let's just construct and check.
                    # Optimization: skip if j == i (same place) or j == i+k (same place after removal)
                    # In 'remaining', the index i is where the block was.
                    # If we insert at i, it goes back.
                    # If we insert at i+k, it goes back (since we removed k items, the gap is closed).
                    # Wait, if we remove k items at i, the item at i+k moves to i.
                    # So inserting at i or i+k in 'remaining' restores original order.
                    
                    if j == i or j == i + k:
                        continue
                        
                    candidate = remaining[:j] + block + remaining[j:]
                    ms = _makespan(candidate, processing_times)
                    if ms < current_ms:
                        current_order = candidate
                        current_ms = ms
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
                
        if improved:
            continue
            
        # Neighborhood 2: Adjacent Swaps
        for i in range(n_jobs - 1):
            if deadline and time.time() > deadline:
                return current_order
            # Swap i and i+1
            candidate = current_order.copy()
            candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
            ms = _makespan(candidate, processing_times)
            if ms < current_ms:
                current_order = candidate
                current_ms = ms
                improved = True
                break
                
    return current_order

def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on a flow-shop instance."""
    start_time = time.time()
    # Safety margin: 150s limit, stop at 145s to ensure return
    deadline = start_time + 145.0
    
    processing_times = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = np.random.default_rng(42)
    
    # 1. Initial Solution: NEH
    best_order = _neh_order(processing_times)
    best_ms = _makespan(best_order, processing_times)
    
    # Polish the initial solution with VND
    polished = _apply_vnd(processing_times, best_order, deadline=deadline)
    polished_ms = _makespan(polished, processing_times)
    
    if polished_ms < best_ms:
        best_order = polished
        best_ms = polished_ms
        
    # 2. Multi-start / Perturbation Loop
    # While time remains, perturb the best solution and polish it
    perturb_count = 0
    while time.time() < deadline - 2.0:
        # Check if we have enough time for a full VND pass
        # VND complexity depends on N. For N=100, it's fast. For N=1000, it's slow.
        # Estimate time for one VND pass? Hard. 
        # Let's just try. If it takes too long, the deadline check inside _apply_vnd will stop it.
        
        # Perturb the current best order
        # Strategy: Random Or-opt moves
        perturbed = best_order.copy()
        
        # Perform 1-3 random Or-opt moves
        n_moves = rng.integers(1, 4)
        for _ in range(n_moves):
            k = rng.integers(1, 4) # Block size 1-3
            if len(perturbed) < k:
                continue
            i = rng.integers(0, len(perturbed) - k + 1)
            block = perturbed[i:i+k]
            remaining = perturbed[:i] + perturbed[i+k:]
            j = rng.integers(0, len(remaining) + 1)
            perturbed = remaining[:j] + block + remaining[j:]
        
        # Polish the perturbed order
        candidate_order = _apply_vnd(processing_times, perturbed, deadline=deadline)
        candidate_ms = _makespan(candidate_order, processing_times)
        
        if candidate_ms < best_ms:
            best_order = candidate_order
            best_ms = candidate_ms
            perturb_count += 1
        else:
            # If no improvement, we might want to diversify more or restart from NEH
            # Occasionally restart from a fresh NEH to escape local optima
            if perturb_count % 5 == 0:
                fresh_neh = _neh_order(processing_times, rng=rng)
                fresh_polished = _apply_vnd(processing_times, fresh_neh, deadline=deadline)
                fresh_ms = _makespan(fresh_polished, processing_times)
                if fresh_ms < best_ms:
                    best_order = fresh_polished
                    best_ms = fresh_ms

    return list(best_order)
# EVOLVE-BLOCK-END
