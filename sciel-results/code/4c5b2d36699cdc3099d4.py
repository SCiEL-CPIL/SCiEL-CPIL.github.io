import numpy as np
import time

def _makespan(pt, perm):
    """
    Calculate makespan for a given permutation of job indices.
    pt: (N, M) array of processing times.
    perm: list or array of job indices in order.
    """
    n = len(perm)
    if n == 0:
        return 0.0
    
    # C[i][j] = max(C[i-1][j], C[i][j-1]) + pt[perm[i]][j]
    # We only need the last row to get the makespan (last machine completion time of last job)
    # But we need the previous row to compute the current one.
    
    # Initialize previous row (completion times for machine 0..M-1 for job i-1)
    # For i=0, prev_row is all zeros.
    prev_row = np.zeros(pt.shape[1])
    
    for i in range(n):
        job_idx = perm[i]
        current_row = np.zeros(pt.shape[1])
        # For the first machine (j=0), C[i][0] = C[i-1][0] + pt[i][0]
        current_row[0] = prev_row[0] + pt[job_idx][0]
        
        if pt.shape[1] > 1:
            # For j > 0
            # C[i][j] = max(C[i-1][j], C[i][j-1]) + pt[i][j]
            # We can loop here or use vectorization?
            # The dependency on C[i][j-1] makes full vectorization hard in one line without a loop or scan.
            # But M is usually small.
            for j in range(1, pt.shape[1]):
                current_row[j] = max(prev_row[j], current_row[j-1]) + pt[job_idx][j]
        
        prev_row = current_row
        
    return prev_row[-1]

def _neh(pt, weight_func=None):
    """
    NEH heuristic.
    weight_func: optional function to determine sorting order. Default is sum.
    """
    n = pt.shape[0]
    if n == 0:
        return []
    
    # Calculate weights for sorting
    if weight_func is None:
        weights = pt.sum(axis=1)
    else:
        weights = weight_func(pt)
        
    # Sort jobs by descending weight
    jobs = np.argsort(-weights)
    
    seq = []
    for job in jobs:
        # Try inserting job into all positions in seq
        best_pos = 0
        best_cost = np.inf
        
        # Optimization: If seq is empty, just append
        if not seq:
            seq.append(job)
            continue
            
        current_len = len(seq)
        for pos in range(current_len + 1):
            # Create new sequence
            new_seq = seq[:pos] + [job] + seq[pos:]
            cost = _makespan(pt, new_seq)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
                
        seq = seq[:best_pos] + [job] + seq[best_pos:]
        
    return seq

def _local_search(pt, perm, max_iters=100):
    """
    First-improvement swap local search.
    """
    current_perm = list(perm)
    current_cost = _makespan(pt, current_perm)
    n = len(current_perm)
    
    improved = True
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Swap i and j
                current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
                new_cost = _makespan(pt, current_perm)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    # Swap back
                    current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
                    # Note: In first improvement, we typically continue searching
                    # or restart. Here we just continue.
    
    return current_perm, current_cost

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    pt = np.asarray(processing_times)
    n = pt.shape[0]
    
    if n == 0:
        return []
    if n == 1:
        return [0]
        
    # 1. Generate diverse initial solutions using NEH variants
    candidates = []
    
    # Variant 1: Sum
    p1 = _neh(pt)
    candidates.append(p1)
    
    # Variant 2: Max
    p2 = _neh(pt, weight_func=lambda x: x.max(axis=1))
    candidates.append(p2)
    
    # Variant 3: Min
    p3 = _neh(pt, weight_func=lambda x: x.min(axis=1))
    candidates.append(p3)
    
    # Variant 4: Randomized NEH (noise)
    if n > 5:
        p4 = _neh(pt, weight_func=lambda x: x.sum(axis=1) + np.random.rand(n) * 0.1 * x.sum(axis=1).max())
        candidates.append(p4)
        
    # Apply local search to each candidate
    best_perm = None
    best_cost = np.inf
    
    for p in candidates:
        if time.time() > deadline:
            break
        p_ls, cost_ls = _local_search(pt, p)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_perm = p_ls
            
    if best_perm is None:
        return list(np.argsort(-pt.sum(axis=1)))
        
    # 2. Simulated Annealing
    current_perm = list(best_perm)
    current_cost = best_cost
    global_best_perm = list(current_perm)
    global_best_cost = current_cost
    
    T = 1.0
    cooling = 0.999
    min_T = 1e-5
    
    # Estimate max steps based on time
    # We will just loop until deadline
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Generate neighbor: random swap
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i == j:
            continue
            
        neighbor_perm = current_perm.copy()
        neighbor_perm[i], neighbor_perm[j] = neighbor_perm[j], neighbor_perm[i]
        
        neighbor_cost = _makespan(pt, neighbor_perm)
        
        delta = neighbor_cost - current_cost
        
        if delta < 0:
            # Accept
            current_perm = neighbor_perm
            current_cost = neighbor_cost
            if current_cost < global_best_cost:
                global_best_cost = current_cost
                global_best_perm = list(current_perm)
        else:
            # Probabilistic accept
            if np.random.rand() < np.exp(-delta / T):
                current_perm = neighbor_perm
                current_cost = neighbor_cost
                
        T *= cooling
        
    return global_best_perm
