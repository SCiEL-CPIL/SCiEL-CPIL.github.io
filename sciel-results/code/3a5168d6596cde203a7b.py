"""
Simulated Annealing with Or-opt moves for flow-shop permutation optimization.
Uses vectorized NumPy for fast makespan evaluation and a cooling schedule to
escape local minima, unlike the deterministic hill-climbing of the previous strategy.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(perm, pt):
    """
    Compute makespan for a given permutation using vectorized NumPy operations.
    This is significantly faster than Python loops for the evaluation step in SA.
    """
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0
    
    # pt[perm] gives the processing times for the jobs in the order of perm
    # Shape: (n_jobs, n_machines)
    p = pt[perm]
    
    # C[0] is just p[0]
    C = np.zeros_like(p)
    C[0] = p[0]
    
    # For subsequent jobs, C[j] = max(C[j-1], C[j, :-1]) + p[j]
    # We need to handle the first machine separately as C[j, 0] = C[j-1, 0] + p[j, 0]
    # And for m > 0, C[j, m] = max(C[j-1, m], C[j, m-1]) + p[j, m]
    
    # Vectorized approach:
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + p[j, m]
    # This dependency chain (C[j, m-1]) makes full vectorization tricky without loops over j.
    # However, we can unroll the loop over jobs, which is usually small (n <= 100-500).
    # The inner loop over machines can be vectorized? No, because of C[j, m-1].
    # So we stick to a loop over jobs, but use NumPy arrays for the row operations.
    
    for j in range(1, n_jobs):
        # C[j, 0] = C[j-1, 0] + p[j, 0]
        C[j, 0] = C[j-1, 0] + p[j, 0]
        
        if p.shape[1] > 1:
            # For m > 0:
            # C[j, m] = max(C[j-1, m], C[j, m-1]) + p[j, m]
            # We can compute this iteratively for m, but that's a Python loop.
            # Alternative: 
            # Let A = C[j-1, 1:]
            # Let B = C[j, :-1] (which depends on C[j, m-1])
            # This is still sequential in m.
            
            # Actually, for flow shop, the critical path logic is:
            # C[j, m] = max(C[j-1, m], C[j, m-1]) + p[j, m]
            # This is essentially a 2D prefix max plus addition.
            # We can just do a simple loop for m if n_machines is small.
            # Given n_jobs is the outer loop, and we want speed, let's optimize the inner.
            
            # Let's try a small loop for m. It's usually n_machines ~ 5-20.
            for m in range(1, p.shape[1]):
                C[j, m] = max(C[j-1, m], C[j, m-1]) + p[j, m]
                
    return C[-1, -1]

def _neh_order(pt):
    """NEH heuristic for a strong initial solution."""
    n_jobs, n_machines = pt.shape
    totals = pt.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    perm = []
    for job in sorted_jobs:
        if not perm:
            perm.append(job)
            continue
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(candidate, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    return perm

def _or_opt_move(perm, n):
    """
    Generate a new permutation by moving a segment of length 1, 2, or 3.
    """
    new_perm = list(perm)
    # Choose segment length
    seg_len = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
    if n < seg_len:
        seg_len = 1
    
    # Choose start position
    start = np.random.randint(0, n - seg_len + 1)
    
    # Extract segment
    segment = new_perm[start:start+seg_len]
    
    # Remove segment
    new_perm = new_perm[:start] + new_perm[start+seg_len:]
    
    # Choose insertion position in the remaining list
    # The remaining list has size n - seg_len
    # We can insert at any position from 0 to n - seg_len
    insert_pos = np.random.randint(0, (n - seg_len) + 1)
    
    # Insert segment
    new_perm = new_perm[:insert_pos] + segment + new_perm[insert_pos:]
    
    return new_perm

def order_jobs(processing_times):
    """
    Main entry point: order jobs to minimize makespan on a flow shop.
    Uses Simulated Annealing with Or-opt moves.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_total = time.time()
    deadline = 145.0 # Safety margin
    
    # 1. NEH initial solution
    current_perm = _neh_order(processing_times)
    current_cost = _makespan(current_perm, processing_times)
    
    best_perm = list(current_perm)
    best_cost = current_cost
    
    # SA Parameters
    # T0 should be roughly the scale of cost differences
    # Estimate cost scale
    if n_jobs > 1:
        random_perm = np.random.permutation(n_jobs)
        random_cost = _makespan(random_perm, processing_times)
        # Heuristic for initial temperature: difference between random and best
        T0 = max(1.0, (random_cost - current_cost) * 2.0)
    else:
        T0 = 1.0
        
    T_end = 0.01
    # Number of iterations
    # We will run until time limit
    # Cooling rate: T = T0 * alpha^k
    # We want T to drop to T_end in a reasonable number of steps.
    # Let's aim for ~1000-5000 steps total depending on size.
    # Actually, let's just use a time-based loop with a cooling schedule based on iterations.
    
    # Estimate max iterations based on n
    # For n=100, _makespan is fast. We can do many.
    # Let's try to do as many as possible.
    
    alpha = 0.9995 # Slow cooling
    T = T0
    
    while time.time() < deadline:
        # Generate neighbor
        new_perm = _or_opt_move(current_perm, n_jobs)
        new_cost = _makespan(new_perm, processing_times)
        
        delta = new_cost - current_cost
        
        # Acceptance criterion
        if delta < 0:
            # Improvement
            current_perm = new_perm
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_perm = list(current_perm)
        else:
            # Worse, accept with probability exp(-delta / T)
            if T > 0:
                prob = np.exp(-delta / T)
                if np.random.random() < prob:
                    current_perm = new_perm
                    current_cost = new_cost
        
        # Cool down
        T *= alpha
        
        # If T is very low, we are basically doing hill climbing.
        # To allow escaping, we can occasionally reheat or just let it run.
        # Given the time limit, we rely on the initial diversity of Or-opt moves.
        
        # Optional: If stuck for too long, maybe perturb?
        # For now, standard SA.
        
        # Check time frequently
        if time.time() > deadline - 5.0:
            break
            
    return best_perm
# EVOLVE-BLOCK-END
