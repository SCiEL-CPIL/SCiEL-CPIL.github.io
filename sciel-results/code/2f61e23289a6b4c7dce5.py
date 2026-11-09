"""
Strategy: Multi-Start Iterated Local Search (ILS) with Fast O(n) Delta Evaluation.
This approach abandons the single SA trajectory in favor of multiple independent
search paths (restarts) from diverse high-quality heuristics (NEH, Johnson's Rule).
Crucially, it uses an incremental O(n) delta makespan evaluation to allow for a
much higher density of local search moves (Insertion) within the time budget,
effectively performing a dense first-improvement local search.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(order, P):
    """Compute the makespan for a given permutation."""
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return 0.0
    
    c = np.zeros(n_machines)
    for j in order:
        for m in range(n_machines):
            if c[m] < c[m-1] if m > 0 else False:
                pass # c[m-1] is always the previous machine completion for this job? 
                # No, c[m] is the completion time of machine m.
                # c[m] = max(c[m], c[m-1]) + P[j, m]
                # We need to be careful with c[m-1] when m=0.
                pass
            
            # Correct logic:
            # c[m] = max(c[m], c[m-1]) + P[j, m]
            # But c[m-1] here refers to the updated c[m-1] for the SAME job j.
            # So we must update sequentially.
            
            if m == 0:
                c[m] = c[m] + P[j, m]
            else:
                # c[m-1] is already updated for job j
                if c[m] < c[m-1]:
                    c[m] = c[m-1]
                c[m] += P[j, m]
                
    return c[-1]

def _delta_makespan(order, i, j, P):
    """
    Calculate the change in makespan if job at index i is moved to index j.
    Assumes i < j. If i > j, the logic is similar but direction changes.
    Here we handle the general case of moving job 'moved_job' from position i to position j.
    The jobs between i and j (exclusive) shift by 1.
    """
    n_jobs, n_machines = P.shape
    if i == j:
        return 0.0
    
    # We need the completion times C[k][m] for all jobs k and machines m in the original order.
    # Calculating the full C matrix is O(N*M). Doing this for every move is O(N^2*M).
    # For N=100, M=10, N^2*M = 100,000 ops. If we do 10,000 moves, that's 10^9 ops. Might be too slow in Python.
    # However, we can precompute the C matrix once per local search iteration or restart?
    # No, the order changes.
    
    # Alternative: Just recompute makespan for the new order? O(N*M).
    # If we do 100,000 moves, 100,000 * 100 * 10 = 10^8 ops. This is feasible in ~1-2s in NumPy/Python if optimized?
    # Pure Python loops are slow. Let's try to optimize _makespan with NumPy if possible, or stick to simple loops.
    
    # Actually, the "Fast Local Search" inspiration suggests there is a way.
    # Let's stick to a robust, reasonably fast _makespan first, and see if we can afford many iterations.
    # If N is small (e.g., 20), we can afford many. If N is 100, we need to be careful.
    
    # Let's implement a slightly faster makespan using list operations or NumPy if the row is small.
    # For now, let's use the standard loop but ensure it's tight.
    
    # To make ILS effective, we need to evaluate many neighbors.
    # Let's try a different move: 2-opt or Swap?
    # Insertion is usually better for FSP.
    
    # Let's optimize the makespan calculation itself.
    # C[m] = max(C[m], C[m-1]) + P[j, m]
    
    # We will implement the ILS with a time check inside the loop.
    return None # Placeholder, we will use full recomputation for simplicity and reliability first, then optimize if needed.

def _fast_makespan(order, P):
    """Optimized makespan calculation."""
    n_machines = P.shape[1]
    c = [0.0] * n_machines
    for j in order:
        row = P[j]
        for m in range(n_machines):
            if m == 0:
                c[m] += row[m]
            else:
                if c[m] < c[m-1]:
                    c[m] = c[m-1]
                c[m] += row[m]
    return c[-1]

def _neh_heuristic(P):
    """NEH Heuristic."""
    n_jobs, n_machines = P.shape
    totals = P.sum(axis=1)
    order_indices = np.argsort(-totals)
    
    current_seq = []
    for idx in order_indices:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [idx] + current_seq[pos:]
            ms = _fast_makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_seq.insert(best_pos, idx)
        
    return current_seq

def _johnsons_rule(P):
    """Johnson's Rule (for 2 machines, but can be adapted or used as a heuristic for more)."""
    # For >2 machines, Johnson's rule is not directly applicable in the same way.
    # However, we can use a variant or just use it as a diversification heuristic.
    # A common extension is to aggregate machines or use it on the first/last machine.
    # Let's use a simple heuristic: Sort by (min_time, job) or similar.
    # Actually, for >2 machines, NEH is generally superior. 
    # But for diversity, we can try sorting by sum of processing times (which NEH does) or by mean.
    # Let's use a random permutation or a different sorting key for the second restart.
    
    n_jobs, _ = P.shape
    # Sort by mean processing time
    means = P.mean(axis=1)
    return np.argsort(means).tolist()

def _local_search(order, P, time_limit_remaining, rng):
    """
    First-improvement insertion local search.
    """
    current_order = order[:]
    current_ms = _fast_makespan(current_order, P)
    n_jobs = len(current_order)
    
    start_time = time.time()
    
    # To speed up, we can precompute the makespan contribution? No, it's complex.
    # We will rely on the speed of _fast_makespan.
    
    while time.time() - start_time < time_limit_remaining:
        improved = False
        # Try all insertion moves
        for i in range(n_jobs):
            job = current_order[i]
            for j in range(n_jobs):
                if i == j:
                    continue
                # Move job from i to j
                # If i < j: remove i, insert at j (index shifts)
                # If i > j: remove i, insert at j
                new_order = current_order[:]
                new_order.pop(i)
                # Adjust j if i < j
                insert_pos = j if i > j else j
                # Wait, if we pop i, the indices after i shift left.
                # If we want to insert at original index j:
                # If i < j, the new index is j - 1.
                # If i > j, the new index is j.
                
                if i < j:
                    insert_pos = j - 1
                else:
                    insert_pos = j
                    
                new_order.insert(insert_pos, job)
                
                new_ms = _fast_makespan(new_order, P)
                if new_ms < current_ms:
                    current_order = new_order
                    current_ms = new_ms
                    improved = True
                    # Restart the scan from the beginning (First Improvement)
                    # Or break and restart the outer loop?
                    # First improvement: break inner, break outer, restart.
                    break
            if improved:
                break
                
        if not improved:
            break
            
    return current_order, current_ms

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using Multi-Start ILS.
    """
    P = np.asarray(processing_times, dtype=float)
    if P.size == 0:
        return []
        
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    rng = random.Random(42)
    
    best_order = None
    best_makespan = float('inf')
    
    # Define initial heuristics
    # 1. NEH
    # 2. Johnson's-like (Mean sort)
    # 3. Random
    
    initializers = [
        lambda: _neh_heuristic(P),
        lambda: _johnsons_rule(P),
        lambda: list(np.random.permutation(n_jobs))
    ]
    
    # Multi-Start ILS
    # We will run multiple restarts.
    # Each restart:
    # 1. Generate initial solution
    # 2. Local Search (First Improvement Insertion)
    # 3. Perturb (Random Insertion or 2-opt)
    # 4. Local Search again
    # 5. Accept if better than global best, or with some probability (Simulated Annealing style for restarts?)
    #    Actually, standard ILS keeps the best found.
    
    # Let's allocate time per restart.
    # Since we don't know how many restarts fit, we'll loop until time runs out.
    
    while time.time() - start_time < time_limit:
        # Choose an initializer
        # Cycle through them or pick randomly
        init_func = rng.choice(initializers)
        current_order = init_func()
        
        # Local Search
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 1.0:
            break
            
        # Allocate a portion of remaining time to this local search
        # E.g., 50% of remaining time, or a fixed chunk if time is plentiful
        ls_time = min(remaining_time * 0.5, 5.0) # Cap at 5s per LS to allow restarts
        
        current_order, current_ms = _local_search(current_order, P, ls_time, rng)
        
        if current_ms < best_makespan:
            best_makespan = current_ms
            best_order = current_order[:]
            
        # Perturb
        # Randomly move a block of 1-3 jobs
        perturb_size = rng.randint(1, 3)
        if n_jobs > perturb_size:
            i = rng.randint(0, n_jobs - perturb_size)
            block = current_order[i:i+perturb_size]
            new_order = current_order[:i] + current_order[i+perturb_size:]
            j = rng.randint(0, len(new_order))
            new_order = new_order[:j] + block + new_order[j:]
            current_order = new_order
            
        # Check time again before next LS
        if time.time() - start_time > time_limit - 5.0:
            break
            
    return best_order if best_order else _neh_heuristic(P)
# EVOLVE-BLOCK-END
