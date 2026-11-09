"""
Flow-shop permutation construction using Iterated Local Search (ILS)
with First Improvement Insertion Local Search.

Strategy:
1. Generate diverse initial solutions using NEH and CDS heuristics.
2. Improve each using First Improvement Insertion Local Search.
3. Use the best solution as the seed for Iterated Local Search.
4. ILS loop:
   a. Perturb the current best solution (random segment reversal or random move).
   b. Apply First Improvement Insertion Local Search to the perturbed solution.
   c. If the result is better than the global best, update global best.
   d. Repeat until time limit.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(order, P):
    """
    Compute makespan for a given job order on processing_times matrix P.
    Uses a simple loop which is fast enough for local search steps.
    """
    if len(order) == 0:
        return 0.0
    n_machines = P.shape[1]
    # c_machines[m] is the completion time of the last job processed on machine m
    c_machines = np.zeros(n_machines)
    c_curr = 0.0
    
    for j in order:
        row = P[j]
        for m in range(n_machines):
            c_curr = max(c_machines[m], c_curr) + row[m]
            c_machines[m] = c_curr
            
    return c_curr

def _neh_heuristic(P):
    """
    NEH (Nawaz-Enscore-Ham) heuristic.
    """
    n_jobs, n_machines = P.shape
    totals = P.sum(axis=1)
    order_indices = np.argsort(-totals)
    
    current_seq = []
    for idx in order_indices:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [idx] + current_seq[pos:]
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_seq.insert(best_pos, idx)
        
    return current_seq

def _cds_heuristic(P):
    """
    Campbell, Duke, and Sarma (CDS) heuristic.
    """
    n_jobs, n_machines = P.shape
    if n_machines < 2:
        return list(range(n_jobs))
        
    best_seq = None
    best_ms = float('inf')
    
    for i in range(1, n_machines):
        P2 = np.zeros((n_jobs, 2))
        P2[:, 0] = P[:, :i].sum(axis=1)
        P2[:, 1] = P[:, n_machines-i:].sum(axis=1)
        
        # Johnson's rule
        set_A = []
        set_B = []
        for j in range(n_jobs):
            if P2[j, 0] < P2[j, 1]:
                set_A.append((P2[j, 0], j))
            else:
                set_B.append((P2[j, 1], j))
                
        set_A.sort(key=lambda x: x[0])
        set_B.sort(key=lambda x: -x[0])
        
        seq = [j for _, j in set_A] + [j for _, j in set_B]
        ms = _makespan(seq, P)
        
        if ms < best_ms:
            best_ms = ms
            best_seq = seq
            
    return best_seq if best_seq else list(range(n_jobs))

def _local_search_first_improvement_insertion(order, P, time_limit):
    """
    First Improvement Insertion Local Search.
    Moves a job to a different position if it improves the makespan.
    Stops as soon as an improvement is found for a job, then restarts the scan.
    This is much faster than Best Improvement and allows more iterations.
    """
    n = len(order)
    if n < 2:
        return order, _makespan(order, P)
    
    current_order = list(order)
    current_ms = _makespan(current_order, P)
    
    improved = True
    while improved and (time.time() < time_limit):
        improved = False
        
        for i in range(n):
            if time.time() >= time_limit:
                break
                
            job_i = current_order[i]
            # Temporarily remove job i
            remaining = current_order[:i] + current_order[i+1:]
            
            # Try inserting job_i at all positions j in remaining
            # Note: inserting at position i (original) or i+1 (if i < j) results in same order
            for j in range(len(remaining) + 1):
                # Skip positions that result in the same permutation
                # If j == i, we are putting it back where it was (before removal, index i in original)
                # Actually, remaining has length n-1.
                # Original index i.
                # If we insert at j in remaining:
                # If j == i, the job goes back to position i. (No change)
                # If j == i+1, the job goes to position i+1? No.
                # Let's just check if the resulting order is different.
                # A simpler check: if j == i, skip.
                # Also if j == i+1? 
                # Original: [0, 1, 2, 3]. i=1 (job 1). remaining=[0, 2, 3].
                # Insert at 0: [1, 0, 2, 3] (Move left)
                # Insert at 1: [0, 1, 2, 3] (Original) -> Skip
                # Insert at 2: [0, 2, 1, 3] (Move right)
                # Insert at 3: [0, 2, 3, 1] (Move right)
                
                if j == i:
                    continue
                
                new_order = remaining[:j] + [job_i] + remaining[j:]
                new_ms = _makespan(new_order, P)
                
                if new_ms < current_ms - 1e-12:
                    current_order = new_order
                    current_ms = new_ms
                    improved = True
                    break # Restart scan from beginning for this new order
    
    return current_order, current_ms

def _perturb(order, strength):
    """
    Perturb the order by performing 'strength' random moves or reversals.
    """
    n = len(order)
    new_order = list(order)
    
    for _ in range(strength):
        if random.random() < 0.5:
            # Random swap
            if n > 1:
                i, j = random.sample(range(n), 2)
                new_order[i], new_order[j] = new_order[j], new_order[i]
        else:
            # Random segment reversal (2-Opt)
            if n > 2:
                i, j = random.sample(range(n), 2)
                if i > j:
                    i, j = j, i
                new_order[i:j+1] = reversed(new_order[i:j+1])
                
    return new_order

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using ILS.
    """
    P = np.asarray(processing_times, dtype=np.float64)
    if P.size == 0:
        return []
        
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    start_time = time.time()
    # Safety margin: return before 150s
    time_limit = start_time + 145.0
    
    # 1. Generate diverse initial candidates
    candidates = []
    
    # NEH
    seq_neh = _neh_heuristic(P)
    candidates.append(seq_neh)
    
    # CDS
    seq_cds = _cds_heuristic(P)
    candidates.append(seq_cds)
    
    # Random
    random_perm = list(range(n_jobs))
    random.shuffle(random_perm)
    candidates.append(random_perm)
    
    # SPT (Shortest Processing Time) on first machine
    spt_perm = list(np.argsort(P[:, 0]))
    candidates.append(spt_perm)
    
    # LPT (Longest Processing Time) on last machine
    lpt_perm = list(np.argsort(-P[:, -1]))
    candidates.append(lpt_perm)

    # 2. Improve each candidate with Local Search
    best_order = None
    best_makespan = float('inf')
    
    # Allocate time for initial local searches
    # If n_jobs is large, local search might take a while.
    # We will do a quick pass first.
    ls_time_per_candidate = 5.0  # seconds per candidate
    
    for seq in candidates:
        if time.time() > time_limit - 10.0:
            break
            
        # Run local search with a small time limit to get a good local optimum
        ls_deadline = time.time() + min(ls_time_per_candidate, time_limit - time.time() - 5.0)
        if ls_deadline <= time.time():
             continue
             
        improved_seq, ms = _local_search_first_improvement_insertion(seq, P, ls_deadline)
        
        if ms < best_makespan:
            best_makespan = ms
            best_order = improved_seq

    if best_order is None:
        # Fallback
        best_order = list(range(n_jobs))
        best_makespan = _makespan(best_order, P)

    # 3. Iterated Local Search (ILS)
    # Use the remaining time to explore the neighborhood of the best solution.
    current_order = best_order
    current_ms = best_makespan
    
    # Adaptive perturbation strength
    # Start with small perturbations, increase if stuck?
    # Or just random strength between 1 and 5.
    
    no_improve_count = 0
    patience = 50 # Iterations without improvement before increasing perturbation
    
    while time.time() < time_limit - 2.0:
        # Determine perturbation strength
        # If stuck, perturb more
        if no_improve_count > patience:
            strength = random.randint(3, 6)
            no_improve_count = 0
            patience *= 1.5 # Increase patience? Or reset? Let's just increase strength.
        else:
            strength = random.randint(1, 3)
            
        perturbed_order = _perturb(current_order, strength)
        
        # Run local search on the perturbed solution
        # Limit time for this local search to ensure we can do many iterations
        # Estimate time for one LS step. For n=50, it's fast.
        # We'll use a dynamic deadline: use 10% of remaining time or 1 second, whichever is smaller?
        # Actually, First Improvement LS is fast. We can run it until it converges or a short timeout.
        ls_deadline = time.time() + 1.0 # 1 second per ILS iteration
        
        new_order, new_ms = _local_search_first_improvement_insertion(perturbed_order, P, ls_deadline)
        
        if new_ms < current_ms - 1e-12:
            current_order = new_order
            current_ms = new_ms
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Update global best
        if current_ms < best_makespan - 1e-12:
            best_makespan = current_ms
            best_order = current_order

    return best_order
# EVOLVE-BLOCK-END
