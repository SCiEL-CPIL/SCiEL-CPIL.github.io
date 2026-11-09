"""
Flow-shop permutation construction using NEH initialization, 2-Opt and Insertion Local Search,
with Simulated Annealing for escaping local optima.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(order, P):
    """Compute makespan for a given job order on processing_times matrix P."""
    if len(order) == 0:
        return 0.0
    n_machines = P.shape[1]
    # C[m] is the completion time of the last job processed on machine m
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

def _local_search_2opt_insertion(order, P, time_limit):
    """
    Alternating 2-Opt (reversal) and Insertion (move) local search.
    Returns improved order.
    """
    n = len(order)
    if n < 2:
        return order
    
    current_order = list(order)
    current_ms = _makespan(current_order, P)
    
    improved = True
    while improved and (time.time() < time_limit):
        improved = False
        
        # 2-Opt move: reverse segment [i, j]
        best_move_2opt = None
        best_ms_2opt = current_ms
        
        # Limit range if n is large to keep time bounded, but 2-opt is O(N^2 * M)
        # For N=100, 100^2 = 10k evals. Each eval O(M). If M=10, 100k ops. Fast.
        # For N=500, 250k evals. 2.5M ops. Still fast.
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Reverse segment i to j
                new_order = current_order[:i] + current_order[i:j+1][::-1] + current_order[j+1:]
                ms = _makespan(new_order, P)
                if ms < best_ms_2opt:
                    best_ms_2opt = ms
                    best_move_2opt = new_order
        
        if best_move_2opt is not None:
            current_order = best_move_2opt
            current_ms = best_ms_2opt
            improved = True
            continue # Restart search with new order
        
        # Insertion move: move job i to position j
        best_move_ins = None
        best_ms_ins = current_ms
        
        for i in range(n):
            job_i = current_order[i]
            remaining = current_order[:i] + current_order[i+1:]
            for j in range(len(remaining) + 1):
                if j == i or j == i + 1:
                    continue
                new_order = remaining[:j] + [job_i] + remaining[j:]
                ms = _makespan(new_order, P)
                if ms < best_ms_ins:
                    best_ms_ins = ms
                    best_move_ins = new_order
        
        if best_move_ins is not None:
            current_order = best_move_ins
            current_ms = best_ms_ins
            improved = True
            
    return current_order

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using NEH/CDS initialization,
    Local Search (2-Opt + Insertion), and Simulated Annealing.
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
    time_limit = 145.0  # Safety margin for 150s limit
    
    # 1. Generate initial candidates
    candidates = []
    seq_neh = _neh_heuristic(P)
    candidates.append(seq_neh)
    
    seq_cds = _cds_heuristic(P)
    candidates.append(seq_cds)
    
    # Add a random permutation?
    random_perm = list(range(n_jobs))
    random.shuffle(random_perm)
    candidates.append(random_perm)
    
    # Select best initial sequence
    best_order = None
    best_makespan = float('inf')
    for seq in candidates:
        ms = _makespan(seq, P)
        if ms < best_makespan:
            best_makespan = ms
            best_order = seq
            
    # 2. Local Search
    # Use a portion of the time for local search
    ls_time_limit = start_time + (time_limit - start_time) * 0.4
    best_order = _local_search_2opt_insertion(best_order, P, ls_time_limit)
    best_makespan = _makespan(best_order, P)
    
    # 3. Simulated Annealing
    if time.time() < time_limit - 5.0:
        # SA parameters
        T_start = best_makespan * 0.1
        T_end = T_start * 1e-6
        cooling_rate = 0.999
        T = T_start
        
        current_order = best_order
        current_ms = best_makespan
        
        best_sa_order = best_order
        best_sa_ms = best_makespan
        
        n = n_jobs
        
        while T > T_end and time.time() < time_limit - 1.0:
            # Generate neighbor: random swap or insertion
            if random.random() < 0.5 and n > 1:
                # Swap
                i, j = random.sample(range(n), 2)
                new_order = current_order[:]
                new_order[i], new_order[j] = new_order[j], new_order[i]
            else:
                # Insertion
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                job_i = current_order[i]
                remaining = current_order[:i] + current_order[i+1:]
                # Adjust j if j > i
                if j > i:
                    j = j - 1
                new_order = remaining[:j+1] + [job_i] + remaining[j+1:]
                
            new_ms = _makespan(new_order, P)
            delta = new_ms - current_ms
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_order = new_order
                current_ms = new_ms
                
                if current_ms < best_sa_ms:
                    best_sa_ms = current_ms
                    best_sa_order = current_order[:]
            
            T *= cooling_rate
            
        best_order = best_sa_order
        best_makespan = best_sa_ms
        
        # Final local search polish
        if time.time() < time_limit - 0.5:
            best_order = _local_search_2opt_insertion(best_order, P, time_limit - 0.5)
            best_makespan = _makespan(best_order, P)
            
    return best_order
# EVOLVE-BLOCK-END
