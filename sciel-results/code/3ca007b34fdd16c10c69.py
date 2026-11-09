"""
Flow-shop permutation construction using CDS/NEH initialization,
2-Opt/Insertion local search, and Simulated Annealing for refinement.
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

def _johnsons_rule(P2):
    """
    Apply Johnson's Rule to a 2-machine matrix P2 (N x 2).
    Returns the sequence of job indices.
    """
    n_jobs = P2.shape[0]
    set_A = []
    set_B = []
    for j in range(n_jobs):
        if P2[j, 0] < P2[j, 1]:
            set_A.append((P2[j, 0], j))
        else:
            set_B.append((P2[j, 1], j))
            
    set_A.sort(key=lambda x: x[0])
    set_B.sort(key=lambda x: -x[0])
    
    sequence = [j for _, j in set_A] + [j for _, j in set_B]
    return sequence

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
        
        seq = _johnsons_rule(P2)
        ms = _makespan(seq, P)
        
        if ms < best_ms:
            best_ms = ms
            best_seq = seq
            
    return best_seq if best_seq else list(range(n_jobs))

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

def _local_search(order, P, time_limit):
    """
    Alternating 2-Opt (reversal) and Insertion (move) local search.
    Uses best-improvement strategy.
    """
    n = len(order)
    if n < 2:
        return order
    
    current_order = list(order)
    current_ms = _makespan(current_order, P)
    
    improved = True
    while improved and (time.time() < time_limit):
        improved = False
        
        # 1. Try 2-Opt moves (reversal of segment)
        best_move_2opt = None
        best_ms_2opt = current_ms
        
        # For large N, limit the search range to avoid O(N^2) explosion if time is tight
        # But for typical flow shop sizes (< 200), O(N^2) is usually fine.
        # We iterate all pairs.
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_order = current_order[:i] + current_order[i:j+1][::-1] + current_order[j+1:]
                ms = _makespan(new_order, P)
                if ms < best_ms_2opt:
                    best_ms_2opt = ms
                    best_move_2opt = new_order
        
        if best_move_2opt is not None:
            current_order = best_move_2opt
            current_ms = best_ms_2opt
            improved = True
            continue # Restart with new order
        
        # 2. Try Insertion moves (move job i to position j)
        best_move_ins = None
        best_ms_ins = current_ms
        
        for i in range(n):
            job_i = current_order[i]
            remaining = current_order[:i] + current_order[i+1:]
            for j in range(len(remaining) + 1):
                # Skip if inserting back to the same place
                # In 'remaining', position i corresponds to original i+1? 
                # Original: ... A[i] A[i+1] ...
                # Remaining: ... A[i+1] ... (A[i] removed)
                # Inserting at j in remaining:
                # If j == i, we get ... A[i+1] A[i] ... which is a swap with next? No.
                # If we remove A[i], remaining[k] = A[k+1] for k >= i.
                # Inserting at j:
                # If j < i: ... A[j] ... A[i] ... (shifted)
                # If j == i: ... A[i+1] A[i] ... (A[i] moves to i+1)
                # If j == i-1: ... A[i-1] A[i] ... (same position)
                
                # Simple check: if the resulting order is identical, skip.
                # But constructing the list is cheap compared to makespan.
                # We can just evaluate all, O(N^2) evaluations.
                new_order = remaining[:j] + [job_i] + remaining[j:]
                
                # Optimization: check if order changed
                # if new_order == current_order: continue
                # This comparison is O(N), makespan is O(N*M). If M>1, it's worth it? 
                # Maybe not strictly, but let's just compute makespan.
                
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
    Order jobs to minimize makespan using CDS/NEH initialization,
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
    # Safety margin: The limit is 150s. We should stop checking/iterating by 145s.
    time_limit = 145.0
    
    # 1. Generate initial candidates
    candidates = []
    
    # CDS Heuristic
    seq_cds = _cds_heuristic(P)
    candidates.append(seq_cds)
    
    # NEH Heuristic
    seq_neh = _neh_heuristic(P)
    candidates.append(seq_neh)
    
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
    best_order = _local_search(best_order, P, ls_time_limit)
    best_makespan = _makespan(best_order, P)
    
    # 3. Simulated Annealing
    # Use remaining time for SA to escape local optima
    if time.time() < time_limit - 5.0:
        T_start = best_makespan * 0.1
        T_end = T_start * 1e-6
        cooling_rate = 0.999
        T = T_start
        
        current_order = best_order
        current_ms = best_makespan
        
        best_sa_order = best_order
        best_sa_ms = best_makespan
        
        n = n_jobs
        
        # Random seed for reproducibility if needed, but default is fine
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
                # j is now index in remaining
                # Insert at j
                # Note: if j < 0 or j > len(remaining), handle? 
                # j is in [0, n-1]. If j > i, j becomes j-1.
                # If i=0, j=0 -> skip.
                # If i=0, j=1 -> j becomes 0. Insert at 0.
                # If i=n-1, j=0 -> j stays 0. Insert at 0.
                # If i=n-1, j=n-1 -> skip.
                # Max j in remaining is n-2 (if i < n-1) or n-1 (if i=0? no, len is n-1).
                # len(remaining) is n-1. Indices 0..n-2.
                # If j was n-1 and i < n-1, j becomes n-2. OK.
                # If j was n-1 and i=0, j becomes n-2. OK.
                # So j is always valid index for insertion?
                # Insertion at index j means new_order = remaining[:j] + [job_i] + remaining[j:]
                # This is valid for 0 <= j <= len(remaining).
                
                new_order = remaining[:j+1] + [job_i] + remaining[j+1:]
                # Wait, standard insertion:
                # If we want to place job_i at position k in the final list.
                # If we remove from i, and want to insert at k in the new list.
                # The inspiration code used: new_order = remaining[:j+1] + [job_i] + remaining[j+1:]
                # This inserts after index j in remaining.
                # Let's stick to the inspiration's logic which seemed to work.
                
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
            best_order = _local_search(best_order, P, time_limit - 0.5)
            best_makespan = _makespan(best_order, P)
            
    return best_order
# EVOLVE-BLOCK-END
