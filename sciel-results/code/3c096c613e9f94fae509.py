"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as _np
import time as _time
import random as _random

def _makespan(perm, pt):
    """Compute makespan for a given permutation."""
    if len(perm) == 0:
        return 0.0
    n = len(perm)
    m = pt.shape[1]
    if n == 0:
        return 0.0
    
    C_prev = _np.zeros(m)
    for j in perm:
        C_curr = _np.zeros(m)
        for k in range(m):
            if k == 0:
                C_curr[0] = C_prev[0] + pt[j, 0]
            else:
                C_curr[k] = max(C_prev[k], C_curr[k-1]) + pt[j, k]
        C_prev = C_curr
        
    return C_prev[-1]

def _delta_insertion(pt, perm, job, pos):
    """
    Calculate the makespan if job is inserted at position pos in perm.
    """
    candidate = perm[:pos] + [job] + perm[pos:]
    return _makespan(candidate, pt)

def _critical_path_priority(pt):
    """
    Compute a priority score for each job based on its contribution to the critical path.
    We use the sum of processing times as a proxy, but we can also consider the position in the critical path.
    For simplicity, we use the total processing time, but we could refine this.
    """
    # A better heuristic: sort by total processing time descending
    # But to be "priority-based constructive", we insert jobs one by one
    # minimizing the increase in makespan.
    # This is essentially the NEH heuristic but with a different insertion criterion.
    # NEH uses total processing time to determine order.
    # Here, we can use a different order, e.g., by the processing time on the first and last machine.
    # Let's use a combination: sort by (p[j,0] + p[j,m-1]) descending.
    m = pt.shape[1]
    if m == 1:
        priority = pt[:, 0]
    else:
        priority = pt[:, 0] + pt[:, m-1]
    
    return _np.argsort(-priority)

def _constructive_heuristic(pt):
    """
    Construct a permutation by iteratively inserting jobs.
    Start with an empty permutation.
    For each job in the priority order, insert it at the position that minimizes the makespan.
    """
    n, m = pt.shape
    if n == 0:
        return []
    
    priority_order = _critical_path_priority(pt)
    
    current_perm = []
    for job_idx in priority_order:
        job = job_idx
        best_pos = 0
        best_mk = float('inf')
        
        for pos in range(len(current_perm) + 1):
            candidate_perm = current_perm[:pos] + [job] + current_perm[pos:]
            mk = _makespan(candidate_perm, pt)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
        
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        
    return current_perm

def _large_neighborhood_search(perm, pt, max_iterations=100):
    """
    Perform a large neighborhood search.
    1. Destroy: Remove k random jobs from the permutation.
    2. Repair: Re-insert the removed jobs one by one at the position that minimizes the makespan.
    3. Accept if the new makespan is better.
    """
    n = len(perm)
    if n == 0:
        return perm, _makespan(perm, pt)
    
    current_perm = perm[:]
    current_mk = _makespan(current_perm, pt)
    
    for _ in range(max_iterations):
        # Destroy: remove k random jobs
        k = min(3, n)
        removed_jobs = _random.sample(current_perm, k)
        remaining_perm = [job for job in current_perm if job not in removed_jobs]
        
        # Repair: re-insert removed jobs
        new_perm = remaining_perm[:]
        for job in removed_jobs:
            best_pos = 0
            best_mk = float('inf')
            for pos in range(len(new_perm) + 1):
                candidate_perm = new_perm[:pos] + [job] + new_perm[pos:]
                mk = _makespan(candidate_perm, pt)
                if mk < best_mk:
                    best_mk = mk
                    best_pos = pos
            new_perm = new_perm[:best_pos] + [job] + new_perm[best_pos:]
        
        new_mk = _makespan(new_perm, pt)
        if new_mk < current_mk:
            current_perm = new_perm
            current_mk = new_mk
    
    return current_perm, current_mk

def _tabu_search(perm, pt, tabu_list_size=50, max_iterations=1000):
    """
    Perform tabu search.
    """
    n = len(perm)
    if n == 0:
        return perm, _makespan(perm, pt)
    
    current_perm = perm[:]
    current_mk = _makespan(current_perm, pt)
    best_perm = current_perm[:]
    best_mk = current_mk
    
    # Tabu list stores (job, from_pos, to_pos)
    tabu_list = []
    
    for _ in range(max_iterations):
        best_move = None
        best_move_mk = float('inf')
        
        for i in range(n):
            job = current_perm[i]
            for j in range(n + 1):
                if j == i or j == i + 1:
                    continue
                
                # Check if move is tabu
                move_key = (job, i, j)
                if move_key in tabu_list:
                    continue
                
                candidate = current_perm[:i] + current_perm[i+1:]
                insert_pos = j
                if j > i:
                    insert_pos = j - 1
                candidate = candidate[:insert_pos] + [job] + candidate[insert_pos:]
                
                cand_mk = _makespan(candidate, pt)
                
                # Aspiration criterion: accept if it improves the best known solution
                if cand_mk < best_mk or cand_mk < best_move_mk:
                    best_move_mk = cand_mk
                    best_move = (candidate, cand_mk, move_key)
        
        if best_move is not None:
            current_perm, current_mk, move_key = best_move
            tabu_list.append(move_key)
            if len(tabu_list) > tabu_list_size:
                tabu_list.pop(0)
            
            if current_mk < best_mk:
                best_mk = current_mk
                best_perm = current_perm[:]
    
    return best_perm, best_mk

def order_jobs(processing_times):
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
        
    start_time = _time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # 1. Initial solution via priority-based constructive heuristic
    current_perm = _constructive_heuristic(processing_times)
    current_mk = _makespan(current_perm, processing_times)
    best_perm = current_perm[:]
    best_mk = current_mk
    
    # 2. Large neighborhood search
    current_perm, current_mk = _large_neighborhood_search(current_perm, processing_times, max_iterations=50)
    if current_mk < best_mk:
        best_perm = current_perm[:]
        best_mk = current_mk
        
    # 3. Tabu search
    current_perm, current_mk = _tabu_search(current_perm, processing_times, tabu_list_size=50, max_iterations=500)
    if current_mk < best_mk:
        best_perm = current_perm[:]
        best_mk = current_mk
        
    # 4. Simulated Annealing with Insertion Moves if time permits
    if _time.time() < deadline:
        T = best_mk * 0.1
        cooling = 0.999
        min_T = 1e-5
        
        current_perm = best_perm[:]
        current_mk = best_mk
        
        while _time.time() < deadline:
            if T < min_T:
                break
                
            i = _random.randint(0, n - 1)
            j = _random.randint(0, n)
            if j == i or j == i + 1:
                T *= cooling
                continue
                
            job = current_perm[i]
            candidate = current_perm[:i] + current_perm[i+1:]
            insert_pos = j
            if j > i:
                insert_pos = j - 1
            candidate = candidate[:insert_pos] + [job] + candidate[insert_pos:]
            
            cand_mk = _makespan(candidate, processing_times)
            delta = cand_mk - current_mk
            
            if delta < 0 or _random.random() < _np.exp(-delta / T):
                current_perm = candidate
                current_mk = cand_mk
                
                if current_mk < best_mk:
                    best_mk = current_mk
                    best_perm = current_perm[:]
                    
            T *= cooling
    
    return list(best_perm)
# EVOLVE-BLOCK-END
