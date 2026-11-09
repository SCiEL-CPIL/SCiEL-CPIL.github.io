"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job order."""
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    
    # C[i][j] = completion time of job i on machine j
    # C[i][j] = max(C[i-1][j], C[i][j-1]) + p[order[i]][j]
    completion = np.zeros((n_jobs, n_machines))
    for i, job_idx in enumerate(order):
        for j in range(n_machines):
            p = processing_times[job_idx, j]
            if j == 0:
                completion[i, j] = completion[i-1, j] + p if i > 0 else p
            else:
                if i == 0:
                    completion[i, j] = completion[i, j-1] + p
                else:
                    completion[i, j] = max(completion[i-1, j], completion[i, j-1]) + p
    
    return completion[-1, -1]


def _neh_rule(processing_times):
    """NEH (Nawaz-Enscore-Ham) heuristic: order jobs by decreasing total processing time."""
    totals = processing_times.sum(axis=1)
    return list(np.argsort(-totals))


def _insertion_improve(processing_times, order):
    """Improve order by trying all insertions of each job into all positions."""
    n_jobs = len(order)
    if n_jobs <= 2:
        return order, _makespan(processing_times, order)
    
    current_order = list(order)
    current_ms = _makespan(processing_times, current_order)
    
    improved = True
    while improved:
        improved = False
        for i in range(n_jobs):
            job = current_order[i]
            # Remove job from position i
            remaining = current_order[:i] + current_order[i+1:]
            
            best_pos = i
            best_ms = current_ms
            
            # Try inserting job at each position in remaining
            for pos in range(len(remaining) + 1):
                trial_order = remaining[:pos] + [job] + remaining[pos:]
                ms = _makespan(processing_times, trial_order)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            
            if best_ms < current_ms:
                current_order = remaining[:best_pos] + [job] + remaining[best_pos:]
                current_ms = best_ms
                improved = True
    
    return current_order, current_ms


def _swap_improve(processing_times, order):
    """Improve order by trying all pairwise swaps."""
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan(processing_times, order)
    
    current_order = list(order)
    current_ms = _makespan(processing_times, current_order)
    
    improved = True
    while improved:
        improved = False
        for i in range(n_jobs):
            for j in range(i + 1, n_jobs):
                trial_order = current_order[:]
                trial_order[i], trial_order[j] = trial_order[j], trial_order[i]
                ms = _makespan(processing_times, trial_order)
                if ms < current_ms:
                    current_order = trial_order
                    current_ms = ms
                    improved = True
    
    return current_order, current_ms


def _tabu_search(processing_times, order, max_iters=200, tabu_tenure=7):
    """Tabu search with aspiration criterion."""
    n_jobs = len(order)
    if n_jobs <= 2:
        return order, _makespan(processing_times, order)
    
    current_order = list(order)
    current_ms = _makespan(processing_times, current_order)
    best_order = list(current_order)
    best_ms = current_ms
    
    # Tabu list: store (i, j) pairs that are tabu
    tabu_list = []
    
    for it in range(max_iters):
        best_neighbor_order = None
        best_neighbor_ms = current_ms
        
        for i in range(n_jobs):
            for j in range(i + 1, n_jobs):
                # Check if this move is tabu
                is_tabu = False
                for (ti, tj), t_iter in tabu_list:
                    if (ti == i and tj == j) or (ti == j and tj == i):
                        if it - t_iter < tabu_tenure:
                            is_tabu = True
                            break
                
                trial_order = current_order[:]
                trial_order[i], trial_order[j] = trial_order[j], trial_order[i]
                ms = _makespan(processing_times, trial_order)
                
                # Aspiration: allow tabu move if it improves best
                if ms < best_ms or (not is_tabu and ms < best_neighbor_ms):
                    best_neighbor_ms = ms
                    best_neighbor_order = trial_order
        
        if best_neighbor_order is not None:
            # Add current move to tabu list
            # Find which swap was made
            for i in range(n_jobs):
                if current_order[i] != best_neighbor_order[i]:
                    for j in range(n_jobs):
                        if current_order[j] == best_neighbor_order[i] and i != j:
                            tabu_list.append(((i, j), it))
                            break
                    break
            
            # Prune old tabu entries
            tabu_list = [(move, t_iter) for move, t_iter in tabu_list if it - t_iter < tabu_tenure]
            
            current_order = best_neighbor_order
            current_ms = best_neighbor_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = list(current_order)
        else:
            break
    
    return best_order, best_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    
    Strategy:
    1. Start with NEH heuristic
    2. Improve with insertion-based local search
    3. Improve with swap-based local search
    4. Run tabu search for further improvement
    5. Multiple restarts with perturbation if time allows
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Step 1: NEH heuristic
    best_order = _neh_rule(processing_times)
    best_ms = _makespan(processing_times, best_order)
    
    # Step 2: Insertion-based local search
    if time.time() < deadline:
        best_order, best_ms = _insertion_improve(processing_times, best_order)
    
    # Step 3: Swap-based local search
    if time.time() < deadline:
        best_order, best_ms = _swap_improve(processing_times, best_order)
    
    # Step 4: Tabu search
    if time.time() < deadline:
        tabu_order, tabu_ms = _tabu_search(processing_times, best_order, max_iters=300, tabu_tenure=8)
        if tabu_ms < best_ms:
            best_order = tabu_order
            best_ms = tabu_ms
    
    # Step 5: Additional restarts with perturbation if time allows
    if time.time() < deadline:
        n_restarts = 0
        while time.time() < deadline and n_restarts < 10:
            # Perturb the best solution
            perturbed_order = list(best_order)
            # Randomly swap 2-3 pairs
            n_swaps = min(3, n_jobs - 1)
            for _ in range(n_swaps):
                i, j = np.random.choice(n_jobs, 2, replace=False)
                perturbed_order[i], perturbed_order[j] = perturbed_order[j], perturbed_order[i]
            
            # Local search on perturbed solution
            local_order, local_ms = _insertion_improve(processing_times, perturbed_order)
            local_order, local_ms = _swap_improve(processing_times, local_order)
            
            # Tabu search on locally improved solution
            tabu_order, tabu_ms = _tabu_search(processing_times, local_order, max_iters=150, tabu_tenure=6)
            
            if tabu_ms < best_ms:
                best_order = tabu_order
                best_ms = tabu_ms
            
            n_restarts += 1
    
    return list(best_order)
# EVOLVE-BLOCK-END
