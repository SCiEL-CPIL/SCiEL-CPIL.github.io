<<<<<<< SEARCH
def _local_search(perm, processing_times, deadline, margin):
    """
    Local search using 2-opt, swap, and insertion moves with fast delta evaluation.
    """
    n_jobs, n_machines = processing_times.shape
    best = perm.copy()
    C = _completion_matrix(best, processing_times)
    best_cost = _makespan_from_C(C)
    improved = True
    
    while improved:
        improved = False
        
        # 2-Opt: Reverse a segment [i, j]
        for i in range(n_jobs - 1):
            if time.monotonic() > deadline - margin:
                return best, best_cost
            for j in range(i + 1, n_jobs):
                new_cost = _eval_2opt(best, processing_times, C, i, j)
                if new_cost < best_cost - 1e-12:
                    # Apply the move
                    best[i:j + 1] = best[i:j + 1][::-1]
                    C = _completion_matrix(best, processing_times)
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Swap: Exchange two positions i, j
        for i in range(n_jobs):
            if time.monotonic() > deadline - margin:
                return best, best_cost
            for j in range(i + 1, n_jobs):
                new_cost = _eval_swap(best, processing_times, C, i, j)
                if new_cost < best_cost - 1e-12:
                    best[i], best[j] = best[j], best[i]
                    C = _completion_matrix(best, processing_times)
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Insertion: Remove job at i, insert at j
        for i in range(n_jobs):
            if time.monotonic() > deadline - margin:
                return best, best_cost
            for j in range(n_jobs):
                if i == j:
                    continue
                new_cost = _eval_insert(best, processing_times, C, i, j)
                if new_cost < best_cost - 1e-12:
                    # Apply insertion
                    job = best[i]
                    if i < j:
                        best[i:j] = best[i+1:j+1]
                        best[j] = job
                    else:
                        best[j+1:i+1] = best[j:i]
                        best[j] = job
                    C = _completion_matrix(best, processing_times)
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break
                
    return best, best_cost
=======
def _local_search_first_improvement(perm, processing_times, deadline, margin):
    """
    Local search using 2-opt, swap, and insertion moves.
    Uses First-Improvement strategy: accepts the first improving move found.
    This is faster per iteration than Best-Improvement, allowing more iterations.
    """
    n_jobs, n_machines = processing_times.shape
    best = perm.copy()
    C = _completion_matrix(best, processing_times)
    best_cost = _makespan_from_C(C)
    improved = True
    
    while improved:
        improved = False
        
        # 2-Opt: Reverse a segment [i, j]
        for i in range(n_jobs - 1):
            if time.monotonic() > deadline - margin:
                return best, best_cost
            for j in range(i + 1, n_jobs):
                new_cost = _eval_2opt(best, processing_times, C, i, j)
                if new_cost < best_cost - 1e-12:
                    # Apply the move
                    best[i:j + 1] = best[i:j + 1][::-1]
                    C = _completion_matrix(best, processing_times)
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Swap: Exchange two positions i, j
        for i in range(n_jobs):
            if time.monotonic() > deadline - margin:
                return best, best_cost
            for j in range(i + 1, n_jobs):
                new_cost = _eval_swap(best, processing_times, C, i, j)
                if new_cost < best_cost - 1e-12:
                    best[i], best[j] = best[j], best[i]
                    C = _completion_matrix(best, processing_times)
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Insertion: Remove job at i, insert at j
        for i in range(n_jobs):
            if time.monotonic() > deadline - margin:
                return best, best_cost
            for j in range(n_jobs):
                if i == j:
                    continue
                new_cost = _eval_insert(best, processing_times, C, i, j)
                if new_cost < best_cost - 1e-12:
                    # Apply insertion
                    job = best[i]
                    if i < j:
                        best[i:j] = best[i+1:j+1]
                        best[j] = job
                    else:
                        best[j+1:i+1] = best[j:i]
                        best[j] = job
                    C = _completion_matrix(best, processing_times)
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break
                
    return best, best_cost


def _iterated_local_search(perm, pt, deadline, margin, rng):
    """
    Iterated Local Search:
    1. Local Search to convergence.
    2. Perturbation (Shake).
    3. Local Search again.
    4. Accept if better, or with probability.
    """
    n_jobs, n_machines = pt.shape
    current_perm = perm.copy()
    current_cost = _calc_makespan(current_perm, pt)
    
    best_perm = current_perm.copy()
    best_cost = current_cost
    
    while time.monotonic() < deadline - margin:
        # 1. Local Search
        current_perm, current_cost = _local_search_first_improvement(current_perm, pt, deadline, margin)
        
        if current_cost < best_cost - 1e-12:
            best_perm = current_perm.copy()
            best_cost = current_cost
            
        if time.monotonic() > deadline - margin:
            break
            
        # 2. Perturbation
        # Randomly select a few moves to shake the solution
        # Strategy: Random 2-opt reversal of a small segment
        # Or random swap of 2-3 jobs
        if n_jobs > 2:
            # Choose a random segment length between 2 and min(5, n_jobs)
            seg_len = rng.integers(2, min(6, n_jobs + 1))
            i = rng.integers(0, n_jobs - seg_len + 1)
            j = i + seg_len - 1
            current_perm[i:j+1] = current_perm[i:j+1][::-1]
            
            # Optionally add a random swap to diversify more
            if n_jobs > 4 and rng.random() < 0.5:
                a = rng.integers(0, n_jobs)
                b = rng.integers(0, n_jobs)
                if a != b:
                    current_perm[a], current_perm[b] = current_perm[b], current_perm[a]
        
        # 3. Local Search on perturbed solution
        current_perm, current_cost = _local_search_first_improvement(current_perm, pt, deadline, margin)
        
        if current_cost < best_cost - 1e-12:
            best_perm = current_perm.copy()
            best_cost = current_cost
        else:
            # If not improved, we can either revert to best or keep current (to explore).
            # Reverting to best is "hill climbing" with restarts.
            # Keeping current allows wandering.
            # Let's revert to best to stay in the basin of the best solution found so far,
            # but the perturbation is applied to the *current* which is the local optimum.
            # If we revert, the next perturbation is from the best solution.
            # This is standard ILS.
            current_perm = best_perm.copy()
            current_cost = best_cost

    return best_perm, best_cost
>>>>>>> REPLACE

