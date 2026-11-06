# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0 or n_machines == 0:
        return 0.0
    
    # C[j][m] = completion time of job j on machine m
    # We can compute this incrementally
    # For job j in permutation, on machine m:
    # C[j][m] = max(C[j][m-1], C[j-1][m]) + P[j][m]
    # where C[j][-1] = 0 (job hasn't started on first machine)
    # and C[-1][m] = 0 (no previous job)
    
    prev_machine_times = np.zeros(n_machines)  # C[j-1][m] for all m
    makespan = 0.0
    
    for job_idx in perm:
        current_job_start = 0.0
        for m in range(n_machines):
            # Time when job can start on machine m
            # It's max of: when previous job finished on machine m, and when current job finished on previous machine
            current_job_start = max(current_job_start, prev_machine_times[m]) + processing_times[job_idx, m]
            prev_machine_times[m] = current_job_start
        makespan = current_job_start
    
    return makespan


def _v_shaped_heuristic(processing_times):
    """
    Generate initial solution using V-shaped heuristic.
    Jobs with high total processing on first machine go to the left (ascending),
    jobs with high total processing on last machine go to the right (descending).
    The job with the smallest sum of first and last machine processing goes in the middle.
    """
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return list(range(n_jobs))
    if n_jobs == 1:
        return [0]
    if n_machines <= 1:
        # For single machine, sort by processing time descending
        return list(np.argsort(-processing_times[:, 0]))
    
    first_machine = processing_times[:, 0]
    last_machine = processing_times[:, -1]
    sums = first_machine + last_machine
    
    # Sort jobs by sum of first and last machine processing times
    sorted_indices = np.argsort(sums)
    
    # The job with smallest sum goes in the middle
    # Jobs to the left: sorted in descending order of first machine processing
    # Jobs to the right: sorted in ascending order of last machine processing
    
    middle_idx = len(sorted_indices) // 2
    middle_job = sorted_indices[middle_idx]
    
    left_jobs = sorted_indices[:middle_idx]
    right_jobs = sorted_indices[middle_idx + 1:]
    
    # Left side: sort by first machine processing time descending
    left_jobs = left_jobs[np.argsort(-first_machine[left_jobs])]
    # Right side: sort by last machine processing time ascending
    right_jobs = right_jobs[np.argsort(last_machine[right_jobs])]
    
    permutation = list(left_jobs) + [middle_job] + list(right_jobs)
    return permutation


def _swap_makespan_delta(perm, i, j, processing_times):
    """
    Compute the delta in makespan when swapping positions i and j in perm.
    This is an approximation; for exact computation we'd need to recompute.
    For tabu search, we'll use full recomputation for accuracy but optimize where possible.
    """
    # For simplicity and correctness, we'll just recompute makespan
    # but this is O(n*m) per evaluation
    new_perm = perm.copy()
    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    return _makespan(new_perm, processing_times)


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using V-shaped heuristic
    followed by Tabu Search with dynamic tabu list.
    """
    processing_times = np.asarray(processing_times)
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Generate initial solution using V-shaped heuristic
    perm = _v_shaped_heuristic(processing_times)
    best_perm = perm.copy()
    best_makespan = _makespan(perm, processing_times)
    
    # Tabu Search parameters
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Dynamic tabu list: store (i, j) pairs of last swaps
    tabu_list = {}  # (i, j) -> iteration when it will no longer be tabu
    tabu_tenure = 10  # Base tenure
    
    # Diversification: track frequency of job positions
    pos_freq = np.zeros((n_jobs, n_jobs), dtype=np.int32)  # pos_freq[job][position]
    for pos, job in enumerate(perm):
        pos_freq[job][pos] += 1
    
    iteration = 0
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # Dynamic tabu tenure: increase when stuck
        if no_improve_count > max_no_improve:
            tabu_tenure = min(tabu_tenure + 2, 30)
            no_improve_count = 0
        elif no_improve_count < max_no_improve // 2:
            tabu_tenure = max(tabu_tenure - 1, 5)
        
        # Generate neighborhood: swap any two positions
        # For efficiency, we'll sample a subset of moves if n_jobs is large
        if n_jobs <= 100:
            # Try all swaps
            candidates = []
            for i in range(n_jobs):
                for j in range(i + 1, n_jobs):
                    # Check tabu
                    key1 = (i, j)
                    key2 = (j, i)
                    if tabu_list.get(key1, -1) > iteration or tabu_list.get(key2, -1) > iteration:
                        continue
                    # Compute new makespan
                    new_perm = perm.copy()
                    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                    new_makespan = _makespan(new_perm, processing_times)
                    candidates.append((new_makespan, i, j, new_perm))
        else:
            # For larger instances, sample moves
            n_samples = min(200, n_jobs * (n_jobs - 1) // 2)
            candidates = []
            for _ in range(n_samples):
                i = np.random.randint(0, n_jobs)
                j = np.random.randint(0, n_jobs)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                key1 = (i, j)
                key2 = (j, i)
                if tabu_list.get(key1, -1) > iteration or tabu_list.get(key2, -1) > iteration:
                    continue
                new_perm = perm.copy()
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                new_makespan = _makespan(new_perm, processing_times)
                candidates.append((new_makespan, i, j, new_perm))
        
        if not candidates:
            # All moves are tabu; pick random non-tabu or break
            break
        
        # Select best candidate
        candidates.sort(key=lambda x: x[0])
        best_makespan_new, best_i, best_j, best_new_perm = candidates[0]
        
        # Accept if it's improving or if it's the best non-tabu move (to escape local optima)
        if best_makespan_new < best_makespan:
            # Improving move
            perm = best_new_perm
            best_perm = perm.copy()
            best_makespan = best_makespan_new
            no_improve_count = 0
        else:
            # Non-improving: accept with some probability for diversification
            # Use aspiration criterion: accept if it's better than best found
            if best_makespan_new < best_makespan:
                perm = best_new_perm
                best_makespan = best_makespan_new
                no_improve_count = 0
            else:
                # Accept non-improving move with probability to escape local optima
                if np.random.random() < 0.3:
                    perm = best_new_perm
                else:
                    # Try next best
                    for cand in candidates[1:]:
                        if cand[0] < best_makespan:
                            perm = cand[3]
                            best_makespan = cand[0]
                            best_perm = perm.copy()
                            best_i, best_j = cand[1], cand[2]
                            no_improve_count = 0
                            break
                    else:
                        # No improving move found; accept best non-tabu to diversify
                        perm = best_new_perm
                        no_improve_count += 1
        
        # Update tabu list
        tabu_list[(best_i, best_j)] = iteration + tabu_tenure
        tabu_list[(best_j, best_i)] = iteration + tabu_tenure
        
        # Update position frequencies
        for pos, job in enumerate(perm):
            pos_freq[job][pos] += 1
        
        # Clean up old tabu entries
        if iteration % 50 == 0:
            tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}
    
    # Return best permutation found
    return list(best_perm)
# EVOLVE-BLOCK-END
