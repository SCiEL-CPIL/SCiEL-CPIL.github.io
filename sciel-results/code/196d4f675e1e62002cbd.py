import numpy as np
import time
from collections import deque

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 114.0  # safety margin before 150s kill
    
    n_jobs, n_machines = processing_times.shape
    
    # Helper to calculate makespan for a given permutation (list of job indices)
    def calc_makespan(perm):
        # perm is a list of job indices
        completion = np.zeros(n_machines)
        for j in perm:
            job_times = processing_times[j]
            for m in range(n_machines):
                if m == 0:
                    st = completion[0]
                else:
                    st = max(completion[m], completion[m-1])
                completion[m] = st + job_times[m]
        return completion[-1]

    # 1. V-shaped heuristic construction
    # In flow shop, optimal solutions often have a V-shape:
    # Jobs with large total processing time on first machine come first,
    # jobs with large total processing time on last machine come last
    
    total_first = processing_times[:, 0]  # Processing time on first machine
    total_last = processing_times[:, -1]   # Processing time on last machine
    
    # Sort jobs by decreasing processing time on first machine
    sorted_by_first = np.argsort(-total_first)
    # Sort jobs by decreasing processing time on last machine  
    sorted_by_last = np.argsort(-total_last)
    
    # Build V-shaped sequence
    seq = []
    left_ptr = 0
    right_ptr = n_jobs - 1
    use_left = True
    
    # Alternate between adding from left (high first-machine time) and right (high last-machine time)
    while left_ptr <= right_ptr:
        if use_left and left_ptr <= right_ptr:
            # Add job with high first-machine processing time
            job = sorted_by_first[left_ptr]
            # Check if this job is already in the "right" portion
            if not any(job == sorted_by_last[r] for r in range(right_ptr + 1, n_jobs)):
                seq.append(job)
                left_ptr += 1
                use_left = False
            else:
                use_left = True
        elif not use_left and right_ptr >= left_ptr:
            # Add job with high last-machine processing time
            job = sorted_by_last[right_ptr]
            # Check if this job is already in the "left" portion
            if not any(job == sorted_by_first[l] for l in range(left_ptr)):
                seq.append(job)
                right_ptr -= 1
                use_left = True
            else:
                use_left = False
        else:
            # Fallback: add remaining jobs
            for j in range(n_jobs):
                if j not in seq:
                    seq.append(j)
            break
    
    # Ensure all jobs are included
    for j in range(n_jobs):
        if j not in seq:
            seq.append(j)
    
    # Refine with NEH-style insertion
    seq = list(seq)
    best_ms = calc_makespan(seq)
    
    # Try to improve initial solution with local insertion
    for _ in range(min(50, n_jobs)):  # Limited refinement
        improved = False
        for i in range(len(seq)):
            if time.time() > deadline:
                return seq
            job = seq[i]
            for j in range(len(seq)):
                if i == j:
                    continue
                # Move job from i to j
                new_seq = seq[:i] + seq[i+1:]
                new_seq.insert(j, job)
                ms = calc_makespan(new_seq)
                if ms < best_ms:
                    best_ms = ms
                    seq = new_seq
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    best_seq = list(seq)
    best_ms = calc_makespan(best_seq)

    # 2. Tabu Search
    # Dynamic tabu list: store (position1, position2) pairs of recent moves
    tabu_list = deque()
    tabu_tenure = max(5, n_jobs // 2)  # Dynamic tenure based on problem size
    
    current_seq = list(seq)
    current_ms = best_ms
    best_seq = list(current_seq)
    best_ms = current_ms
    
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        # Check if tabu list needs trimming
        while len(tabu_list) > tabu_tenure:
            tabu_list.popleft()
        
        # Find best non-tabu move
        best_move = None
        best_move_ms = float('inf')
        best_move_tabu = False
        
        n = len(current_seq)
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                # Check if this move is tabu
                move_key = (min(i, j), max(i, j))
                is_tabu = move_key in tabu_list
                
                # Try the swap
                current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
                ms = calc_makespan(current_seq)
                
                # Aspiration criterion: accept tabu move if it improves best
                if is_tabu and ms >= best_ms:
                    # Revert and skip
                    current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
                    continue
                
                if ms < best_move_ms:
                    best_move_ms = ms
                    best_move = (i, j)
                    best_move_tabu = is_tabu
                
                # Revert swap
                current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
        
        if best_move is None:
            break
        
        i, j = best_move
        # Apply the best move
        current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
        current_ms = best_move_ms
        
        # Add to tabu list
        move_key = (min(i, j), max(i, j))
        tabu_list.append(move_key)
        
        # Update best
        if current_ms < best_ms:
            best_ms = current_ms
            best_seq = list(current_seq)
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Dynamic tabu tenure adjustment
        if no_improve_count > max_no_improve:
            # Increase diversity by increasing tabu tenure
            tabu_tenure = min(tabu_tenure + 2, n_jobs)
            no_improve_count = 0
        elif no_improve_count < max_no_improve // 2:
            # Decrease tabu tenure for more intensification
            tabu_tenure = max(tabu_tenure - 1, 3)
    
    return best_seq

