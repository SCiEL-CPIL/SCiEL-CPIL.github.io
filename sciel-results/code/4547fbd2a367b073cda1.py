def _tabu_search(initial_seq, processing_times, max_iters=2000, tabu_tenure=7):
    n_jobs = len(initial_seq)
    if n_jobs < 2:
        return initial_seq, _makespan(initial_seq, processing_times)

    current_seq = initial_seq[:]
    current_ms = _makespan(current_seq, processing_times)
    best_seq = current_seq[:]
    best_ms = current_ms
    
    # Tabu list: deque of job indices
    tabu_jobs = deque()
    
    for _ in range(max_iters):
        best_neighbor_ms = float('inf')
        best_neighbor_seq = None
        best_neighbor_job = -1
        
        # Store best tabu neighbor as fallback
        best_tabu_neighbor_ms = float('inf')
        best_tabu_neighbor_seq = None
        best_tabu_neighbor_job = -1

        for i in range(n_jobs):
            job = current_seq[i]
            remaining = current_seq[:i] + current_seq[i+1:]
            
            for j in range(len(remaining) + 1):
                if j == i:
                    continue
                
                candidate_seq = remaining[:j] + [job] + remaining[j:]
                ms = _makespan(candidate_seq, processing_times)
                
                is_tabu = job in tabu_jobs
                
                # Aspiration
                if ms < best_ms:
                    is_tabu = False
                
                if not is_tabu:
                    if ms < best_neighbor_ms:
                        best_neighbor_ms = ms
                        best_neighbor_seq = candidate_seq
                        best_neighbor_job = job
                else:
                    if ms < best_tabu_neighbor_ms:
                        best_tabu_neighbor_ms = ms
                        best_tabu_neighbor_seq = candidate_seq
                        best_tabu_neighbor_job = job
        
        # Decide which neighbor to take
        if best_neighbor_seq is not None:
            current_seq = best_neighbor_seq
            current_ms = best_neighbor_ms
            moved_job = best_neighbor_job
        elif best_tabu_neighbor_seq is not None:
            # All were tabu, take the best tabu one
            current_seq = best_tabu_neighbor_seq
            current_ms = best_tabu_neighbor_ms
            moved_job = best_tabu_neighbor_job
        else:
            break
            
        if current_ms < best_ms:
            best_ms = current_ms
            best_seq = current_seq[:]
            
        # Update Tabu List
        if moved_job != -1:
            # Remove if already there (shouldn't be if tenure < n_jobs, but safe)
            if moved_job in tabu_jobs:
                tabu_jobs.remove(moved_job)
            tabu_jobs.append(moved_job)
            
            if len(tabu_jobs) > tabu_tenure:
                tabu_jobs.popleft()
                
    return best_seq, best_ms
