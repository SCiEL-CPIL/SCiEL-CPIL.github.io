    def _local_search(processing_times, sequence, deadline):
        n_jobs = len(sequence)
        current_seq = list(sequence)
        current_makespan = _calc_makespan_fast(processing_times, current_seq)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_makespan = current_makespan
            best_seq = current_seq
            
            # Try insertion moves
            for i in range(n_jobs):
                if time.time() >= deadline:
                    return current_seq, current_makespan
                for j in range(n_jobs):
                    if j == i:
                        continue
                    new_seq = _insertion_move(current_seq, i, j)
                    new_makespan = _calc_makespan_fast(processing_times, new_seq)
                    if new_makespan < best_makespan:
                        best_makespan = new_makespan
                        best_seq = new_seq
            
            # Try 2-opt moves
            for i in range(n_jobs - 1):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, n_jobs):
                    new_seq = current_seq[:i] + list(reversed(current_seq[i:j+1])) + current_seq[j+1:]
                    new_makespan = _calc_makespan_fast(processing_times, new_seq)
                    if new_makespan < best_makespan:
                        best_makespan = new_makespan
                        best_seq = new_seq
                        
            # Try Or-opt moves (move block of size 2)
            if n_jobs >= 3:
                for i in range(n_jobs - 2):
                    if time.time() >= deadline:
                        break
                    block = current_seq[i:i+2]
                    remaining = current_seq[:i] + current_seq[i+2:]
                    for j in range(len(remaining) + 1):
                        new_seq = remaining[:j] + block + remaining[j:]
                        new_makespan = _calc_makespan_fast(processing_times, new_seq)
                        if new_makespan < best_makespan:
                            best_makespan = new_makespan
                            best_seq = new_seq

            if best_makespan < current_makespan:
                current_seq = best_seq
                current_makespan = best_makespan
                improved = True
            else:
                improved = False
        
        return current_seq, current_makespan
    