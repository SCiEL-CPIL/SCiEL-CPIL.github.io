    def _insertion_local_search(processing_times, seq, time_limit, start_time):
        n_jobs = len(seq)
        current_ms = _makespan(processing_times, seq)
        
        # We will do a single pass of first-improvement insertion.
        # If we find an improvement, we restart the pass (or continue).
        # To avoid infinite loops, we limit the number of restarts or just do one pass.
        # One pass is O(N^2 M).
        
        improved = True
        # Limit passes to avoid getting stuck if improvements are small
        max_passes = 10
        
        while improved and max_passes > 0 and (time.time() - start_time) < time_limit - 5:
            improved = False
            max_passes -= 1
            
            for i in range(n_jobs):
                if (time.time() - start_time) > time_limit - 5:
                    return seq, current_ms
                
                job = seq[i]
                seq.pop(i)
                
                best_pos = i
                best_ms = current_ms
                
                # Try inserting at all positions
                # Optimization: If N is large, this is slow.
                # We can limit the search to a window around i?
                # No, insertion can be far away.
                
                for j in range(n_jobs):
                    if j == i:
                        continue
                    seq.insert(j, job)
                    new_ms = _makespan(processing_times, seq)
                    if new_ms < best_ms:
                        best_ms = new_ms
                        best_pos = j
                        break
                    else:
                        seq.pop(j)
                
                if best_pos != i:
                    # Improvement found. Job is at best_pos.
                    current_ms = best_ms
                    improved = True
                    # Restart the pass to account for index shifts?
                    # Or just break and restart?
                    # Breaking and restarting is safer and often faster for convergence.
                    break
                else:
                    # No improvement. Job is not in seq.
                    seq.insert(i, job)
                    
        return seq, current_ms
    