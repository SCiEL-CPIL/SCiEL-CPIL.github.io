<<<<<<< SEARCH
    # Local search from best initial
    if time.time() < deadline - 10:
        best_priority, best_score = _local_search(
            best_priority, times, predecessors, cycle_time, n,
            max_iters=20000, deadline=deadline - 5
        )
    
    # If we have more time, try local search from a few other starting points
    remaining_time = deadline - 5 - time.time()
    if remaining_time > 20 and best_score > 1:
        rng = np.random.RandomState(42)
        for start_idx in range(3):
            if time.time() > deadline - 5:
                break
            # Start from a different candidate
            pri = candidates[start_idx + 1].copy() if start_idx + 1 < len(candidates) else cp.copy()
            # Add some noise
            pri = pri + rng.normal(0, 0.1, n)
            
            pri, score = _local_search(
                pri, times, predecessors, cycle_time, n,
                max_iters=10000, deadline=deadline - 5 - max(0, (2 - start_idx) * 15)
            )
            
            if score < best_score:
                best_score = score
                best_priority = pri
    
    return best_priority
=======
    # Local search from best initial
    if time.time() < deadline - 10:
        best_priority, best_score = _local_search(
            best_priority, times, predecessors, cycle_time, n,
            max_iters=30000, deadline=deadline - 5
        )
    
    # If we have more time, try local search from a few other starting points
    remaining_time = deadline - 5 - time.time()
    if remaining_time > 20 and best_score > 1:
        rng = np.random.RandomState(42)
        for start_idx in range(5):
            if time.time() > deadline - 5:
                break
            # Start from a different candidate
            if start_idx + 1 < len(candidates):
                pri = candidates[start_idx + 1].copy()
            else:
                pri = cp.copy()
            # Add some noise
            pri = pri + rng.normal(0, 0.1, n)
            
            # Allocate remaining time evenly
            time_per_run = (deadline - 5 - time.time()) / (5 - start_idx)
            dl = time.time() + max(5, time_per_run)
            
            pri, score = _local_search(
                pri, times, predecessors, cycle_time, n,
                max_iters=15000, deadline=dl
            )
            
            if score < best_score:
                best_score = score
                best_priority = pri
    
    # Final polish: try to refine with smaller steps if we have time
    remaining_time = deadline - 2 - time.time()
    if remaining_time > 5 and best_score > 1:
        polish_priority = best_priority.copy()
        polish_score = best_score
        for it in range(min(5000, int(remaining_time * 100))):
            if time.time() > deadline - 2:
                break
            i, j = random.sample(range(n), 2)
            orig_i, orig_j = polish_priority[i], polish_priority[j]
            # Small step
            polish_priority[i] += random.uniform(-0.1, 0.1)
            polish_priority[j] += random.uniform(-0.1, 0.1)
            s = _greedy_balance(times, predecessors, cycle_time, polish_priority)
            if s < polish_score:
                polish_score = s
                best_priority = polish_priority.copy()
                best_score = s
            else:
                polish_priority[i] = orig_i
                polish_priority[j] = orig_j
    
    return best_priority
>>>>>>> REPLACE
