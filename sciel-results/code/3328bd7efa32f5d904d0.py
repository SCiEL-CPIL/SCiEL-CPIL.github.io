    for target_scope in range(current_best_scope - 1, 104, -1):
        if time.time() > backtrack_deadline:
            break
        res = _backtrack_reduce(best_result, k, target_scope, backtrack_deadline, rng)
        if res is not None:
            best_result = res
            best_score = target_scope
            # We found a solution for this scope. 
            # Should we try to go lower?
            # Yes, the loop continues to target_scope - 1.
