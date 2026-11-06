    def dfs(idx, current_set):
        nonlocal best_set
        if time.time() > deadline - 2:
            return
            
        # Pruning 1: Upper bound
        if len(current_set) + (n_vars - idx) <= len(best_set):
            return
        
        if idx == n_vars:
            if len(current_set) > len(best_set):
                best_set = set(current_set)
            return
        
        # Option 1: Exclude idx
        dfs(idx + 1, current_set)
        
        # Option 2: Include idx
        if is_valid_extension(current_set, idx):
            dfs(idx + 1, current_set | {idx})
