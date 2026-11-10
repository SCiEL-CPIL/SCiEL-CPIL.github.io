    def dfs(mask, curr_val, selected_mask):
        nonlocal best_val, best_mask, time_exceeded
        
        if time.time() - start_time > DEADLINE:
            time_exceeded = True
            return

        if curr_val + calc_mask_sum(mask) <= best_val:
            return

        if mask == 0:
            if curr_val > best_val:
                best_val = curr_val
                best_mask = selected_mask
            return

        v = -1
        for idx in sorted_order:
            if mask & (1 << idx):
                v = idx
                break
        
        if v == -1:
            return

        # Include v
        new_mask_include = mask & ~(1 << v) & ~neighbors[v]
        new_selected_include = selected_mask | (1 << v)
        dfs(new_mask_include, curr_val + weights[v], new_selected_include)
        
        if time_exceeded:
            return

        # Exclude v
        new_mask_exclude = mask & ~(1 << v)
        dfs(new_mask_exclude, curr_val, selected_mask)
