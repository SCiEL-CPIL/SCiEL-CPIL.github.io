<<<<<<< SEARCH
    # --- Final polish: try to reduce scope by 1 repeatedly ---
    def final_polish(rows, deadline):
        cur = [list(r) for r in rows]
        n = len(cur)
        cur_masks = [row_mask(r) for r in cur]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            cur_scope = scope_of(cur)
            for i in range(n):
                if time.time() >= deadline:
                    break
                cur_scope_i = cur[i][-1]
                if cur_scope_i <= 1:
                    continue
                used_wo = 0
                for ri in range(n):
                    if ri != i:
                        used_wo |= cur_masks[ri]
                # Try to find row with max < cur_scope_i
                res = find_row_dfs(used_wo, cur_scope_i, cur_scope_i - 1, 80000, min(deadline, time.time() + 1.0))
                if res is not None:
                    new_row, new_mask = res
                    if new_row[-1] < cur[i][-1]:
                        cur[i] = new_row
                        cur_masks[i] = new_mask
                        improved = True
                        break  # Restart loop with new scope
        return cur
=======
    # --- Final polish: try to reduce scope by 1 repeatedly ---
    def final_polish(rows, deadline):
        cur = [list(r) for r in rows]
        n = len(cur)
        cur_masks = [row_mask(r) for r in cur]
        global_mask = 0
        for m in cur_masks:
            global_mask |= m
        improved = True
        while improved and time.time() < deadline:
            improved = False
            cur_scope = scope_of(cur)
            for i in range(n):
                if time.time() >= deadline:
                    break
                cur_scope_i = cur[i][-1]
                if cur_scope_i <= 1:
                    continue
                old_mask = cur_masks[i]
                used_wo = global_mask ^ old_mask
                # Try to find row with max < cur_scope_i
                res = find_row_dfs(used_wo, cur_scope_i, cur_scope_i - 1, 80000, min(deadline, time.time() + 1.0))
                if res is not None:
                    new_row, new_mask = res
                    if new_row[-1] < cur[i][-1]:
                        cur[i] = new_row
                        cur_masks[i] = new_mask
                        global_mask = global_mask ^ old_mask | new_mask
                        improved = True
                        break  # Restart loop with new scope
        return cur
>>>>>>> REPLACE
