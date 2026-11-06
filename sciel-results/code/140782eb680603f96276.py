    # Helper: Calculate delta in violations (Dry Run)
    def calc_delta_dry(adj_mask, common_cnt, u, v, c_old, c_new, n, current_violations):
        if c_old == c_new:
            return 0, current_violations
        
        delta = 0
        
        # 1. Remove (u, v) from c_old
        # Pairs (u, w) where w in N_{c_old}(v) \ {u}
        mask_v = adj_mask[c_old][v]
        w = 0
        while w < n:
            if (mask_v >> w) & 1:
                if w != u:
                    old_cnt = common_cnt[c_old][u][w]
                    new_cnt = old_cnt - 1
                    old_viol = max(0, old_cnt - 1)
                    new_viol = max(0, new_cnt - 1)
                    delta += new_viol - old_viol
            w += 1
            
        # Pairs (v, w) where w in N_{c_old}(u) \ {v}
        mask_u = adj_mask[c_old][u]
        w = 0
        while w < n:
            if (mask_u >> w) & 1:
                if w != v:
                    old_cnt = common_cnt[c_old][v][w]
                    new_cnt = old_cnt - 1
                    old_viol = max(0, old_cnt - 1)
                    new_viol = max(0, new_cnt - 1)
                    delta += new_viol - old_viol
            w += 1

        # 2. Add (u, v) to c_new
        # Pairs (u, w) where w in N_{c_new}(v) \ {u}
        mask_v_new = adj_mask[c_new][v]
        w = 0
        while w < n:
            if (mask_v_new >> w) & 1:
                if w != u:
                    old_cnt = common_cnt[c_new][u][w]
                    new_cnt = old_cnt + 1
                    old_viol = max(0, old_cnt - 1)
                    new_viol = max(0, new_cnt - 1)
                    delta += new_viol - old_viol
            w += 1
            
        # Pairs (v, w) where w in N_{c_new}(u) \ {v}
        mask_u_new = adj_mask[c_new][u]
        w = 0
        while w < n:
            if (mask_u_new >> w) & 1:
                if w != v:
                    old_cnt = common_cnt[c_new][v][w]
                    new_cnt = old_cnt + 1
                    old_viol = max(0, old_cnt - 1)
                    new_viol = max(0, new_cnt - 1)
                    delta += new_viol - old_viol
            w += 1

        new_violations = current_violations + delta
        return delta, new_violations

    # Helper: Apply the change to state
    def apply_change(adj_mask, common_cnt, u, v, c_old, c_new, n):
        if c_old == c_new:
            return

        # 1. Remove (u, v) from c_old
        mask_v = adj_mask[c_old][v]
        w = 0
        while w < n:
            if (mask_v >> w) & 1:
                if w != u:
                    common_cnt[c_old][u][w] -= 1
                    common_cnt[c_old][w][u] -= 1
            w += 1
            
        mask_u = adj_mask[c_old][u]
        w = 0
        while w < n:
            if (mask_u >> w) & 1:
                if w != v:
                    common_cnt[c_old][v][w] -= 1
                    common_cnt[c_old][w][v] -= 1
            w += 1

        adj_mask[c_old][u] &= ~(1 << v)
        adj_mask[c_old][v] &= ~(1 << u)

        # 2. Add (u, v) to c_new
        mask_v_new = adj_mask[c_new][v]
        w = 0
        while w < n:
            if (mask_v_new >> w) & 1:
                if w != u:
                    common_cnt[c_new][u][w] += 1
                    common_cnt[c_new][w][u] += 1
            w += 1
            
        mask_u_new = adj_mask[c_new][u]
        w = 0
        while w < n:
            if (mask_u_new >> w) & 1:
                if w != v:
                    common_cnt[c_new][v][w] += 1
                    common_cnt[c_new][w][v] += 1
            w += 1

        adj_mask[c_new][u] |= (1 << v)
        adj_mask[c_new][v] |= (1 << u)
