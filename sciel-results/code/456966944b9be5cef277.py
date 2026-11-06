    def update_after_flip(u, v, adj_masks, subsets_6, subsets_7, affected_6, affected_7, is_reg_6, is_reg_7, edge_count, total_viol):
        # Flip u, v in adj_masks
        # Process affected 6
        for idx in affected_6[u][v]:
            # Check old reg
            was_reg = is_reg_6[idx]
            # Check new reg
            subset = subsets_6[idx]
            mask = 0
            for x in subset: mask |= (1 << x)
            # Wait, mask is invariant. Precompute masks!
            is_now_reg = check_reg(masks_6[idx], subsets_6[idx], adj_masks)
            
            if was_reg != is_now_reg:
                is_reg_6[idx] = is_now_reg
                delta = 1 if is_now_reg else -1
                total_viol += delta
                # Update edge counts
                for i in range(len(subset)):
                    for j in range(i+1, len(subset)):
                        a, b = subset[i], subset[j]
                        if a > b: a, b = b, a
                        edge_count[a][b] += delta
        # Same for 7
        # ...
        return total_viol
    