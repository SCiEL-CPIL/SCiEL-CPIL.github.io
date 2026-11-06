        else:
            # Revert masks
            for i in range(len(affected)):
                p_idx, l_pos = affected[i]
                pair_masks[p_idx] = old_masks[i]
