            if not np.any(unique_triples_mask):
                # It was redundant? remove_cover missed it?
                # Remove it
                current_blocks.pop(kick_idx_pos)
                used_set.remove(kick_idx)
                coverage[kick_subs] -= 1
                continue
