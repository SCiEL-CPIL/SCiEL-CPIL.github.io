            gains = cov_matrix @ temp_mask_int
            # Mask out blocks that are currently in 'blocks' (excluding the one we removed? No, we removed it from counts, but it's still in the list 'blocks' at index i)
            # Actually, 'blocks' still contains b.
            # But b has 0 gain for uncovered subsets?
            # b was removed from counts. So subsets covered *only* by b are now uncovered.
            # b covers those subsets. So b has high gain.
            # We should allow picking b again? No, we want to replace it.
            # If we pick b again, we just re-add it. No improvement.
            # So we should mask out b?
            # Or just mask out all current blocks?
            # If we mask out all current blocks, we are forced to pick a new block.
            # This is a "swap" strategy.
            
            if blocks:
                gains[np.array(blocks)] = 0
