<<<<<<< SEARCH
                new_mask = block_mask_local | (1 << x)
                # Count how many uncovered t-subsets are subsets of new_mask but not block_mask_local
                # A t-subset is newly covered if: uncovered[i] and (t_masks[i] & ~new_mask) == 0 and (t_masks[i] & ~block_mask_local) != 0
                # Equivalent: uncovered[i] and (t_masks[i] | new_mask) == new_mask and (t_masks[i] | block_mask_local) != block_mask_local
                # Simpler: newly covered = uncovered[i] and (t_masks[i] & ~new_mask) == 0 and (t_masks[i] & ~block_mask_local) != 0
                # Which is: uncovered[i] and (t_masks[i] & ~new_mask) == 0 and (t_masks[i] & (1 << x)) != 0  [since new_mask = block_mask_local | (1<<x)]
                # Actually: (t_masks[i] & ~block_mask_local) != 0 means t_mask has some bit not in block_mask_local
                # And (t_masks[i] & ~new_mask) == 0 means all bits of t_mask are in new_mask
                # So the new bits in t_mask relative to block_mask_local must all be {x}
                # i.e., (t_masks[i] & ~block_mask_local) == (1 << x)  ... no, that's too restrictive
                # Let me just compute directly:
                # gain = count of i where uncovered[i] and (t_masks[i] & ~new_mask) == 0 and (t_masks[i] & ~block_mask_local) != 0
                
                # Vectorized:
                covered_by_new = (t_masks & ~new_mask) == 0
                not_covered_by_old = (t_masks & ~block_mask_local) != 0
                gain = np.count_nonzero(uncovered & covered_by_new & not_covered_by_old)
=======
                new_mask = block_mask_local | (1 << x)
                # Count how many uncovered t-subsets are subsets of new_mask but not block_mask_local
                # A t-subset is newly covered if: uncovered[i] and (t_masks[i] & ~new_mask) == 0 and (t_masks[i] & ~block_mask_local) != 0
                
                # Vectorized:
                # Mask the complements to avoid Python negative int to uint64 overflow
                new_mask_complement = (~new_mask) & valid_mask
                old_mask_complement = (~block_mask_local) & valid_mask
                covered_by_new = (t_masks & new_mask_complement) == 0
                not_covered_by_old = (t_masks & old_mask_complement) != 0
                gain = np.count_nonzero(uncovered & covered_by_new & not_covered_by_old)
>>>>>>> REPLACE
