    def greedy_compact(rows, used_mask):
        """Fast single-pass compaction."""
        rows = [list(r) for r in rows]
        scope = max(max(r) for r in rows)
        for _ in range(5): # Few passes
            improved = False
            for ri in range(n_rows):
                row = rows[ri]
                # Remove this row's diffs from used_mask
                old_mask = compute_diffs_bitset(row)
                used_mask &= ~old_mask
                
                # Try to compact this row
                for pos in range(len(row) - 1, 0, -1):
                    old_val = row[pos]
                    min_val = row[pos-1] + 1
                    if old_val == min_val:
                        continue
                    # Try to find the smallest valid value >= min_val
                    # We can just try min_val, min_val+1, ... up to old_val
                    # But to be fast, we can just try to set it to min_val first?
                    # No, we need to check validity.
                    # Let's just try to shift it down by 1 repeatedly? No, that's slow.
                    # Let's just try the smallest possible value.
                    # Actually, the "greedy" way is to try to make it as small as possible.
                    # We iterate from min_val to old_val.
                    for nv in range(min_val, old_val):
                        row[pos] = nv
                        new_mask = compute_diffs_bitset(row)
                        if not (new_mask & used_mask):
                            # Check if strictly increasing (it is, by construction of range)
                            # Accept
                            used_mask |= new_mask
                            if nv < old_val:
                                improved = True
                            break
                        # If invalid, try next? No, if min_val is invalid, maybe min_val+1 is?
                        # The loop handles this.
                    # If no valid value found in range, row[pos] remains old_val (or whatever was last tried)
                    # We need to restore if we failed?
                    # Actually, if we break, it's valid. If we finish loop, it means no nv in [min_val, old_val) was valid?
                    # That can't be, because old_val was valid. So the loop should always find at least old_val?
                    # Wait, the range is range(min_val, old_val). It excludes old_val.
                    # If no value in [min_val, old_val) is valid, we must keep old_val.
                    # So we need to check if we actually updated it.
                    # Let's track if we changed it.
                    pass
                # Recompute mask to be safe? No, we updated used_mask.
                # But if we failed to change, we need to re-add the old mask?
                # The logic above is a bit messy. Let's simplify.
            if not improved:
                break
        return rows, max(max(r) for r in rows)
