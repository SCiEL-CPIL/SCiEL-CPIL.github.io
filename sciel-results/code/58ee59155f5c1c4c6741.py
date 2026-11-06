            # Sort l_sums descending to allow early termination
            # But we need to keep masks aligned.
            l_pairs = sorted(zip(l_sums, l_masks), key=lambda x: x[0], reverse=True)
            max_r = r_sums_sorted[-1] if r_sums_sorted else 0
            
            for ls, lm in l_pairs:
                if not time_ok():
                    break
                if ls + max_r <= best_total:
                    break # No further ls can improve
            
                limit = LCM - 1 - ls
                if limit < 0:
                    continue
                
                idx = bisect.bisect_right(r_sums_sorted, limit)
                if idx == 0:
                    continue
                
                cand_rs = r_sums_sorted[idx - 1]
                cand_rm = r_masks_sorted[idx - 1]
                
                total = ls + cand_rs
                if total > best_total:
                    best_total = total
                    best_l_mask = lm
                    best_r_mask = cand_rm
