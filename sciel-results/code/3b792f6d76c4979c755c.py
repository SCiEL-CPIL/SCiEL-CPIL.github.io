            if lo <= hi:
                candidates = list(range(lo, hi + 1))
                # Try smaller values first to reduce max
                for v in candidates:
                    if check_and_apply(target_row, last_idx, v):
                        moved = True
                        break
