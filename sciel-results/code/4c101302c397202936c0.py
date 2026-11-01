                for v in range(lo, old_val):
                    if not check_move_conflict_fast(cur, row_diff_sets, ri, pos, v):
                        continue
                    # Tentative move
                    row[pos] = v
                    row_diff_sets[ri] = build_row_diff_set(row)
                    new_scope = get_scope(cur)
                    accept = False
                    if new_scope < cur_scope:
                        accept = True
                        improved = True
                        moved = True
                        cur_scope = new_scope
                    elif new_scope == cur_scope:
                        if rng.random() < 0.05:
                            accept = True
                            improved = True
                            moved = True
                    
                    if accept:
                        break
                    else:
                        # Revert
                        row[pos] = old_val
                        row_diff_sets[ri] = build_row_diff_set(row)
