        def dfs(row_idx, col_idx, current_row, local_diffs, used_diffs):
            # ...
            if col_idx == k + 1:
                rows.append(current_row)
                new_used = used_diffs | local_diffs
                res, rows_found = dfs(row_idx + 1, 0, [], set(), new_used)
                if res:
                    return True, rows_found
                else:
                    rows.pop()
                    return False, None
            # ...
                res, rows_found = dfs(row_idx, col_idx + 1, new_current_row, new_local_diffs, used_diffs)
            # ...
