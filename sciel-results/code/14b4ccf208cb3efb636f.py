                local_diffs.add(d)
                # ...
                res = backtrack(...)
                if res: return res
                # Undo
                for prev in current_row:
                    local_diffs.remove(x - prev)
