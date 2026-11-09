            candidates = []
            for x in range(last_elem + 1, D + 1):
                new_diffs_mask = 0
                valid = True
                for y in elements:
                    d = x - y
                    bit = 1 << d
                    if used_diffs & bit:
                        valid = False
                        break
                    new_diffs_mask |= bit
                
                if valid:
                    new_covered = (new_diffs_mask & TARGET_MASK) & ~covered
                    score = bin(new_covered).count('1')
                    candidates.append((score, x, new_diffs_mask))
            
            # Sort by score descending
            candidates.sort(key=lambda item: -item[0])
            
            for score, x, new_diffs_mask in candidates:
                new_used = used_diffs | new_diffs_mask
                new_elements = elements + [x]
                res = dfs(new_elements, new_used, x)
                if res is not None:
                    return res
