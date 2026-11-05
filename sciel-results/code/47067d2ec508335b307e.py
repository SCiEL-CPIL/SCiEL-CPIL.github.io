        def dfs(idx, S, used_diffs, need):
            if not need:
                return list(S)
            if len(S) > 32:
                return None
            if idx == len(candidates):
                return None
            
            rem = len(candidates) - idx
            max_new = rem * (rem - 1) // 2 + rem * len(S)
            if len(need) > max_new:
                return None

            x = candidates[idx]
            
            # Try adding x
            valid = True
            new_diffs = []
            for s in S:
                d = x - s
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                new_used = used_diffs.union(new_diffs)
                new_need = need.difference(new_used)
                res = dfs(idx + 1, S + [x], new_used, new_need)
                if res is not None:
                    return res

            # Try skipping x
            return dfs(idx + 1, S, used_diffs, need)
