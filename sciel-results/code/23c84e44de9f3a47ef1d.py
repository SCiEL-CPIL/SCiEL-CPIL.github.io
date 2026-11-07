        def dfs(k, used):
            if k == 12: return True
            i = order[k]
            for v in cands[i]:
                if v not in used:
                    used.add(v)
                    if dfs(k+1, used): return True
                    used.remove(v)
            return False
        