import itertools
import sys

def construct(n):
    if n == 1:
        return "1"
    
    all_perms = set(itertools.permutations(range(1, n + 1)))
    N = len(all_perms)
    
    sys.setrecursionlimit(10000)
    
    def dfs(current, seen, depth=0):
        if len(seen) == N:
            return current
        
        suffix = current[-(n-1):]
        candidates = list(range(1, n + 1))
        
        def lru_key(sym):
            if sym in suffix:
                return suffix.index(sym)
            else:
                return -1
        
        candidates.sort(key=lru_key)
        
        for sym in candidates:
            new_window = tuple(suffix + [sym])
            if new_window not in seen and len(set(new_window)) == n:
                result = dfs(current + [sym], seen | {new_window}, depth + 1)
                if result is not None:
                    return result
        
        return None
    
    initial = list(range(1, n + 1))
    result = dfs(initial, {tuple(initial)})
    
    if result is None:
        return "".join(map(str, initial))
    
    return "".join(map(str, result))
