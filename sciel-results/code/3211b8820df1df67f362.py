# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def _is_smooth(n):
    y = n
    for p in (2, 3, 5):
        while y % p == 0:
            y //= p
    return y == 1

def _build_smooth_list(limit):
    return [x for x in range(1, limit + 1) if _is_smooth(x)]

def _precompute_optimal(limit=128):
    smooth = _build_smooth_list(limit)
    n = len(smooth)
    
    # Precompute incompatibility: i and j are incompatible if one divides the other
    incompat = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and (smooth[j] % smooth[i] == 0 or smooth[i] % smooth[j] == 0):
                incompat[i][j] = True
    
    # For each target, find minimum antichain size
    # Use iterative deepening DFS with good pruning
    
    def find_min_antichain(target):
        if target == 0:
            return []
        # Check size 1
        for s in smooth:
            if s == target:
                return [s]
        
        # DFS with size limit
        def dfs(start, rem, chosen, max_size):
            if rem == 0:
                return chosen[:]
            if len(chosen) >= max_size:
                return None
            if rem < 0:
                return None
            
            # Pruning: need at least (rem // max_smooth) more terms, but that's weak
            # Better: the smallest possible next term is smooth[start] if start < n
            # We need to be able to fill rem with at most (max_size - len(chosen)) terms
            # Each term is at most smooth[-1] = limit, so we need rem <= (max_size - len(chosen)) * limit
            # That's always true since rem <= target <= limit
            
            # Stronger pruning: sum of chosen + rem = target, and we need distinct antichain terms
            # The minimum sum of k more terms (antichain) is at least 1+2+...+k = k(k+1)/2
            # But antichain terms don't have to be consecutive small numbers
            
            for i in range(start, n):
                s = smooth[i]
                if s > rem:
                    break
                # Check compatibility
                ok = True
                for c in chosen:
                    ci = smooth.index(c)
                    if incompat[ci][i]:
                        ok = False
                        break
                if not ok:
                    continue
                
                chosen.append(s)
                res = dfs(i + 1, rem - s, chosen, max_size)
                if res is not None:
                    return res
                chosen.pop()
            
            return None
        
        for size in range(2, 31):
            res = dfs(0, target, [], size)
            if res is not None:
                return res
        return None
    
    # Use index-based for speed
    def find_min_antichain_idx(target):
        if target == 0:
            return []
        for i, s in enumerate(smooth):
            if s == target:
                return [s]
        
        def dfs(start, rem, chosen_idx, max_size):
            if rem == 0:
                return [smooth[i] for i in chosen_idx]
            if len(chosen_idx) >= max_size:
                return None
            if rem < 0:
                return None
            
            for i in range(start, n):
                s = smooth[i]
                if s > rem:
                    break
                ok = True
                for ci in chosen_idx:
                    if incompat[ci][i]:
                        ok = False
                        break
                if not ok:
                    continue
                
                chosen_idx.append(i)
                res = dfs(i + 1, rem - s, chosen_idx, max_size)
                if res is not None:
                    return res
                chosen_idx.pop()
            
            return None
        
        for size in range(2, 31):
            res = dfs(0, target, [], size)
            if res is not None:
                return res
        return None
    
    start = time.time()
    deadline = start + 45
    
    optimal = [None] * (limit + 1)
    for t in range(1, limit + 1):
        if time.time() > deadline:
            break
        optimal[t] = find_min_antichain_idx(t)
    
    # Fill missing
    for t in range(1, limit + 1):
        if optimal[t] is None:
            optimal[t] = find_min_antichain_idx(t)
    
    return optimal

def construct():
    return _precompute_optimal(128)
# EVOLVE-BLOCK-END
