import time
import sys

def _compute_table():
    """Pre-compute minimal antichain representations for targets 1..128."""
    start_time = time.time()
    deadline = 85.0  # Leave 5s safety margin before 90s kill
    
    # Generate all 2,3,5-smooth numbers <= 128
    smooth = set()
    for a in range(8):
        for b in range(5):
            for c in range(4):
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if val <= 128:
                    smooth.add(val)
    
    smooth_list = sorted(smooth)
    n = len(smooth_list)
    
    # Precompute divisibility: divs[i][j] = True if smooth_list[i] divides smooth_list[j]
    # and smooth_list[i] != smooth_list[j]
    divs = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and smooth_list[j] % smooth_list[i] == 0:
                divs[i][j] = True
    
    reps = [None] * 129
    
    # k=1: smooth numbers
    for t in range(1, 129):
        if t in smooth:
            reps[t] = (t,)
    
    # k=2: pairs
    for t in range(1, 129):
        if reps[t] is not None:
            continue
        for i in range(n):
            a = smooth_list[i]
            if a >= t:
                break
            b = t - a
            if b < a:
                continue
            if b in smooth:
                # Check antichain: a and b must not divide each other
                # Find index of b
                b_idx = smooth_list.index(b) if b in smooth_list else -1
                if b_idx >= 0 and not divs[i][b_idx] and not divs[b_idx][i]:
                    reps[t] = (a, b)
                    break
    
    # k=3: triples
    for t in range(1, 129):
        if reps[t] is not None:
            continue
        found = False
        for i in range(n):
            a = smooth_list[i]
            if a >= t:
                break
            for j in range(i + 1, n):
                b = smooth_list[j]
                if a + b >= t:
                    break
                c = t - a - b
                if c <= b:
                    continue
                if c in smooth:
                    c_idx = smooth_list.index(c)
                    if c_idx > j:
                        # Check antichain: no pair divides another
                        if not divs[i][j] and not divs[j][i] and \
                           not divs[i][c_idx] and not divs[c_idx][i] and \
                           not divs[j][c_idx] and not divs[c_idx][j]:
                            reps[t] = (a, b, c)
                            found = True
                            break
            if found:
                break
    
    # Check if we need to do DFS for k>=4
    unresolved = [t for t in range(1, 129) if reps[t] is None]
    
    if unresolved:
        # Use DFS with bitmask for antichain checking
        # For each target, try k=4,5,6,... until found
        
        def dfs(rem, start_idx, current_terms, k_remaining):
            if time.time() > deadline:
                return None
            if k_remaining == 0:
                return current_terms if rem == 0 else None
            if rem <= 0:
                return None
            if start_idx >= n:
                return None
            
            # Pruning: need at least k_remaining terms from start_idx onward
            if n - start_idx < k_remaining:
                return None
            
            # Pruning: minimum possible sum
            if k_remaining <= n - start_idx:
                min_possible = sum(smooth_list[start_idx:start_idx + k_remaining])
                if min_possible > rem:
                    return None
            
            # Pruning: maximum possible sum
            max_start = n - k_remaining
            if max_start >= start_idx:
                max_possible = sum(smooth_list[max_start:max_start + k_remaining])
                if max_possible < rem:
                    return None
            
            for idx in range(start_idx, n):
                val = smooth_list[idx]
                if val > rem:
                    break
                valid = True
                for existing_idx, existing_val in current_terms:
                    if val % existing_val == 0 or existing_val % val == 0:
                        valid = False
                        break
                if not valid:
                    continue
                res = dfs(rem - val, idx + 1, current_terms + [(idx, val)], k_remaining - 1)
                if res is not None:
                    return res
            return None
        
        for t in unresolved:
            if time.time() > deadline:
                break
            for k in range(4, 11):
                res = dfs(t, 0, [], k)
                if res is not None:
                    reps[t] = tuple(v for _, v in res)
                    break
    
    # Fallback: if any targets still unresolved, use a more thorough DFS
    for t in range(1, 129):
        if reps[t] is None:
            found = False
            for k in range(4, 31):
                if time.time() > deadline:
                    break
                res = dfs(t, 0, [], k)
                if res is not None:
                    reps[t] = tuple(v for _, v in res)
                    found = True
                    break
    
    return reps

_TABLE = _compute_table()

def construct():
    return [list(_TABLE[t]) for t in range(1, 129)]
