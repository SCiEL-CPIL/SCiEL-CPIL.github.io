import time
from math import factorial

_START_TIME = time.time()
_TIMEOUT = 85

def _time_left():
    return _TIMEOUT - (time.time() - _START_TIME)

def construct():
    target = factorial(12)  # 479001600
    
    # Precompute all divisors of target that are > 12
    all_divs = []
    limit = int(target**0.5) + 1
    for i in range(1, limit):
        if target % i == 0:
            if i > 12:
                all_divs.append(i)
            j = target // i
            if j > 12 and j != i:
                all_divs.append(j)
    all_divs = sorted(set(all_divs))
    
    # Search for minimal M
    for M in range(24, 300):
        if _time_left() < 3:
            break
        
        valid = [d for d in all_divs if d <= M]
        
        found = [None]
        
        def dfs(rem, min_idx, path):
            if _time_left() < 0.2:
                return
            if rem == 1:
                if path:
                    found[0] = list(path)
                return
            if min_idx >= len(valid):
                return
            
            for i in range(min_idx, len(valid)):
                d = valid[i]
                if d > rem:
                    break
                if rem % d != 0:
                    continue
                
                path.append(d)
                dfs(rem // d, i + 1, path)
                if found[0] is not None:
                    path.pop()
                    return
                path.pop()
        
        dfs(target, 0, [])
        if found[0] is not None:
            return found[0]
    
    return [target]
