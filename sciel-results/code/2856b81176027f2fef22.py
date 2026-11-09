import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin for 90s limit

    n = 128
    
    # Precompute constraints
    constraints = [[] for _ in range(n)]
    for i in range(n):
        for d in range(1, i // 3 + 1):
            a = i + 1 - 3 * d
            if a < 1:
                break
            idx_a = a - 1
            idx_ad = a + d - 1
            idx_a2d = a + 2 * d - 1
            constraints[i].append((idx_a, idx_ad, idx_a2d))

    def search_backtracking(max_c, time_limit, seed):
        random.seed(seed)
        colors = [-1] * n
        
        def backtrack(idx):
            if time.time() > start_time + time_limit:
                return False
            
            if idx == n:
                return True
            
            forbidden = [False] * max_c
            for i1, i2, i3 in constraints[idx]:
                c1 = colors[i1]
                c2 = colors[i2]
                c3 = colors[i3]
                
                if c1 != -1 and c2 != -1 and c3 != -1:
                    if c1 == c2 and c2 == c3:
                        forbidden[c1] = True
                    else:
                        present = {c1, c2, c3}
                        if len(present) == 2:
                            for c in present:
                                forbidden[c] = True
            
            order = list(range(max_c))
            random.shuffle(order)
            
            for c in order:
                if not forbidden[c]:
                    colors[idx] = c
                    if backtrack(idx + 1):
                        return True
                    colors[idx] = -1
            return False

        if backtrack(0):
            return colors
        return None

    # Try 3 colors with multiple restarts
    # 20 restarts, 2 seconds each? 40s total.
    # If found, return.
    
    attempts = 20
    for i in range(attempts):
        if time.time() > deadline - 10.0: # Leave 10s for 4-color fallback
            break
        limit = 2.0
        res = search_backtracking(3, limit, seed=i)
        if res:
            return res

    # Try 4 colors
    if time.time() < deadline:
        remaining = deadline - time.time()
        # Use a single longer run for 4 colors
        res = search_backtracking(4, remaining, seed=100)
        if res:
            return res

    # Fallback greedy 4 colors
    def try_greedy_4():
        colors = [-1] * n
        for i in range(n):
            forbidden = [False] * 4
            for i1, i2, i3 in constraints[i]:
                c1, c2, c3 = colors[i1], colors[i2], colors[i3]
                if c1 != -1 and c2 != -1 and c3 != -1:
                    if c1 == c2 and c2 == c3:
                        forbidden[c1] = True
                    else:
                        present = {c1, c2, c3}
                        if len(present) == 2:
                            for c in present:
                                forbidden[c] = True
            for c in range(4):
                if not forbidden[c]:
                    colors[i] = c
                    break
            if colors[i] == -1:
                return None
        return colors

    res = try_greedy_4()
    if res:
        return res

    return [0]*128

