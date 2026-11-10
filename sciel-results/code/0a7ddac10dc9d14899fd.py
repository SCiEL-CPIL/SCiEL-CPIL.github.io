# EVOLVE-BLOCK-START
import sys
import time
sys.setrecursionlimit(10000)

def construct():
    n = 192
    start_time = time.time()
    TIME_LIMIT = 50.0  # Leave 10s for overhead
    
    # Precompute ResidueMasks
    # ResidueMask[a][r] is a bitmask of x in 1..n such that x % a == r
    ResidueMask = [[0] * a for a in range(n + 1)]
    for a in range(1, n + 1):
        for x in range(1, n + 1):
            r = x % a
            ResidueMask[a][r] |= (1 << x)
    
    # Candidates in descending order
    candidates = list(range(n, 0, -1))
    
    def is_valid(a, S, M):
        # Check constraint for a: no triple in S sums to 0 mod a
        if len(S) >= 3:
            for i in range(len(S)):
                b = S[i]
                for j in range(i + 1, len(S)):
                    c = S[j]
                    r = (-b - c) % a
                    mask = ResidueMask[a][r]
                    inter = M & mask
                    inter &= ~(1 << b)
                    inter &= ~(1 << c)
                    if inter != 0:
                        return False
        
        # Check constraint for each b in S
        if len(S) >= 2:
            for b in S:
                target = (-a) % b
                for c in S:
                    if c == b:
                        continue
                    r_c = (target - c) % b
                    mask = ResidueMask[b][r_c]
                    inter = M & mask
                    inter &= ~(1 << b)
                    inter &= ~(1 << c)
                    if inter != 0:
                        return False
        
        return True

    best_set = []
    best_size = 0
    
    def backtrack(start_idx, S, M, current_size):
        nonlocal best_set, best_size
        
        if time.time() - start_time > TIME_LIMIT:
            return

        # Pruning
        if current_size + (len(candidates) - start_idx) <= best_size:
            return
            
        if current_size > best_size:
            best_size = current_size
            best_set = S[:]
            
        for i in range(start_idx, len(candidates)):
            a = candidates[i]
            
            if is_valid(a, S, M):
                S.append(a)
                M_new = M | (1 << a)
                backtrack(i + 1, S, M_new, current_size + 1)
                S.pop()
                
    backtrack(0, [], 0, 0)
    
    return best_set
# EVOLVE-BLOCK-END
