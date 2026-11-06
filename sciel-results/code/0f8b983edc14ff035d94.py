import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    weights = [Fraction(0)] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = Fraction(1, i)
        
    # Precompute conflict pairs for each x: (a, b) where a < b < x and a + x = 2*b
    conflict_pairs = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, x):
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if 1 <= b < x and b > a:
                    conflict_pairs[x].append((a, b))
    
    # Suffix sums using Fraction for exact upper bounds
    suffix = [Fraction(0)] * (N + 2)
    for i in range(N, 0, -1):
        suffix[i] = suffix[i + 1] + weights[i]
        
    best_score = Fraction(0)
    best_set = []
    
    in_S = [False] * (N + 1)
    S_list = []
    
    # Greedy initialization to get a decent starting bound
    for i in range(1, N + 1):
        valid = True
        for (a, b) in conflict_pairs[i]:
            if in_S[a] and in_S[b]:
                valid = False
                break
        if valid:
            in_S[i] = True
            S_list.append(i)
            
    if S_list:
        best_score = sum(weights[x] for x in S_list)
        best_set = list(S_list)
        
    # Reset for B&B
    in_S = [False] * (N + 1)
    S_list = []
    
    def is_valid_add(x):
        for (a, b) in conflict_pairs[x]:
            if in_S[a] and in_S[b]:
                return False
        return True

    def dfs(idx, current_sum):
        nonlocal best_score, best_set
        
        # Time check
        if time.time() > deadline:
            return
            
        # Pruning: even if we take all remaining, we can't beat best
        if current_sum + suffix[idx] <= best_score:
            return
            
        if idx > N:
            if current_sum > best_score:
                best_score = current_sum
                best_set = list(S_list)
            return
            
        x = idx
        
        # Option 1: Include x
        if is_valid_add(x):
            in_S[x] = True
            S_list.append(x)
            
            dfs(idx + 1, current_sum + weights[x])
            
            S_list.pop()
            in_S[x] = False
            
        # Option 2: Exclude x
        dfs(idx + 1, current_sum)

    dfs(1, Fraction(0))
    
    # Hill-climbing polish: try to add any element that fits
    if best_set:
        in_S = [False] * (N + 1)
        for x in best_set:
            in_S[x] = True
        improved = True
        while improved:
            improved = False
            for x in range(1, N + 1):
                if not in_S[x] and is_valid_add(x):
                    in_S[x] = True
                    best_set.append(x)
                    improved = True
                    break
    
    return list(best_set)
