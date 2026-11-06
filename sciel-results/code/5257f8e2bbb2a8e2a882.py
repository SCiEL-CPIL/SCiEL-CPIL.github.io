import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit
    
    N = 256
    weights = [Fraction(0)] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = Fraction(1, i)
        
    # Suffix sums for pruning (exact Fraction)
    # suffix[i] = sum of weights from i to N
    suffix = [Fraction(0)] * (N + 2)
    for i in range(N, 0, -1):
        suffix[i] = suffix[i + 1] + weights[i]
        
    best_score = Fraction(0)
    best_set = []
    
    in_S = [False] * (N + 1)
    S_list = []
    
    # Greedy initialization to get a decent starting bound
    # This helps pruning significantly
    for i in range(1, N + 1):
        valid = True
        for a in S_list:
            # Check if a, b, i form AP with i largest
            # 2b = a + i => b = (a+i)/2
            s = a + i
            if s % 2 == 0:
                b = s // 2
                if in_S[b]:
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
        # Check if adding x creates a 3-AP with existing elements
        # Since x is the largest in the current prefix, we only check patterns (a, b, x)
        # where a, b are in S and a < b < x.
        # This is equivalent to checking if (a+x)/2 is in S for any a in S.
        for a in S_list:
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if in_S[b]:
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
    
    return list(best_set)
