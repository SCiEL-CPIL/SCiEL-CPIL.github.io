# EVOLVE-BLOCK-START
import time
from fractions import Fraction

DENOMINATORS = list(range(2, 49))
N = len(DENOMINATORS)
ONE = Fraction(1, 1)
ZERO = Fraction(0, 1)

def construct():
    deadline = time.time() + 80
    
    dp = [set() for _ in range(N + 1)]
    dp[0].add(ZERO)
    
    for i in range(N):
        if time.time() > deadline:
            break
        d = DENOMINATORS[i]
        inv_d = Fraction(1, d)
        dp[i+1] = set(dp[i])
        for s in dp[i]:
            ns = s + inv_d
            if ns < ONE:
                dp[i+1].add(ns)
                
    if not dp[N]:
        return [0]
        
    best_sum = max(dp[N])
    current_sum = best_sum
    indices = []
    
    for i in range(N, 0, -1):
        d = DENOMINATORS[i-1]
        inv_d = Fraction(1, d)
        if current_sum in dp[i-1]:
            pass
        else:
            indices.append(i-1)
            current_sum -= inv_d
            
    return indices
# EVOLVE-BLOCK-END
