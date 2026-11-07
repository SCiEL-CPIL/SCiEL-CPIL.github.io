import math
import numpy as np
from scipy.optimize import linear_sum_assignment

P = {'n': 12, 'maximum_start': 1000000}

def construct():
    n = P['n']
    
    # Calculate LCM of 1..12
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    LCM = 1
    for i in range(1, n + 1):
        LCM = lcm(LCM, i)
        
    best_m = 0
    best_L = -1
    
    # We only need to check m in [0, LCM)
    for m in range(LCM):
        # Lower bound for L: max distance to next multiple
        min_L = 0
        for i in range(1, n + 1):
            next_mult = ((m // i) + 1) * i
            dist = next_mult - m
            if dist > min_L:
                min_L = dist
        
        # Check L from min_L upwards
        # Max L is bounded, say 200
        for L in range(min_L, 200):
            if L < n:
                continue
                
            # Create cost matrix 12 x L
            # 0 if divisible, large cost otherwise
            cost_matrix = np.full((n, L), 1000)
            
            for i in range(n):
                d = i + 1
                # Values in (m, m+L] are m+1, ..., m+L
                # Index j corresponds to value m + 1 + j
                for j in range(L):
                    val = m + 1 + j
                    if val % d == 0:
                        cost_matrix[i][j] = 0
            
            # Find minimum weight matching
            # linear_sum_assignment returns row_ind, col_ind
            # We need to check if the matching uses only 0-cost edges
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Check if all assigned pairs have cost 0
            if np.all(cost_matrix[row_ind, col_ind] == 0):
                # Valid matching found for this L
                if L > best_L:
                    best_L = L
                    best_m = m
                break
                
    return int(best_m)
