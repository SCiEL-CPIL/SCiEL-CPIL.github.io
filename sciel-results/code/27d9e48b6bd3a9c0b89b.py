import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    period = reduce(lcm, divisors)
    
    start_time = time.time()
    deadline = start_time + 85
    
    d_np = np.array(divisors)
    d_col = d_np[:, None]
    
    for W in range(n - 1, 100):
        if time.time() > deadline:
            break
            
        j_indices = np.arange(W + 1)
        
        for L in range(1, period + 1):
            if time.time() > deadline:
                break
                
            # Quick feasibility check
            feasible = True
            for d in divisors:
                k = (L + d - 1) // d
                if k * d > L + W:
                    feasible = False
                    break
            if not feasible:
                continue
                
            mod_matrix = (L + j_indices[None, :]) % d_col
            cost = np.where(mod_matrix == 0, 0, np.inf)
            
            row_ind, col_ind = linear_sum_assignment(cost)
            
            if np.sum(cost[row_ind, col_ind]) == 0:
                result = [0] * n
                for i in range(n):
                    result[i] = L + col_ind[i]
                
                if len(set(result)) == n and all(1 <= x <= 1000000 for x in result) and all(result[i] % divisors[i] == 0 for i in range(n)):
                    return [int(x) for x in result]
            
    return divisors[:]
