import time
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of n
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort()
    N = len(divisors)
    
    # ILP Formulation:
    # Minimize sum(x_i)
    # Subject to: sum(d_i * x_i) == target
    #             x_i in {0, 1}
    
    c = np.ones(N) # Objective coefficients
    A = np.array(divisors) # Constraint matrix (1 row)
    
    # Equality constraint: A @ x == target
    # LinearConstraint(A, target, target)
    
    # Bounds: 0 <= x_i <= 1
    lb = np.zeros(N)
    ub = np.ones(N)
    
    # Integrality: 0 for continuous, 1 for integer
    # We want binary, so integer is sufficient with bounds 0,1
    integrality = np.ones(N)
    
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, target, target),
            integrality=integrality,
            bounds=Bounds(lb, ub)
        )
        
        if res.status == 0:
            # res.x contains the solution
            selected = []
            for i in range(N):
                if res.x[i] > 0.5: # Binary check
                    selected.append(divisors[i])
            return selected
        else:
            # Fallback to DP if ILP fails
            pass
            
    except Exception:
        pass

    # Fallback DP
    INF = 10**9
    dp = [[INF] * (target + 1) for _ in range(N + 1)]
    dp[0][0] = 0
    
    for i in range(1, N + 1):
        d = divisors[i-1]
        for s in range(target + 1):
            val1 = dp[i-1][s]
            val2 = INF
            if s >= d:
                if dp[i-1][s-d] != INF:
                    val2 = dp[i-1][s-d] + 1
            dp[i][s] = min(val1, val2)
            
    if dp[N][target] == INF:
        return []

    selected = []
    s = target
    for i in range(N, 0, -1):
        d = divisors[i-1]
        if s >= d and dp[i][s] == dp[i-1][s-d] + 1:
            selected.append(d)
            s -= d
            
    return selected
