"""
Strategy: Maximum Independent Set via Integer Linear Programming (ILP).
The problem is to find the largest subset of [1, 180] with no three distinct
elements a, b, c satisfying 1/a = 1/b + 1/c.
This is equivalent to finding the maximum independent set in a 3-uniform
hypergraph where vertices are the numbers and edges are the forbidden triples.
We use scipy.optimize.milp to solve the ILP formulation.
"""
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 180
    numbers = list(range(1, N + 1))
    
    # 1. Generate all forbidden triples
    # 1/a = 1/b + 1/c  =>  a = bc / (b+c)
    # Parameterization: b = kx(x+y), c = ky(x+y), a = kxy
    # with gcd(x,y)=1, x!=y, and k >= 1.
    # Constraints: a,b,c <= N.
    triples = []
    # x, y are small. (x+y)*max(x,y) <= 180.
    # max(x,y) <= sqrt(180) ~ 13.4. So x,y <= 14 is sufficient.
    # We use 20 to be safe.
    max_xy = 20
    for x in range(1, max_xy + 1):
        for y in range(1, max_xy + 1):
            if x == y:
                continue
            if math.gcd(x, y) != 1:
                continue
            M = (x + y) * max(x, y)
            if M > N:
                continue
            max_k = N // M
            for k in range(1, max_k + 1):
                a = k * x * y
                b = k * x * (x + y)
                c = k * y * (x + y)
                if a <= N and b <= N and c <= N:
                    # Store 0-indexed indices
                    triples.append((a - 1, b - 1, c - 1))
    
    # If no triples, any subset is valid, so return all
    if not triples:
        return numbers

    # 2. Solve ILP
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except ImportError:
        # Fallback if SciPy is not available
        return _fallback_heuristic(numbers, triples)

    n_vars = N
    # Objective: Maximize sum(x_i)  <=>  Minimize -sum(x_i)
    c = -np.ones(n_vars)
    
    # Constraints: For each triple (u,v,w), x_u + x_v + x_w <= 2
    n_cons = len(triples)
    A = np.zeros((n_cons, n_vars))
    for i, (u, v, w) in enumerate(triples):
        A[i, u] = 1
        A[i, v] = 1
        A[i, w] = 1
        
    constraints = [
        LinearConstraint(A, ub=2)
    ]
    
    integrality = np.ones(n_vars)
    bounds = Bounds(lb=0, ub=1)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        # Check if we are running out of time
        if time.time() > deadline:
            return _fallback_heuristic(numbers, triples)
            
        if res.status == 0:  # Optimal
            x_sol = res.x
            # Threshold to handle floating point errors
            selected = [int(i) + 1 for i in range(n_vars) if x_sol[i] > 0.5]
            if selected:
                return selected
            else:
                # Should not happen, but return seed if empty
                return list(range(91, 181))
        
        elif res.status == 2:  # Infeasible (should not happen, empty set is always feasible)
            return list(range(91, 181))
            
        else:
            # Other status codes, try fallback
            pass
            
    except Exception:
        pass

    # Fallback if ILP failed or timed out
    return _fallback_heuristic(numbers, triples)

def _fallback_heuristic(numbers, triples):
    # Simple greedy: Start with all, remove elements involved in conflicts.
    # This is a heuristic and may not be optimal, but better than nothing.
    # Since the ILP is expected to work, this is just a safety net.
    # We can also just return the seed (91-180) which is known to be valid.
    return list(range(91, 181))
