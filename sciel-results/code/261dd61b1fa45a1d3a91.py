import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

def construct():
    n = 256
    # Generate all 3-term arithmetic progressions
    # A 3-AP is (a, b, c) such that b-a = c-b = d > 0
    # a >= 1, c <= n
    # a, a+d, a+2d
    progressions = []
    for d in range(1, n // 2 + 1):
        max_a = n - 2 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            progressions.append((a - 1, b - 1, c - 1)) # 0-indexed

    # Number of constraints
    m = len(progressions)
    
    # Objective: maximize sum(1/i * x_i)
    # milp minimizes, so we use -1/i
    c_obj = -np.array([1.0 / i for i in range(1, n + 1)])
    
    # Constraint matrix A
    # Each row corresponds to a progression (a, b, c)
    # Constraint: x_a + x_b + x_c <= 2
    # We build a sparse-like structure or just a dense matrix since m*n is small enough
    # m is ~16k, n is 256. 16k * 256 * 8 bytes ~ 32 MB. Dense is fine.
    A = np.zeros((m, n), dtype=np.int8)
    for i, (a, b, c) in enumerate(progressions):
        A[i, a] = 1
        A[i, b] = 1
        A[i, c] = 1
        
    # Constraints: A x <= 2
    # Lower bound is -inf, upper bound is 2
    linear_constraint = LinearConstraint(A, lb=-np.inf, ub=2)
    
    # Variable bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are integers
    integrality = np.ones(n)
    
    # Initial guess: Greedy solution
    # This helps the solver start from a good feasible point
    x0 = np.zeros(n)
    selected_greedy = []
    for x in range(n):
        # Check if adding x (1-indexed x+1) creates a 3-AP
        # We check all pairs (a, b) in selected_greedy
        # If 2*b - a == x+1 or 2*a - b == x+1 or ...
        # Actually, simpler: check if x+1 forms a 3-AP with any two existing elements
        # Or just use the logic from the seed but adapted for 0-indexed internal check
        # Let's just reuse the seed logic for x0 generation
        pass
        
    # Re-implementing seed logic for x0
    selected_greedy = []
    for x in range(1, n + 1):
        valid = True
        for i in range(len(selected_greedy)):
            for j in range(i + 1, len(selected_greedy)):
                a = selected_greedy[i]
                b = selected_greedy[j]
                # Check if x completes a 3-AP with a, b
                # Cases:
                # 1. a, b, x -> 2b - a = x
                # 2. a, x, b -> 2x - a = b -> x = (a+b)/2
                # 3. x, a, b -> 2a - x = b -> x = 2a - b
                # 4. a, b, x is covered by 1.
                # 5. b, a, x? No, a<b.
                # The condition "no 3-AP" means for any distinct a,b,c in set, they don't form AP.
                # When adding x, we check if it forms AP with any pair (a,b) in set.
                # Pairs are unordered, but AP is ordered.
                # If 2*b - a == x: AP (a, b, x)
                # If 2*a - b == x: AP (x, b, a) -> (x, b, a) is AP if 2b-x=a? No.
                # Let's just check all permutations of (a, b, x) for AP property.
                # Or simpler: check if x is in the set of "forbidden" values given the current set.
                # Forbidden values y such that (a, y, b) or (a, b, y) or (y, a, b) is AP.
                # (a, y, b): 2y = a+b => y = (a+b)/2. If a+b even.
                # (a, b, y): 2b = a+y => y = 2b-a.
                # (y, a, b): 2a = y+b => y = 2a-b.
                
                if 2 * b - a == x:
                    valid = False
                    break
                if (a + b) % 2 == 0 and (a + b) // 2 == x:
                    valid = False
                    break
                if 2 * a - b == x:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            selected_greedy.append(x)
            x0[x - 1] = 1

    # Run MILP
    # Time limit: 50 seconds to be safe
    res = milp(c=c_obj, constraints=linear_constraint, integrality=integrality, bounds=bounds, x0=x0, options={"time_limit": 50})
    
    if res.status == 0: # Optimal
        # res.x is the optimal solution
        selected = [i + 1 for i in range(n) if res.x[i] > 0.5]
    elif res.status == 1: # Iteration limit reached (time limit)
        # res.x contains the best incumbent found
        if res.x is not None:
            selected = [i + 1 for i in range(n) if res.x[i] > 0.5]
        else:
            selected = selected_greedy
    else:
        # Other errors, fall back to greedy
        selected = selected_greedy
        
    # Ensure non-empty
    if not selected:
        selected = [1]
        
    return selected

construct()
