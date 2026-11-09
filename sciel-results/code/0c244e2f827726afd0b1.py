"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    def elapsed():
        return time.time() - start_time
    
    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    
    # Precompute conflicts
    # conflict[i][j] is True if lcm(numbers[i], numbers[j]) <= 256
    # We need to select a subset S such that no two elements in S are in conflict.
    # This is a Maximum Weight Independent Set problem.
    
    # Build constraint matrix A and vector b for ILP
    # Variables x_i in {0, 1}
    # Constraints: x_i + x_j <= 1 for all (i, j) with conflict
    # Objective: Maximize sum( (1/numbers[i]) * x_i )
    
    # Collect conflicts
    conflicts = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = numbers[i], numbers[j]
            l = a * b // gcd(a, b)
            if l <= N:
                conflicts.append((i, j))
    
    # If no conflicts, take all numbers
    if not conflicts:
        return numbers
    
    # Build A (m x n) where m = len(conflicts)
    # Each row corresponds to a constraint x_i + x_j <= 1
    m = len(conflicts)
    A = np.zeros((m, n))
    for k, (i, j) in enumerate(conflicts):
        A[k, i] = 1
        A[k, j] = 1
    
    # Objective: minimize -sum(w_i x_i)
    # w_i = 1 / numbers[i]
    w = np.array([1.0 / numbers[i] for i in range(n)])
    c = -w
    
    # Constraints: A x <= 1
    # LinearConstraint(A, lb=-np.inf, ub=1)
    constraints = LinearConstraint(A, lb=-np.inf, ub=1)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are integers
    integrality = np.ones(n)
    
    try:
        # Solve ILP
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.success and res.x is not None:
            # Extract selected indices
            selected_indices = [i for i in range(n) if res.x[i] > 0.5]
            result = sorted(numbers[i] for i in selected_indices)
            
            # Validate
            valid = True
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    a, b = result[i], result[j]
                    l = a * b // gcd(a, b)
                    if l <= 256:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and result:
                return result
        
        # If ILP failed or result invalid, fall through to greedy
    except Exception:
        pass
    
    # Fallback: Greedy heuristic
    # Sort by descending weight (ascending number) to pick "valuable" small numbers first?
    # Actually, picking smaller numbers gives higher weight 1/a.
    # But smaller numbers conflict with more others.
    # A better greedy for independent set: pick node with max weight, remove it and its neighbors, repeat.
    
    # To be safe and fast, let's just pick a valid set.
    # Greedy: iterate through numbers, add if no conflict with current set.
    # Order: Try to maximize sum. 
    # A simple effective heuristic: Sort by 1/a (descending, so a ascending).
    # But this might get stuck in a local optimum.
    # Let's try a few orders or just a robust one.
    # Given the time limit, a simple greedy is safe.
    
    # Let's try greedy by descending weight (smallest a first)
    # This might yield a smaller set but higher weight?
    # Or maybe we should prefer larger numbers to avoid conflicts?
    # Larger numbers have lower weight but fewer conflicts.
    # Let's try greedy by ascending a (highest weight first) and see.
    
    result = []
    # Sort indices by weight descending (a ascending)
    indices = sorted(range(n), key=lambda i: numbers[i])
    
    for i in indices:
        if elapsed() > DEADLINE:
            break
        a = numbers[i]
        ok = True
        for b in result:
            if a * b // gcd(a, b) <= 256:
                ok = False
                break
        if ok:
            result.append(a)
            
    if result:
        return result
    
    # Last resort: return [256]
    return [256]

# EVOLVE-BLOCK-END
