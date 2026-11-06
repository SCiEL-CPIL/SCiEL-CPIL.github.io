import time
import random
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    n = 256
    
    # 1. Generate all 3APs
    # Constraint: x_a + x_b + x_c <= 2 for all 3APs (a, b, c)
    # This is a linear constraint.
    # We want to maximize sum(1/x * x_x)
    # linprog minimizes c^T x. So c = -1/x.
    
    c = np.array([-1.0 / i for i in range(1, n + 1)])
    
    # Constraints A_ub x <= b_ub
    # For each 3AP (a, b, c), x_a + x_b + x_c <= 2
    # a, b, c are 1-indexed in math, 0-indexed in array.
    # a, b, c distinct.
    
    A_rows = []
    b_vals = []
    
    for d in range(1, (n - 1) // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c_val = a + 2 * d
            # Indices: a-1, b-1, c_val-1
            row = np.zeros(n)
            row[a - 1] = 1
            row[b - 1] = 1
            row[c_val - 1] = 1
            A_rows.append(row)
            b_vals.append(2)
            
    if not A_rows:
        # Fallback if no APs (impossible for n=256)
        return list(range(1, n + 1))
        
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    
    # Bounds: 0 <= x_i <= 1
    bounds = [(0, 1) for _ in range(n)]
    
    # Solve LP Relaxation
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status != 0:
            # Fallback to greedy if solver fails
            return _fallback_greedy(n, deadline, start_time)
        
        x_opt = res.x
    except Exception:
        return _fallback_greedy(n, deadline, start_time)
    
    # Round the LP solution
    # Sort by value descending (x_i close to 1)
    # We need a valid 3AP-free set.
    
    # Strategy: Greedy add based on LP value
    # Indices sorted by x_opt[i] descending
    indices = np.argsort(x_opt)[::-1]
    
    S = set()
    # Precompute conflicts for fast check
    # For each y, list of (a, b) such that {y, a, b} is 3AP
    conflict_pairs = [[] for _ in range(n + 1)]
    for y in range(1, n + 1):
        d = 1
        while y + 2 * d <= n:
            conflict_pairs[y].append((y + d, y + 2 * d))
            d += 1
        d = 1
        while y - d >= 1:
            conflict_pairs[y].append((y - d, y + d))
            d += 1
        d = 1
        while y - 2 * d >= 1:
            conflict_pairs[y].append((y - 2 * d, y - d))
            d += 1
            
    def is_valid_addition(S, y):
        for a, b in conflict_pairs[y]:
            if a in S and b in S:
                return False
        return True
        
    for idx in indices:
        y = idx + 1
        if is_valid_addition(S, y):
            S.add(y)
            
    # Local Search Polish
    def local_search(S, max_iterations=100):
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            # Try additions
            for y in range(1, n + 1):
                if y in S: continue
                if is_valid_addition(S, y):
                    S.add(y)
                    improved = True
            # Try swaps
            missing = [y for y in range(1, n + 1) if y not in S]
            S_sorted = sorted(S, reverse=True)
            for y in missing:
                for x in S_sorted:
                    if x <= y: break
                    # Check swap
                    S_test = set(S)
                    S_test.remove(x)
                    S_test.add(y)
                    # Check validity of S_test
                    # Only need to check conflicts involving y or x?
                    # Removing x might break validity? No, removing helps.
                    # Adding y might create 3APs.
                    # Check if y creates 3AP with existing (S - {x})
                    valid = True
                    for a, b in conflict_pairs[y]:
                        if a in S_test and b in S_test:
                            valid = False
                            break
                    if valid:
                        S = S_test
                        improved = True
                        S_sorted = sorted(S, reverse=True)
                        break
        return S
        
    S = local_search(S)
    
    # Check time
    if time.time() > deadline - 5:
        return sorted(S)
        
    # Multiple restarts with perturbation of LP solution?
    # Or just return S
    
    return sorted(S)

def _fallback_greedy(n, deadline, start_time):
    # Original greedy logic
    conflict_pairs = [[] for _ in range(n + 1)]
    for y in range(1, n + 1):
        d = 1
        while y + 2 * d <= n:
            conflict_pairs[y].append((y + d, y + 2 * d))
            d += 1
        d = 1
        while y - d >= 1:
            conflict_pairs[y].append((y - d, y + d))
            d += 1
        d = 1
        while y - 2 * d >= 1:
            conflict_pairs[y].append((y - 2 * d, y - d))
            d += 1
            
    def is_valid_addition(S, y):
        for a, b in conflict_pairs[y]:
            if a in S and b in S:
                return False
        return True
        
    S = set()
    for x in range(1, n + 1):
        if is_valid_addition(S, x):
            S.add(x)
    return sorted(S)
