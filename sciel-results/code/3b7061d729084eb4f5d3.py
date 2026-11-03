"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def solve(a, b):
    n = len(a)
    start_time = time.time()
    # Safety margin: finish before 120s, aim for 110s
    time_limit = 110.0
    
    # 1. Initialization: Rank Matching
    # Facilities with high flow sums go to locations with low distance sums (central)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    current_cost = np.sum(a * b[np.ix_(p, p)])
    
    stall_count = 0
    max_stall = 10
    
    # 2. Iterative Benders-like Linearization Loop
    while time.time() - start_time < time_limit:
        # Master Problem: Linearized Cost
        # Approximate the quadratic cost by fixing the location of j's in the distance term
        # C[i, l] = sum_j a[i, j] * b[l, p[j]]
        # This is computed as matrix multiplication: a @ b[:, p]
        C = a @ b[:, p]
        
        # Solve the Linear Assignment Problem
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = col_ind
        
        # Subproblem: Evaluate Exact Quadratic Cost
        new_cost = np.sum(a * b[np.ix_(p_new, p_new)])
        
        if new_cost < current_cost:
            current_cost = new_cost
            p = p_new
            stall_count = 0
        else:
            stall_count += 1
            if stall_count >= max_stall:
                # Diversification: Perturb the current solution
                # Swap a few random pairs to escape local minima
                num_swaps = min(5, n // 2)
                for _ in range(num_swaps):
                    i = np.random.randint(0, n)
                    j = np.random.randint(0, n)
                    if i != j:
                        p[i], p[j] = p[j], p[i]
                # Recalculate cost after perturbation
                current_cost = np.sum(a * b[np.ix_(p, p)])
                stall_count = 0
                max_stall = 10 # Keep threshold same or increase slightly?
        
    # 3. Final Polish: Limited 2-Opt if time remains
    # This helps refine the solution found by the linearization loop
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5.0:
        p, current_cost = local_search_2opt(a, b, p, current_cost, time.time(), start_time, time_limit)
        
    return p.tolist()


def local_search_2opt(a, b, p, current_cost, now, start_time, time_limit):
    n = len(p)
    # Perform a few passes of 2-opt, checking time frequently
    # We limit the number of full passes to avoid getting stuck in slow loops
    max_passes = 10 
    for _ in range(max_passes):
        if time.time() - start_time > time_limit - 2.0:
            break
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit - 2.0:
                return p, current_cost
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit - 2.0:
                    return p, current_cost
                # Swap p[i] and p[j]
                p[i], p[j] = p[j], p[i]
                new_cost = np.sum(a * b[np.ix_(p, p)])
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    p[i], p[j] = p[j], p[i]
        if not improved:
            break
    return p, current_cost
# EVOLVE-BLOCK-END
