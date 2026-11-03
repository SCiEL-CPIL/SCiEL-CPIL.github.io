# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute row sums for initial heuristic
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    # Initial rank-matching: heavy flows to central locations
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()

    def compute_cost(p):
        """Compute exact QAP cost."""
        return int(np.sum(a * b[np.ix_(p, p)]))

    def two_opt_delta(p, i, j):
        """
        Compute delta cost for swapping facilities i and j in assignment p.
        Assumes b is symmetric.
        """
        li = p[i]
        lj = p[j]
        
        d = b[lj, p] - b[li, p]
        d[i] = 0
        d[j] = 0
        
        a_diff = a[i, :] - a[j, :] + a[:, i] - a[:, j]
        a_diff[i] = 0
        a_diff[j] = 0
        
        return int(np.dot(a_diff, d))

    def hungarian_reassign(p):
        """
        Use Hungarian algorithm to globally reassign facilities based on linearized costs.
        """
        S = a + a.T
        bp = b[:, p]  # shape (n, n), bp[l, j] = b[l, p[j]]
        C = S @ bp.T
        
        row_ind, col_ind = linear_sum_assignment(C)
        
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        
        return new_p

    def perturb(p, k=10):
        """
        Apply k random swaps to escape local minima.
        """
        p = p.copy()
        for _ in range(k):
            i, j = random.sample(range(n), 2)
            p[i], p[j] = p[j], p[i]
        return p

    def two_opt_polish(p, time_limit):
        """
        Run 2-opt until no improvement or time limit reached.
        Returns improved permutation and cost.
        """
        cost = compute_cost(p)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n):
                if time.time() > time_limit:
                    break
                for j in range(i+1, n):
                    delta = two_opt_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    pass
        return p, cost

    best_p = p.copy()
    best_cost = compute_cost(best_p)
    
    # Phase 1: Initial 2-opt local search
    p, best_cost = two_opt_polish(p, deadline - 40.0)
    best_p = p.copy()
    
    # Phase 2: Iterative Hungarian reassignment with perturbation
    for iteration in range(20):
        if time.time() > deadline - 20.0:
            break
        
        # Hungarian reassign
        new_p = hungarian_reassign(p)
        new_cost = compute_cost(new_p)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_p = new_p.copy()
            p = new_p.copy()
        else:
            # If Hungarian didn't improve, try perturbation
            perturbed_p = perturb(p, k=5 + iteration)
            perturbed_cost = compute_cost(perturbed_p)
            
            if perturbed_cost < best_cost:
                best_cost = perturbed_cost
                best_p = perturbed_p.copy()
                p = perturbed_p.copy()
            else:
                # Try a few more perturbations
                found_improvement = False
                for attempt in range(3):
                    if time.time() > deadline - 20.0:
                        break
                    perturbed_p = perturb(p, k=3 + attempt * 2)
                    perturbed_cost = compute_cost(perturbed_p)
                    if perturbed_cost < best_cost:
                        best_cost = perturbed_cost
                        best_p = perturbed_p.copy()
                        p = perturbed_p.copy()
                        found_improvement = True
                        break
                if not found_improvement:
                    break
        
        # 2-opt polish after Hungarian or perturbation
        if time.time() > deadline - 15.0:
            break
        
        p, polish_cost = two_opt_polish(p, deadline - 10.0)
        if polish_cost < best_cost:
            best_cost = polish_cost
            best_p = p.copy()
    
    # Phase 3: Final high-precision polish pass
    if time.time() < deadline - 5.0:
        # Use best_p as starting point for final polish
        final_p, final_cost = two_opt_polish(best_p.copy(), deadline - 2.0)
        if final_cost < best_cost:
            best_cost = final_cost
            best_p = final_p.copy()
    
    # Final check: ensure best_p is the best found
    final_cost = compute_cost(best_p)
    if final_cost < best_cost:
        best_cost = final_cost
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
