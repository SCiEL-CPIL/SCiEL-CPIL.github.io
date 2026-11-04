"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s kill

    # Precompute symmetric versions for potential use, though we work with raw a,b for exact cost
    # a_sym = (a + a.T) / 2.0
    # b_sym = (b + b.T) / 2.0

    def compute_cost(p):
        # sum_{i,j} a[i,j] * b[p[i], p[j]]
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    # 1. Initial Solution Generation
    # Combine rank-matching (current) and Hungarian on surrogate (inspiration)
    
    # Method A: Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    
    # Method B: Hungarian on linear surrogate
    # Surrogate cost: a[i, :] @ b[:, j]? No, that's not a linear assignment cost.
    # The QAP cost is sum a[i,j] b[p[i], p[j]].
    # A common linear relaxation/surrogate is based on row sums.
    # Let's try: cost[i, j] = a.sum(axis=1)[i] * b.sum(axis=1)[j]
    # This encourages high-flow facilities to go to low-distance locations.
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    surrogate_cost = np.outer(a_sums, b_sums)
    _, p_hung_idx = linear_sum_assignment(surrogate_cost)
    p_hung = p_hung_idx.astype(np.int64)
    
    # Compare and pick the better initial solution
    cost_rank = compute_cost(p_rank)
    cost_hung = compute_cost(p_hung)
    
    if cost_hung < cost_rank:
        best_p = p_hung.copy()
        best_cost = cost_hung
    else:
        best_p = p_rank.copy()
        best_cost = cost_rank

    # 2. Local Search with Efficient Delta Computation
    # We use the O(n) delta swap computation from the current program,
    # but we will use it in a steepest-descent or random-restart hill-climbing fashion.
    # To improve upon the current program's "try all pairs" which is O(n^3) per pass,
    # we can try a randomized local search or a more efficient steepest descent.
    # However, for n=100, O(n^3) = 10^6 operations per pass is very fast (~1ms).
    # So a full pass of steepest descent is cheap.
    
    def compute_delta_swap(p, i, j):
        u = p[i]
        v = p[j]
        
        # Vectorized delta calculation
        # Exclude i and j
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        
        # Term 1: sum_{k!=i,j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
        diff_a_row = a[i, mask] - a[j, mask]
        diff_b_row = b[v, p[mask]] - b[u, p[mask]]
        term1 = np.dot(diff_a_row, diff_b_row)
        
        # Term 2: sum_{k!=i,j} (a[k,i] - a[k,j]) * (b[p[k], v] - b[p[k], u])
        diff_a_col = a[mask, i] - a[mask, j]
        diff_b_col = b[p[mask], v] - b[p[mask], u]
        term2 = np.dot(diff_a_col, diff_b_col)
        
        # Term 3: (a[i,j] - a[j,i]) * (b[v,u] - b[u,v])
        term3 = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
        
        return term1 + term2 + term3

    # 3. Search Loop: Random Restarts with Steepest Descent
    # The current program does 50 restarts. We can increase this or improve the perturbation.
    # The inspiration used Simulated Annealing. SA can escape local minima better.
    # Let's implement a simple SA-like approach or just increase the number of restarts
    # and use a smarter perturbation.
    
    # Given the time budget (115s), and each steepest descent pass taking ~1ms,
    # we can do thousands of passes.
    # Let's do a loop of:
    # 1. Perturb current best
    # 2. Steepest descent until local minimum
    # 3. Update best if improved
    
    # To avoid getting stuck in the same local minima, we use random perturbations.
    
    num_restarts = 200 # Increased from 50, each restart is cheap
    rng = np.random.default_rng(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        if restart == 0:
            p = best_p.copy()
        else:
            # Perturb best_p
            p = best_p.copy()
            # Randomly swap a few pairs. The number of swaps can scale with restart count
            # to explore further away.
            num_swaps = 3 + (restart // 20)
            for _ in range(num_swaps):
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                if i != j:
                    p[i], p[j] = p[j], p[i]

        # Steepest Descent Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            
            # Find the best improving swap
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    delta = compute_delta_swap(p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            
            if best_i != -1:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                improved = True
        
        # Re-evaluate cost
        cur_cost = compute_cost(p)
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_p = p.copy()

    # 4. Final Polish
    # Do a few more passes of steepest descent on the best solution found
    if time.time() < deadline:
        p = best_p.copy()
        # Run a few full passes to ensure it's a local minimum
        for _ in range(5):
            if time.time() > deadline:
                break
            improved = True
            while improved:
                improved = False
                best_delta = 0
                best_i, best_j = -1, -1
                for i in range(n):
                    for j in range(i + 1, n):
                        delta = compute_delta_swap(p, i, j)
                        if delta < best_delta:
                            best_delta = delta
                            best_i, best_j = i, j
                if best_i != -1:
                    p[best_i], p[best_j] = p[best_j], p[best_i]
                    improved = True
        
        cur_cost = compute_cost(p)
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_p = p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
