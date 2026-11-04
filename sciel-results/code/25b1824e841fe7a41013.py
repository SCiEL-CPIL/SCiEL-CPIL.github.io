"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 140.0  # Leave 10s safety margin for 150s limit

    # 1. Initial Solution: Spectral-like Rank Matching
    # Heavy flows (high row sums in a) should go to central locations (low row sums in b)
    # This provides a decent starting point for the iterative LAP relaxation.
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    heavy_idx = np.argsort(-a_sums, kind="stable")
    central_idx = np.argsort(b_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_idx] = central_idx
    
    # Calculate initial cost
    # Cost = sum a[i,j] * b[p[i], p[j]]
    # We can compute this efficiently using matrix operations or loops.
    # For n=150, a loop is fast enough, but let's try to be smarter if needed.
    # Actually, sum_{i,j} a[i,j] b[p[i], p[j]] = trace(A * B_p) where B_p is permuted B.
    # But simpler: for each i, sum_j a[i,j] b[p[i], p[j]]
    def calculate_cost(a, b, p):
        # p[i] is the location assigned to facility i
        # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # This can be computed as sum_i sum_j a[i,j] b[p[i], p[j]]
        # Let B_p[i, j] = b[p[i], p[j]]
        # Then cost = sum_{i,j} a[i,j] B_p[i,j] = np.sum(a * B_p)
        # Constructing B_p is O(n^2).
        B_p = b[np.ix_(p, p)]
        return np.sum(a * B_p)

    cost = calculate_cost(a, b, p)
    best_cost = cost
    best_p = p.copy()

    # 2. Iterative Linear Assignment Relaxation
    # The idea: The QAP objective is sum_{i,j} a_{ij} b_{\pi(i)\pi(j)}.
    # We can view this as minimizing over permutation pi.
    # A common relaxation is to fix the "effective" distance matrix for the LAP.
    # For a given assignment p, the contribution of assigning facility i to location k
    # depends on its interactions with all other facilities j.
    # The "marginal cost" of putting facility i at location k, given that facility j is at p[j],
    # is sum_j a_{ij} b_{k, p[j]}.
    # If we assume the other assignments p[j] (for j != i) are fixed, we can solve for i.
    # However, we want to update all at once or in blocks.
    # 
    # Strategy: 
    # 1. Compute an effective cost matrix C for the LAP.
    #    C[i, k] = sum_j a[i, j] * b[k, p[j]]
    #    This represents the cost of placing facility i at location k, assuming the rest
    #    of the facilities remain at their current locations p[j].
    #    Note: This ignores the interaction between i and the facility currently at k (if any)
    #    and the change in interactions for others, but it's a standard first-order approximation.
    #    Actually, a better approximation for the "linearization" around p is:
    #    C[i, k] = sum_{j != i} a[i, j] * b[k, p[j]] + a[i, i] * b[k, k] (if self-loop exists, but QAP usually i!=j or symmetric)
    #    Standard QAP assumes a[i,i]=0 or doesn't matter. Let's stick to sum_j.
    #
    # 2. Solve LAP: min sum_{i,k} C[i,k] * x[i,k] subject to x being a permutation matrix.
    #    This gives a new permutation p_new.
    #
    # 3. Check if p_new is better than p. If so, accept.
    #    Often, p_new will be very close to p. We can then apply local search (2-opt) to p_new.
    #
    # To make this more effective, we can iterate this relaxation multiple times.
    # However, simply solving LAP with C derived from p might just return p (if p is locally optimal w.r.t single moves).
    # Let's add a bit of "noise" or use a different initialization for the LAP cost matrix?
    # Or, more simply, use this LAP step to escape shallow local minima by occasionally perturbing the cost matrix.
    #
    # Alternative: Use the LAP solution as a candidate, then run 2-opt on it.
    # Since 2-opt is the current strong point, we can use LAP to generate diverse starting points for 2-opt.

    # Let's implement a hybrid:
    # Phase 1: Quick 2-opt on the initial rank-matching solution.
    # Phase 2: Iteratively:
    #   a. Compute effective cost matrix C from current p.
    #   b. Solve LAP to get p_lap.
    #   c. If p_lap is better, set p = p_lap.
    #   d. Run 2-opt on p.
    #   e. Repeat until time limit.

    def local_search_2opt(a, b, p, cost, time_limit, start_time):
        n = len(p)
        # Precompute row sums of A? No, delta calculation needs specific rows.
        # delta for swapping i, j:
        # Let u = p[i], v = p[j].
        # New cost - Old cost = sum_k (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]]) + 2*a[i,j]*b[u,v]
        # Note: The term for k=i and k=j is handled by the +2*a[i,j]*b[u,v] correction if we sum over all k.
        # Actually, the formula sum_k (a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]]) includes k=i and k=j.
        # At k=i: (a[i,i]-a[j,i])*(b[v,p[i]]-b[u,p[i]]) = (0-a[j,i])*(b[v,u]-b[u,u]) = -a[j,i]*b[v,u] (assuming a[i,i]=0, b[u,u]=0)
        # At k=j: (a[i,j]-a[j,j])*(b[v,p[j]]-b[u,p[j]]) = (a[i,j]-0)*(b[v,v]-b[u,v]) = a[i,j]*(-b[u,v]) = -a[i,j]*b[u,v]
        # The cross terms in the original cost:
        # Old: a[i,j]b[u,v] + a[j,i]b[v,u] = 2 a[i,j] b[u,v] (assuming symmetry)
        # New: a[i,j]b[v,v] + a[j,i]b[u,u] = 0 (assuming zero diagonals)
        # Wait, the formula sum_k ... + 2*a[i,j]*b[u,v] is a known efficient delta calculation.
        # Let's verify:
        # Delta = sum_k a[i,k](b[v,p[k]]-b[u,p[k]]) - sum_k a[j,k](b[v,p[k]]-b[u,p[k]]) + 2 a[i,j] b[u,v]
        # This is correct for symmetric QAP with zero diagonals.
        
        improved = True
        while improved and (time.time() - start_time < time_limit - 5):
            improved = False
            for i in range(n):
                if time.time() - start_time > time_limit - 5:
                    return p, cost
                row_i = a[i]
                for j in range(i + 1, n):
                    u = p[i]
                    v = p[j]
                    # If u == v, skip (should not happen in permutation)
                    
                    # Compute delta
                    # diff_a = a[i] - a[j]
                    # diff_b = b[v, p] - b[u, p]
                    # delta = dot(diff_a, diff_b) + 2 * a[i,j] * b[u,v]
                    
                    # Optimization: Check if swap is likely to help?
                    # For now, compute directly.
                    diff_a = row_i - a[j]
                    # b[v, p] is a vector of b[v, p[0]], ..., b[v, p[n-1]]
                    # b[u, p] is a vector of b[u, p[0]], ..., b[u, p[n-1]]
                    diff_b = b[v, p] - b[u, p]
                    
                    delta = np.dot(diff_a, diff_b)
                    delta += 2.0 * a[i, j] * b[u, v]
                    
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break # Restart scan
        return p, cost

    # Run initial local search
    p, cost = local_search_2opt(a, b, p, cost, time_limit, start_time)
    best_cost = cost
    best_p = p.copy()

    # Iterative LAP Relaxation
    iterations = 0
    while time.time() - start_time < time_limit - 10:
        iterations += 1
        
        # Compute effective cost matrix C
        # C[i, k] = sum_j a[i, j] * b[k, p[j]]
        # This is A @ B[:, p] ... wait.
        # C[i, k] = sum_j a[i, j] b[k, p[j]]
        # Let M be matrix where M[j, l] = b[l, p[j]]? No.
        # Let B_p be the matrix where B_p[i, j] = b[p[i], p[j]].
        # Then C[i, k] = sum_j a[i, j] b[k, p[j]]
        # Note that b[k, p[j]] is the entry in row k, column p[j] of B.
        # So C = A @ (B[:, p])^T?
        # Let's check: (A @ X)[i, k] = sum_j A[i, j] X[j, k].
        # We want sum_j A[i, j] B[k, p[j]].
        # So we need X[j, k] = B[k, p[j]].
        # X is the matrix B with columns permuted by p?
        # B[:, p] has columns B[:, p[0]], B[:, p[1]], ...
        # So (B[:, p])[j, k] = B[j, p[k]]? No.
        # numpy indexing: B[:, p] creates a matrix where column k is B[:, p[k]].
        # So (B[:, p])[j, k] = B[j, p[k]].
        # We want X[j, k] = B[k, p[j]].
        # This is (B[:, p]).T ?
        # (B[:, p]).T [j, k] = (B[:, p])[k, j] = B[k, p[j]].
        # Yes! So X = (B[:, p]).T
        # Thus C = A @ (B[:, p]).T
        
        C = a @ (b[:, p]).T
        
        # Solve LAP
        row_ind, col_ind = linear_sum_assignment(C)
        p_lap = np.empty(n, dtype=np.int64)
        p_lap[row_ind] = col_ind
        
        # Check cost of p_lap
        cost_lap = calculate_cost(a, b, p_lap)
        
        # Accept if better
        if cost_lap < cost:
            p = p_lap
            cost = cost_lap
            
            # Run local search on the new solution
            p, cost = local_search_2opt(a, b, p, cost, time_limit, start_time)
            
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        else:
            # If LAP didn't improve, maybe try a random restart or perturbation?
            # To avoid getting stuck, we can try a few random 2-opt swaps or a random permutation if stuck for too long.
            # For now, let's just continue to the next iteration, but this might loop on the same p.
            # To break symmetry, we can add small random noise to C before solving LAP.
            if iterations % 5 == 0:
                # Add small noise to C to explore different permutations
                noise = np.random.normal(0, 1, C.shape) * 0.1 * np.max(np.abs(C))
                C_noisy = C + noise
                row_ind, col_ind = linear_sum_assignment(C_noisy)
                p_lap_noisy = np.empty(n, dtype=np.int64)
                p_lap_noisy[row_ind] = col_ind
                cost_lap_noisy = calculate_cost(a, b, p_lap_noisy)
                if cost_lap_noisy < cost:
                    p = p_lap_noisy
                    cost = cost_lap_noisy
                    p, cost = local_search_2opt(a, b, p, cost, time_limit, start_time)
                    if cost < best_cost:
                        best_cost = cost
                        best_p = p.copy()

    return best_p.tolist()

# EVOLVE-BLOCK-END
