"""Rewritten QAP solver: continuous relaxation + Hungarian + local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Build a cost matrix from row sums (initial heuristic).
    2. Use Hungarian to get initial permutation.
    3. Local search with 2-swap and 3-swap moves under a time budget.
    """
    n = len(a)
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # --- Initial solution via rank-matching + Hungarian refinement ---
    # Build an initial assignment matrix cost for Hungarian:
    # cost[i, j] approximates the marginal cost of placing facility i at location j
    # Use: cost[i,j] = sum_k a[i,k] * b[j, pi0[k]] + sum_k a[k,i] * b[pi0[k], j]
    # where pi0 is a simple rank-matching permutation.
    
    # Simple rank-matching initial permutation
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)
    
    # Compute initial cost
    def compute_cost(p):
        # sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Use vectorized: for each i, sum_j a[i,j]*b[p[i],p[j]]
        # This is: row-wise: a[i,:] dot b[p[i], p[:]]
        # = sum_i (a[i,:] @ b[p[i], p[:]])
        # = sum_i np.dot(a[i], b[p[i], p])
        # Efficient: B_p = b[np.ix_(p, p)]; then cost = np.sum(a * B_p) but that's O(n^2) memory
        # Better: for each i, np.dot(a[i], b[p[i], p]) -- O(n^2) total
        total = 0
        for i in range(n):
            total += np.dot(a[i], b[p[i], p])
        return total
    
    best_p = p.copy()
    best_cost = compute_cost(best_p)
    
    # --- Local search: 2-swap moves ---
    # Precompute row sums of b for delta computation
    # delta of swapping p[i] and p[j]:
    # Let u = p[i], v = p[j]
    # Old contribution involving i: sum_k a[i,k]*b[u, p[k]]
    # Old contribution involving j: sum_k a[j,k]*b[v, p[k]]
    # New: i gets v, j gets u
    # New contribution involving i: sum_k a[i,k]*b[v, p'[k]] where p'[i]=v, p'[j]=u, else same
    #   = a[i,i]*b[v,v] + a[i,j]*b[v,u] + sum_{k!=i,j} a[i,k]*b[v,p[k]]
    # Old contribution involving i: a[i,i]*b[u,u] + a[i,j]*b[u,v] + sum_{k!=i,j} a[i,k]*b[u,p[k]]
    # Similarly for j.
    # Also contributions where other k interacts with i and j change.
    # This gets complex. Let's use a simpler delta formula.
    
    # Actually, the full cost is sum_{i,j} a[i,j] b[p[i], p[j]].
    # Swapping positions i and j (i.e., p[i]<->p[j]) changes:
    # For all (i',j') where p[i'] or p[j'] is u or v, the b values change.
    # Specifically:
    # Old: b[u, p[k]] for all k, and b[p[k], u] for all k, and b[v, p[k]], b[p[k], v]
    # After swap: u and v swap places in the permutation.
    
    # Let's compute delta more carefully:
    # Let S = set of all (i',j') such that p[i'] in {u,v} or p[j'] in {u,v}
    # The change in cost = sum over affected terms.
    
    # A cleaner approach: 
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Define M = b indexed by p: M[i,j] = b[p[i], p[j]]
    # cost = sum a[i,j] * M[i,j]
    # Swapping p[i] and p[j] means rows i,j of M swap their row indices in b, and columns i,j swap.
    # So M_new[i,:] = M_old[j,:] with column swap, etc.
    
    # Let me just compute the delta directly:
    # Let u = p[i], v = p[j]
    # After swap: p'[i]=v, p'[j]=u, p'[k]=p[k] for k!=i,j
    # 
    # Delta = sum_{i',j'} a[i',j'] * (b[p'[i'], p'[j']] - b[p[i'], p[j']])
    # Only terms where p'[i'] != p[i'] or p'[j'] != p[j'] contribute.
    # p'[i'] != p[i'] only for i'=i or i'=j.
    # Similarly for j'.
    # So affected (i',j') pairs are those with i' in {i,j} OR j' in {i,j}.
    
    def swap_delta(i, j, p, a, b):
        u = p[i]
        v = p[j]
        if u == v:
            return 0
        # Terms where i' in {i,j} or j' in {i,j}
        # Group:
        # 1. (i', j') where i'=i and j'=i: a[i,i]*(b[v,v] - b[u,u])
        # 2. (i', j') where i'=i and j'=j: a[i,j]*(b[v,u] - b[u,v])
        # 3. (i', j') where i'=j and j'=i: a[j,i]*(b[u,v] - b[v,u])
        # 4. (i', j') where i'=j and j'=j: a[j,j]*(b[u,u] - b[v,v])
        # 5. (i', j') where i'=i and j' not in {i,j}: a[i,j']*(b[v, p[j']] - b[u, p[j']])
        # 6. (i', j') where i'=j and j' not in {i,j}: a[j,j']*(b[u, p[j']] - b[v, p[j']])
        # 7. (i', j') where j'=i and i' not in {i,j}: a[i',i]*(b[p[i'], v] - b[p[i'], u])
        # 8. (i', j') where j'=j and i' not in {i,j}: a[i',j]*(b[p[i'], u] - b[p[i'], v])
        
        delta = 0.0
        # Terms 1-4
        delta += a[i, i] * (b[v, v] - b[u, u])
        delta += a[i, j] * (b[v, u] - b[u, v])
        delta += a[j, i] * (b[u, v] - b[v, u])
        delta += a[j, j] * (b[u, u] - b[v, v])
        
        # Terms 5-8: sum over k not in {i,j}
        # For efficiency, precompute row/col sums
        # Term 5: sum_{k not in {i,j}} a[i,k] * (b[v, p[k]] - b[u, p[k]])
        #         = sum_k a[i,k]*(b[v,p[k]] - b[u,p[k]]) - a[i,i]*(b[v,v]-b[u,u]) - a[i,j]*(b[v,u]-b[u,v])
        # Term 6: sum_{k not in {i,j}} a[j,k] * (b[u, p[k]] - b[v, p[k]])
        #         = sum_k a[j,k]*(b[u,p[k]] - b[v,p[k]]) - a[j,i]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
        # Term 7: sum_{k not in {i,j}} a[k,i] * (b[p[k], v] - b[p[k], u])
        #         = sum_k a[k,i]*(b[p[k],v]-b[p[k],u]) - a[i,i]*(b[v,v]-b[u,u]) - a[j,i]*(b[v,u]-b[u,v])
        # Term 8: sum_{k not in {i,j}} a[k,j] * (b[p[k], u] - b[p[k], v])
        #         = sum_k a[k,j]*(b[p[k],u]-b[p[k],v]) - a[i,j]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
        
        # Precompute for each location l: 
        #   row_diff[i][l] = b[l, p[k]] terms... this is getting complex.
        
        # Let's just compute directly for k not in {i,j} using vectorized ops
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        ks = np.where(mask)[0]
        
        # Term 5
        if len(ks) > 0:
            delta += np.dot(a[i, ks], b[v, p[ks]] - b[u, p[ks]])
        # Term 6
        if len(ks) > 0:
            delta += np.dot(a[j, ks], b[u, p[ks]] - b[v, p[ks]])
        # Term 7
        if len(ks) > 0:
            delta += np.dot(a[ks, i], b[p[ks], v] - b[p[ks], u])
        # Term 8
        if len(ks) > 0:
            delta += np.dot(a[ks, j], b[p[ks], u] - b[p[ks], v])
        
        return delta
    
    # --- Local search loop ---
    # Use simulated annealing style or simple hill climbing with random restarts
    # For time efficiency, do multiple passes of best-improvement 2-swap
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Try all pairs (random subset for large n)
        if n > 100:
            # Random sampling of pairs
            num_trials = 20000
            for _ in range(num_trials):
                if time.time() > deadline - 5.0:
                    break
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                d = swap_delta(i, j, best_p, a, b)
                if d < -1e-9:
                    # Accept
                    best_p[i], best_p[j] = best_p[j], best_p[i]
                    best_cost += d
                    improved = True
        else:
            # Exhaustive
            for i in range(n):
                if time.time() > deadline - 5.0:
                    break
                for j in range(i+1, n):
                    d = swap_delta(i, j, best_p, a, b)
                    if d < -1e-9:
                        best_p[i], best_p[j] = best_p[j], best_p[i]
                        best_cost += d
                        improved = True
    
    # --- Try Hungarian-based refinement ---
    # Build a cost matrix from the current permutation and use Hungarian to find a better one
    # cost[i, j] = sum_k a[i,k] * b[j, p[k]] + sum_k a[k,i] * b[p[k], j]
    # This is a linearization of the QAP objective around the current solution
    
    if time.time() < deadline - 10.0:
        # Build linear cost matrix
        # For each facility i and location j, approximate marginal cost
        # C[i,j] = sum_k a[i,k]*b[j, p[k]] + sum_k a[k,i]*b[p[k], j]
        # = (a @ b.T @ p_vec)[i,j] ... let's compute directly
        
        # p_vec: for each k, location is p[k]
        # b[j, p[k]] for all j,k: this is b[:, p] which is b indexed by p in columns
        # b[:, p] has shape (n, n): row j, col k = b[j, p[k]]
        b_col_p = b[:, best_p]  # shape (n, n), [j, k] = b[j, p[k]]
        b_row_p = b[best_p, :]  # shape (n, n), [k, j] = b[p[k], j]
        
        # C[i,j] = sum_k a[i,k]*b[j, p[k]] + sum_k a[k,i]*b[p[k], j]
        # = (a @ b_col_p.T)[i,j] + (b_row_p @ a.T)[i,j]
        # a @ b_col_p.T: [i,j] = sum_k a[i,k] * b_col_p[j,k] = sum_k a[i,k]*b[j,p[k]] ✓
        # b_row_p @ a.T: [i,j] = sum_k b_row_p[k,i] * a.T[i,j]... wait
        # b_row_p[k,i] = b[p[k], i], a.T[i,j] = a[j,i]
        # So [i,j] = sum_k b[p[k],i] * a[j,i]... that's not right.
        
        # Let me redo: sum_k a[k,i]*b[p[k], j]
        # = sum_k a[k,i] * b_row_p[k, j]  (since b_row_p[k,j] = b[p[k], j])
        # This is (a.T @ b_row_p)[i, j] = sum_k a.T[i,k] * b_row_p[k,j] = sum_k a[k,i]*b[p[k],j] ✓
        
        C = a @ b_col_p.T + a.T @ b_row_p
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(C)
        p_hung = np.empty(n, dtype=np.int64)
        p_hung[row_ind] = col_ind
        
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
    
    # --- Final local search pass ---
    if time.time() < deadline - 3.0:
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            if n > 100:
                num_trials = 10000
                for _ in range(num_trials):
                    if time.time() > deadline - 3.0:
                        break
                    i, j = np.random.randint(0, n, 2)
                    if i == j:
                        continue
                    d = swap_delta(i, j, best_p, a, b)
                    if d < -1e-9:
                        best_p[i], best_p[j] = best_p[j], best_p[i]
                        best_cost += d
                        improved = True
            else:
                for i in range(n):
                    if time.time() > deadline - 3.0:
                        break
                    for j in range(i+1, n):
                        d = swap_delta(i, j, best_p, a, b)
                        if d < -1e-9:
                            best_p[i], best_p[j] = best_p[j], best_p[i]
                            best_cost += d
                            improved = True
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
