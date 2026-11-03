"""Global Reassignment Meta-Heuristic using Hungarian Algorithm with Dynamic Weights."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _greedy_init(a, b):
    """
    Initialize using a greedy rank-based heuristic.
    Facilities with higher flow sums are matched to locations with lower distance sums.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities in descending order of flow sum
    # Sort locations in ascending order of distance sum (central locations)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _solve_tai256c(a, b, time_limit=115.0):
    n = len(a)
    start_time = time.time()
    
    # Precompute symmetric parts for efficiency if needed, though linear_sum_assignment
    # takes the cost matrix directly.
    # For QAP cost sum_{i,j} a[i,j] b[p[i], p[j]], 
    # if a and b are symmetric, the cost is 0.5 * sum_{i,j} (a[i,j]+a[j,i]) (b[p[i],p[j]]+b[p[j],p[i]])?
    # No, standard QAP is sum_{i,j} a[i,j] b[p[i], p[j]].
    # If a is symmetric, a[i,j]=a[j,i].
    # The cost is sum_i sum_j a[i,j] b[p[i], p[j]].
    
    # Initial solution
    p = _greedy_init(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Precompute a_sym = a + a.T? 
    # Actually, for the linear assignment subproblem, we need to construct a cost matrix C.
    # C[i, l] = marginal cost of assigning facility i to location l, given current assignment of others.
    # Total Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
    # = sum_i sum_{j != i} a[i,j] b[p[i], p[j]] + sum_i a[i,i] b[p[i], p[i]]
    
    # Let's define the marginal cost of facility i at location l:
    # If we fix all other assignments p[j] for j != i, the terms involving i are:
    # sum_{j != i} a[i,j] b[l, p[j]] + sum_{j != i} a[j,i] b[p[j], l] + a[i,i] b[l,l]
    # Since a is symmetric: 2 * sum_{j != i} a[i,j] b[l, p[j]] + a[i,i] b[l,l]
    # Note: sum_{j != i} a[i,j] b[l, p[j]] is the dot product of a[i, :] (excluding i) and b[l, p[~i]].
    # We can compute this for all i and l efficiently using matrix multiplication?
    # Let B_p be the matrix b[np.ix_(p, p)]? No.
    # Let's define M[i, l] = sum_{j} a[i,j] b[l, p[j]].
    # Then M = a @ b[:, p]? No.
    # b[l, p[j]] is element (l, j) of the matrix b[:, p]. Let B_perm = b[:, p].
    # Then M[i, l] = sum_j a[i,j] B_perm[j, l] = (a @ B_perm)[i, l].
    # So M = a @ b[:, p].
    
    # The marginal cost for facility i at location l, assuming others fixed, is:
    # Cost_i(l) = 2 * (M[i, l] - a[i,i] b[l, p[i]]) + a[i,i] b[l,l] ?
    # Wait. The term sum_{j != i} a[i,j] b[l, p[j]] is M[i, l] - a[i,i] b[l, p[i]].
    # So 2 * (M[i, l] - a[i,i] b[l, p[i]]) + a[i,i] b[l,l].
    # Note that the current cost is sum_i (sum_{j != i} a[i,j] b[p[i], p[j]] + a[i,i] b[p[i], p[i]])
    # = sum_i (M[i, p[i]] - a[i,i] b[p[i], p[i]] + a[i,i] b[p[i], p[i]])
    # = sum_i M[i, p[i]].
    # So the total cost is sum_i M[i, p[i]].
    
    # In each iteration, we construct the cost matrix C where C[i, l] is the marginal cost.
    # C[i, l] = 2 * M[i, l] - 2 * a[i,i] b[l, p[i]] + a[i,i] b[l,l]
    # However, the term -2 * a[i,i] b[l, p[i]] depends on the current assignment p[i].
    # When we reassign ALL facilities, this dependency is circular.
    # Standard approach: Use the current p to define the "background" cost, 
    # then solve for the new assignment minimizing the sum of these marginal costs.
    # This is a valid local search move (specifically, a 2-opt or global reassignment step).
    
    # To improve escape from local minima, we can add noise or use a temperature-based acceptance.
    # But linear_sum_assignment finds the global min of the constructed matrix.
    # If we just do this, it's a deterministic local search. 
    # To make it stochastic, we can perturb the cost matrix or restart with different initializations.
    
    # Strategy: 
    # 1. Compute M = a @ b[:, p]
    # 2. Construct C[i, l] = 2 * M[i, l] + a[i,i] b[l,l] - 2 * a[i,i] b[l, p[i]]
    # 3. Solve linear assignment on C.
    # 4. If new cost is better, accept.
    # 5. To diversify, occasionally randomize the initial p or add random noise to C.
    
    # Let's implement a loop of these global reassignment steps.
    # Each step is O(n^3) due to Hungarian. n=256. 256^3 ~ 16M ops.
    # scipy.optimize.linear_sum_assignment is very fast in C. 
    # Let's estimate: 16M ops is ~10ms? 
    # We can do ~1000 iterations in 10s.
    # 1000 iterations in 115s is plenty.
    
    # To avoid getting stuck in a single local optimum, we will:
    # - Keep the best solution found.
    # - If no improvement for K iterations, restart from a perturbed version of the best solution or a new random solution.
    
    no_improve_count = 0
    max_no_improve = 50
    rng = np.random.default_rng(42)
    
    # Precompute diagonal of a and b for faster marginal cost calc?
    # a_diag = np.diag(a)
    # b_diag = np.diag(b)
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5.0:  # Safety margin
            break
            
        # Compute M = a @ b[:, p]
        # b[:, p] is a fancy indexing operation.
        B_perm = b[:, p]
        M = a @ B_perm
        
        # Construct Cost Matrix C
        # C[i, l] = 2 * M[i, l] + a[i,i] * b[l,l] - 2 * a[i,i] * b[l, p[i]]
        # Vectorized:
        # term1 = 2 * M
        # term2 = np.outer(np.diag(a), np.diag(b))  # a[i,i] * b[l,l]
        # term3 = 2 * np.outer(np.diag(a), b[np.diag_indices_from(b)][:, p])? No.
        # b[l, p[i]] for all i, l.
        # Let b_diag_at_p = b[np.arange(n)[:, None], p[None, :]]? No, p is per i.
        # We need a matrix D where D[i, l] = b[l, p[i]].
        # This is just b[:, p] again? No.
        # b[:, p] has columns p. Column i of b[:, p] is b[:, p[i]].
        # So D = b[:, p].
        # Wait, M = a @ b[:, p].
        # The term we want to subtract is 2 * a[i,i] * b[l, p[i]].
        # Let A_diag = np.diag(a)
        # Let B_p = b[:, p]
        # Subtraction term S[i, l] = 2 * A_diag[i] * B_p[l, i]
        # S = 2 * np.outer(A_diag, B_p.T) ?
        # B_p.T[i, l] = B_p[l, i] = b[l, p[i]]. Yes.
        
        A_diag = np.diag(a)
        B_diag = np.diag(b)
        
        C = 2.0 * M + np.outer(A_diag, B_diag) - 2.0 * np.outer(A_diag, B_perm.T)
        
        # Solve Linear Assignment
        row_ind, col_ind = linear_sum_assignment(C)
        
        # new_p is such that new_p[row_ind] = col_ind?
        # linear_sum_assignment returns (u, v) such that cost[u, v] is min.
        # u are rows (facilities), v are cols (locations).
        # So facility u[k] goes to location v[k].
        # We need a permutation array p_new where p_new[i] = location of facility i.
        # So p_new[u] = v.
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        
        # Calculate cost of new solution
        new_cost = _qap_cost(a, b, p_new)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_p = p_new.copy()
            p = p_new.copy() # Update current for next iteration
            no_improve_count = 0
        else:
            no_improve_count += 1
            # If stuck, perturb the current solution to escape local minimum
            if no_improve_count % 10 == 0:
                # Perform a few random swaps on p to diversify
                for _ in range(5):
                    i, j = rng.choice(n, size=2, replace=False)
                    p[i], p[j] = p[j], p[i]
                # Recalculate cost for p
                current_cost = _qap_cost(a, b, p)
                # If this perturbed p is better than best, update best? 
                # Usually perturbation is worse, but might lead to better later.
                # We keep p as the working state.
            
            # If stuck for too long, restart from a completely random or greedy init
            if no_improve_count > max_no_improve:
                # Restart strategy:
                # 1. Random init
                # 2. Greedy init
                # Let's try a random init
                p = rng.permutation(n)
                current_cost = _qap_cost(a, b, p)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                no_improve_count = 0

    return best_p.tolist()

def solve(a, b):
    return _solve_tai256c(a, b)
# EVOLVE-BLOCK-END
