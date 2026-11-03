# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

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
        return np.sum(a * b[np.ix_(p, p)])

    def linearized_cost_matrix(p):
        """
        Compute the linearized cost matrix for Hungarian algorithm.
        For each facility i, the marginal cost of assigning it to location l
        is approximately sum_j a[i,j] * b[l, p[j]] + sum_j a[j,i] * b[p[j], l].
        """
        # For each facility i and location l:
        # cost[i,l] = sum_j a[i,j] * b[l, p[j]] + sum_j a[j,i] * b[p[j], l]
        #           = b[l, p] @ a[i, :] + b[p, l] @ a[:, i]
        #           = (b[:, p] @ a[i, :]) + (a[:, i] @ b[p, :])
        
        # Efficient computation:
        # cost[i,l] = sum_j a[i,j] * b[l, p[j]] + sum_j a[j,i] * b[p[j], l]
        
        # Let B = b (n x n)
        # For fixed p, define P = p
        # cost[i,l] = sum_j a[i,j] * b[l, P[j]] + sum_j a[j,i] * b[P[j], l]
        
        # First term: sum_j a[i,j] * b[l, P[j]] = (a[i, :] @ b[l, P])
        # Second term: sum_j a[j,i] * b[P[j], l] = (a[:, i] @ b[P, l])
        
        # We can vectorize:
        # cost[i,l] = a[i,:] @ b[l,p] + a[:,i] @ b[p,l]
        
        # Let's compute:
        # term1[i,l] = a[i,:] @ b[l,p] = sum_j a[i,j] * b[l,p[j]]
        # term2[i,l] = a[:,i] @ b[p,l] = sum_j a[j,i] * b[p[j],l]
        
        # term1 = a @ b[:, p]  (n x n) where (a @ b[:,p])[i,l] = sum_j a[i,j]*b[:,p][j,l]
        # Wait, that's not right. Let me think again.
        
        # b[l, p[j]] for fixed l and varying j: this is b[l, p] (a row of b)
        # So sum_j a[i,j] * b[l, p[j]] = a[i,:] @ b[l,p]
        
        # Similarly, b[p[j], l] for fixed l and varying j: this is b[p, l] (a column of b)
        # So sum_j a[j,i] * b[p[j], l] = a[:,i] @ b[p,l]
        
        # So cost[i,l] = a[i,:] @ b[l,p] + a[:,i] @ b[p,l]
        
        # To compute this efficiently:
        # term1[i,l] = a[i,:] @ b[l,p]
        # This is: for each i, compute a[i,:] dot b[l,p] for all l
        # = (a @ b[:,p].T)[i,l]? No.
        # a[i,:] @ b[l,p] = sum_j a[i,j] * b[l,p[j]]
        # Let C1 = b[:, p] (shape n x n), where C1[l,j] = b[l, p[j]]
        # Then term1[i,l] = sum_j a[i,j] * C1[l,j] = (a @ C1.T)[i,l]
        
        # C2 = b[p, :] (shape n x n), where C2[j,l] = b[p[j], l]
        # term2[i,l] = sum_j a[j,i] * C2[j,l] = (a.T @ C2)[i,l]
        
        C1 = b[:, p]  # C1[l,j] = b[l, p[j]]
        C2 = b[p, :]  # C2[j,l] = b[p[j], l]
        
        term1 = a @ C1.T  # term1[i,l] = sum_j a[i,j] * C1[l,j] = sum_j a[i,j] * b[l,p[j]]
        term2 = a.T @ C2  # term2[i,l] = sum_j a[j,i] * C2[j,l] = sum_j a[j,i] * b[p[j],l]
        
        return term1 + term2

    best_p = p.copy()
    best_cost = compute_cost(best_p)
    
    # Phase 1: 2-opt local search
    def two_opt_delta(p, i, j):
        """Compute delta cost for swapping facilities i and j."""
        li = p[i]
        lj = p[j]
        
        d = b[lj, p] - b[li, p]
        d[i] = 0
        d[j] = 0
        
        a_diff = a[i, :] - a[j, :] + a[:, i] - a[:, j]
        a_diff[i] = 0
        a_diff[j] = 0
        
        return np.dot(a_diff, d)

    # Do some 2-opt iterations first to get a good starting point
    max_2opt_iters = 3
    for _ in range(max_2opt_iters):
        if time.time() > deadline - 60.0:
            break
        improved = True
        while improved and time.time() < deadline - 60.0:
            improved = False
            for i in range(n):
                if time.time() > deadline - 60.0:
                    break
                for j in range(i+1, n):
                    delta = two_opt_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        best_cost += delta
                        best_p = p.copy()
                        improved = True
                        break
                if improved:
                    break

    # Phase 2: Hungarian reassignment iterations
    # Use the linearized cost to find a globally better assignment
    hungarian_iters = 0
    while time.time() < deadline - 10.0:
        # Compute linearized cost matrix
        cost_mat = linearized_cost_matrix(p)
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_mat)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        
        # Compute new cost
        new_cost = compute_cost(new_p)
        
        if new_cost < best_cost:
            p = new_p
            best_p = p.copy()
            best_cost = new_cost
            hungarian_iters += 1
            
            # After a good Hungarian step, do some 2-opt to polish
            if time.time() < deadline - 15.0:
                improved = True
                while improved and time.time() < deadline - 15.0:
                    improved = False
                    for i in range(n):
                        if time.time() > deadline - 15.0:
                            break
                        for j in range(i+1, n):
                            delta = two_opt_delta(p, i, j)
                            if delta < 0:
                                p[i], p[j] = p[j], p[i]
                                best_cost += delta
                                best_p = p.copy()
                                improved = True
                                break
                        if improved:
                            break
        else:
            # No improvement from Hungarian; try a different strategy
            # Add some noise to escape local minima
            if hungarian_iters > 0:
                # Perturb the current solution
                idx = np.random.choice(n, size=min(10, n), replace=False)
                for i in idx:
                    j = np.random.randint(n)
                    delta = two_opt_delta(p, i, j)
                    if delta < 5000:  # Allow small worsening moves
                        p[i], p[j] = p[j], p[i]
                        best_cost = compute_cost(p)
                        if best_cost < best_p is not None:
                            pass
                        # Recompute properly
                        current_cost = compute_cost(p)
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = p.copy()
            else:
                # First iteration, no improvement yet, break to save time
                break
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
