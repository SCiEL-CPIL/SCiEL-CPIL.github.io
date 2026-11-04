"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def total_cost(a, b, p):
    """Calculate the total cost of the assignment."""
    # Cost = sum a[i, j] * b[p[i], p[j]]
    # b_p = b[p, :]
    # cost = np.sum(a * b[p, :][:, p])
    # Note: b[p, :] gives rows of b indexed by p.
    # We need b[p[i], p[j]].
    # Let B_p = b[p, :]. Then B_p[i, j] = b[p[i], j].
    # We want B_p[i, p[j]].
    # So B_p[:, p] gives the matrix where element (i, j) is B_p[i, p[j]] = b[p[i], p[j]].
    return np.sum(a * b[p, :][:, p])

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110  # 10s margin

    # 1. Initialization
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    best_p = p.copy()
    best_cost = total_cost(a, b, best_p)
    
    # Precompute some static parts if needed? 
    # For now, compute on the fly.

    def vectorized_steepest_descent(p, cost, max_iters=1000, deadline=None):
        """
        Perform steepest descent local search using vectorized delta calculation.
        Returns updated p and cost.
        """
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if deadline and time.time() > deadline:
                break
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            
            # We will iterate over i and find the best j for each i
            # To be safe, we just find the global best swap in one pass over i
            
            for i in range(n):
                x = p[i]
                # Vectors needed
                # V_x = b[x, p]
                V_x = b[x, p]
                
                # S2 = sum a[i, :] * V_x
                S2 = np.dot(a[i, :], V_x)
                
                # D1 = b[p, :] @ a[i, :]
                # b[p, :] is n x n. a[i, :] is n.
                D1 = b[p, :] @ a[i, :]
                
                # M = a * b[p, :]
                # D3 = sum M along axis 1
                # We can compute this as (a * b[p, :]).sum(axis=1)
                D3 = (a * b[p, :]).sum(axis=1)
                
                # D4 = a @ V_x
                D4 = a @ V_x
                
                S = D1 - S2 - D3 + D4
                
                # Corrections
                # term_i_vec = (a[i, i] - a[:, i]) * (b[p[j], x] - b[x, x])
                # b[p[j], x] is column x of b[p, :]. i.e., b[p, :][:, x]
                B_px = b[p, :][:, x]
                b_xx = b[x, x]
                term_i_vec = (a[i, i] - a[:, i]) * (B_px - b_xx)
                
                # term_j_vec = (a[i, :] - diag(a)) * (b[p[j], p[j]] - b[x, p[j]])
                # b[p[j], p[j]] is diag(b[p, :])
                B_pp_diag = np.diag(b[p, :])
                # b[x, p[j]] is V_x
                term_j_vec = (a[i, :] - np.diag(a)) * (B_pp_diag - V_x)
                
                # const_vec = (a[i, i] - diag(a)) * (b[p[j], p[j]] - b[x, x]) + (a[i, :] - a.T[i, :]) * (b[p[j], x] - b[x, p[j]])
                # a.T[i, :] is column i of a
                const_vec = (a[i, i] - np.diag(a)) * (B_pp_diag - b_xx) + (a[i, :] - a.T[i, :]) * (B_px - V_x)
                
                Delta_vec = S - term_i_vec - term_j_vec + const_vec
                
                # Set Delta for j=i to 0 (or large positive) to ignore
                Delta_vec[i] = 0
                
                # Find best j for this i
                min_idx = np.argmin(Delta_vec)
                min_val = Delta_vec[min_idx]
                
                if min_val < best_delta:
                    best_delta = min_val
                    best_i = i
                    best_j = min_idx
            
            if best_i != -1 and best_delta < 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                cost += best_delta
                improved = True
                iters += 1
                
        return p, cost

    # Run Steepest Descent
    p, cost = vectorized_steepest_descent(p, best_cost, max_iters=500, deadline=deadline)
    if cost < best_cost:
        best_p = p.copy()
        best_cost = cost

    # If time remains, try Simulated Annealing to escape local minima
    if time.time() < deadline - 10:
        T = 100.0
        T_min = 0.1
        cooling = 0.9999
        current_p = best_p.copy()
        current_cost = best_cost
        
        while T > T_min and time.time() < deadline - 10:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            # Calculate delta
            x = current_p[i]
            y = current_p[j]
            # Using the O(n) formula for single swap
            # Delta = sum_{k!=i,j} (a_ik - a_jk)(b_y,pk - b_x,pk) + const
            # We can just use total_cost difference for simplicity if n is small,
            # but n=100, so O(n) is better.
            
            # Re-implement fast delta for single swap
            # Sum part
            # (a[i] - a[j]) * (b[y, current_p] - b[x, current_p])
            sum_part = np.dot(a[i] - a[j], b[y, current_p] - b[x, current_p])
            
            # Subtract k=i and k=j terms from sum_part
            # term_i = (a[i,i] - a[j,i]) * (b[y, x] - b[x, x])
            term_i = (a[i, i] - a[j, i]) * (b[y, x] - b[x, x])
            # term_j = (a[i,j] - a[j,j]) * (b[y, y] - b[x, y])
            term_j = (a[i, j] - a[j, j]) * (b[y, y] - b[x, y])
            
            # Constant part
            const = (a[i, i] - a[j, j]) * (b[y, y] - b[x, x]) + (a[i, j] - a[j, i]) * (b[y, x] - b[x, y])
            
            delta = sum_part - term_i - term_j + const
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_p[i], current_p[j] = current_p[j], current_p[i]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_p = current_p.copy()
                    best_cost = current_cost
                    
            T *= cooling
            
        # Polish with Steepest Descent again
        p, cost = vectorized_steepest_descent(best_p, best_cost, max_iters=200, deadline=deadline)
        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost

    return best_p.tolist()

# EVOLVE-BLOCK-END
