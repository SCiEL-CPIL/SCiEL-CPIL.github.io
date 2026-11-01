import numpy as np
import time
from itertools import product, combinations

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Seed solution (Rank 24) - schoolbook
    rank_seed = n * m * p
    U_seed = np.zeros((n * m, rank_seed), dtype=np.float64)
    V_seed = np.zeros((m * p, rank_seed), dtype=np.float64)
    W_seed = np.zeros((p * n, rank_seed), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_seed.copy(), V_seed.copy(), W_seed.copy()
    best_rank = rank_seed

    # Constraints: (u_idx, v_idx, w_idx)
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                constraints.append((u_idx, v_idx, w_idx))
                
    num_constraints = len(constraints)

    # Pre-compute indices for vectorization
    u_idxs = np.array([c[0] for c in constraints])
    v_idxs = np.array([c[1] for c in constraints])
    w_idxs = np.array([c[2] for c in constraints])

    def error_function(U, V, W):
        vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
        return np.sum((vals - 1.0) ** 2)

    def check_exact(U, V, W):
        # Verify exactness with integer/half-integer logic
        # Since inputs are multiples of 0.5, products are multiples of 0.125
        # Sums should be exactly 1.0
        vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
        return np.allclose(vals, 1.0, atol=1e-9)

    def normalize_columns(X):
        norms = np.linalg.norm(X, axis=0)
        norms[norms < 1e-12] = 1.0
        return X / norms

    def try_rank_als(r, max_iters=50, num_restarts=100, seed_base=42):
        if time.time() > deadline - 10.0:
            return None
            
        # Precompute structure for ALS
        # A_U[c, u_c * r + k] = V[v_c, k] * W[w_c, k]
        # We will build these on the fly or use sparse if needed, but dense is fine for small r
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                return None
                
            np.random.seed(seed_base + restart)
            
            # Random initialization
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Normalize initial guesses to prevent explosion/collapse
            U = normalize_columns(U)
            V = normalize_columns(V)
            W = normalize_columns(W)
            
            for it in range(max_iters):
                if time.time() > deadline - 5.0:
                    return None
                
                # Build linear systems for U, V, W
                # Dimensions:
                # U: (n*m, r) -> flat (n*m*r)
                # V: (m*p, r) -> flat (m*p*r)
                # W: (p*n, r) -> flat (p*n*r)
                
                A_U = np.zeros((num_constraints, n * m * r))
                A_V = np.zeros((num_constraints, m * p * r))
                A_W = np.zeros((num_constraints, p * n * r))
                
                # Fill matrices
                # Optimization: Vectorize the filling?
                # For c in range(num_constraints):
                #   u, v, w = constraints[c]
                #   for k in range(r):
                #     A_U[c, u*r+k] = V[v,k] * W[w,k]
                
                # Let's do it efficiently using broadcasting if possible, or loop is fine for small sizes
                # n*m=6, m*p=12, p*n=8. r <= 23. num_constraints=24.
                # 24 * 23 ~ 552 ops per matrix. Very fast.
                
                for c in range(num_constraints):
                    u_c, v_c, w_c = constraints[c]
                    # Vectorized fill for row c
                    # Indices for U part: u_c * r + [0..r-1]
                    idx_u = u_c * r + np.arange(r)
                    idx_v = v_c * r + np.arange(r)
                    idx_w = w_c * r + np.arange(r)
                    
                    A_U[c, idx_u] = V[v_c, :] * W[w_c, :]
                    A_V[c, idx_v] = U[u_c, :] * W[w_c, :]
                    A_W[c, idx_w] = U[u_c, :] * V[v_c, :]
                
                # Solve least squares
                try:
                    U_flat, _, _, _ = np.linalg.lstsq(A_U, np.ones(num_constraints), rcond=None)
                    V_flat, _, _, _ = np.linalg.lstsq(A_V, np.ones(num_constraints), rcond=None)
                    W_flat, _, _, _ = np.linalg.lstsq(A_W, np.ones(num_constraints), rcond=None)
                except np.linalg.LinAlgError:
                    continue
                
                U = U_flat.reshape(n * m, r)
                V = V_flat.reshape(m * p, r)
                W = W_flat.reshape(p * n, r)
                
                # Normalize to keep values in check (scaling invariance)
                U = normalize_columns(U)
                V = normalize_columns(V)
                W = normalize_columns(W)
                
                # Check error
                err = error_function(U, V, W)
                
                if err < 1e-6:
                    # Try to round to nearest 0.5
                    # The solution might be scaled arbitrarily.
                    # We need to find a scaling factor alpha, beta, gamma such that
                    # alpha * U, beta * V, gamma * W have entries in {0, 0.5, 1, ...}
                    # And the product sums to 1.
                    # Actually, ALS solves for the values directly.
                    # If the solution is valid, the entries should be close to half-integers.
                    
                    # Let's try rounding directly first.
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    if check_exact(U_r, V_r, W_r):
                        return U_r, V_r, W_r
                    
                    # If direct rounding fails, it might be due to scaling.
                    # Try to adjust scaling.
                    # The condition is sum_k (a_k U[i,k]) (b_k V[j,k]) (c_k W[l,k]) = 1
                    # If we scale U by s, V by t, W by u, the product scales by stu.
                    # So we can fix one matrix's scale and adjust the others.
                    # However, the entries must be half-integers.
                    
                    # A more robust approach for rounding:
                    # Look at the non-zero entries. They should be close to multiples of 0.5.
                    # If they are not, maybe the rank is too low or we stuck in a bad local min.
                    
                    # Try a small search around the rounded values?
                    # Or try to solve for the scaling factors.
                    
                    # Let's check if the unrounded solution has a consistent scale.
                    # If U[i,k] is close to 0.5 * x, where x is integer.
                    
                    # If check_exact fails, we can try to refine using the rounded values as a start for a very short gradient step?
                    # Or just accept failure for this restart.
                    
                    # One more trick: Sometimes the solution is valid but scaled by 2 or 0.5.
                    # Try scaling the entire solution by 0.5, 1, 2?
                    # No, that changes the sum.
                    
                    # If we have U, V, W such that sum U V W = 1.
                    # If we replace U with U/2, we must replace V with 2V to keep sum=1.
                    # So we can try different distributions of the scale factor.
                    
                    # But the constraint is that entries must be half-integers.
                    # So we want U', V', W' such that:
                    # 1. U' entries in {0, 0.5, 1, ...}
                    # 2. V' entries in {0, 0.5, 1, ...}
                    # 3. W' entries in {0, 0.5, 1, ...}
                    # 4. sum U' V' W' = 1
                    
                    # Let's try to round and then check. If it fails, move on.
                    # The ALS solution for a valid decomposition often lands exactly on the integers/half-integers if the rank is sufficient and the initialization is good.
                    
                    # Let's try one more thing: if the error is small, maybe we can do a few steps of gradient descent from the rounded values to see if we can snap them?
                    # No, rounding is discrete.
                    
                    # Let's try to find the best scaling for each column?
                    # No, scaling is global per matrix or per column?
                    # The decomposition is sum_k U[:,k] V[:,k] W[:,k].
                    # We can scale column k of U by a_k, column k of V by b_k, column k of W by c_k.
                    # Condition: a_k b_k c_k = 1 for all k? No, that's if we want to preserve the term.
                    # Actually, we can scale U[:,k] by s_k, V[:,k] by 1/s_k, W[:,k] by 1.
                    # Then the term is unchanged.
                    # So we have freedom to scale columns independently.
                    
                    # We want to find scalings s_k, t_k, u_k such that:
                    # s_k * U[:,k] are half-integers
                    # t_k * V[:,k] are half-integers
                    # u_k * W[:,k] are half-integers
                    # And s_k t_k u_k = 1 (to preserve the product)
                    
                    # This is a hard constraint satisfaction problem per column.
                    # Given U[:,k], what scalings s make s*U[:,k] half-integers?
                    # If U[:,k] = [0.5, 1.0], s=1 works.
                    # If U[:,k] = [0.25, 0.5], s=2 works.
                    # If U[:,k] = [0.1, 0.2], no simple scaling works to half-integers.
                    
                    # So we can check for each column k if there exists a rational scaling that maps the vector to half-integers.
                    # Since the values are from a linear solve, they might be "almost" rational.
                    
                    # Let's just stick to simple rounding for now. If it works, great.
                    # If not, the ALS might not have converged to a valid discrete point.
                    
                    break 

        return None

    # Try lower ranks
    # Theoretical lower bound for 2x3x4 is likely around 12-16?
    # Let's try 20, 21, 22, 23.
    
    for r in [20, 21, 22, 23]:
        if time.time() > deadline - 10.0:
            break
            
        sol = try_rank_als(r, max_iters=30, num_restarts=50, seed_base=r*100)
        if sol:
            best_U, best_V, best_W = sol
            best_rank = r
            break
            
    # If we didn't find a better rank, return the seed
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
