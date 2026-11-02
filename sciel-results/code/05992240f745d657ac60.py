import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Dimensions of U, V, W
    dim_U = n * m  # 6
    dim_V = m * p  # 21
    dim_W = p * n  # 14
    
    # Target tensor T of shape (dim_U, dim_V, dim_W)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    start_time = time.time()
    deadline = 120.0  # 120 seconds limit
    
    # Try ranks from low to high
    # Theoretical lower bound is max(dim_U, dim_V, dim_W) = 21? No, rank of flattening.
    # Flattening U-((V,W)) is dim_U x (dim_V*dim_W) = 6 x 294. Rank is 6.
    # Flattening V-((U,W)) is dim_V x (dim_U*dim_W) = 21 x 84. Rank is 21?
    # Let's check. The tensor is sparse.
    # A safe lower bound is the maximum rank of any flattening.
    # For M_{2,3,7}, the rank is known to be 24? Or lower?
    # Let's try starting from 20.
    
    best_result = None
    best_rank = n * m * p  # 42
    
    # If time is short, just return schoolbook
    if deadline - (time.time() - start_time) < 10:
        return _schoolbook(n, m, p)
        
    for rank in range(20, best_rank):
        if time.time() - start_time > deadline - 5:
            break
            
        # Try a few random restarts for each rank
        found = False
        for attempt in range(10):
            if time.time() - start_time > deadline - 5:
                break
                
            U = np.random.randn(dim_U, rank)
            V = np.random.randn(dim_V, rank)
            W = np.random.randn(dim_W, rank)
            
            # Scale to help convergence?
            # Normalize columns
            for c in range(rank):
                norm_u = np.linalg.norm(U[:, c])
                if norm_u > 1e-10: U[:, c] /= norm_u
                norm_v = np.linalg.norm(V[:, c])
                if norm_v > 1e-10: V[:, c] /= norm_v
                norm_w = np.linalg.norm(W[:, c])
                if norm_w > 1e-10: W[:, c] /= norm_w
                
            # ALS iterations
            for it in range(100):
                if time.time() - start_time > deadline - 5:
                    break
                    
                # Update U
                # T = sum_c U[:,c] v_c w_c^T
                # For fixed c, U[:,c] should approximate the "residual" projected
                # Standard ALS:
                # Fix V, W. Solve for U.
                # The equation is T = U * (V * W) ? No.
                # T_{ijk} = sum_c U_{ic} V_{jc} W_{kc}
                # For a fixed c, we want to find U_{ic} such that it best fits the part of T not explained by other c's?
                # Or simply minimize sum_c || T - sum U_c v_c w_c ||^2.
                # This is not linear in U if we update all at once.
                # But if we update one column at a time, it is linear.
                
                # Update column c of U
                for c in range(rank):
                    # Compute residual T - sum_{c' != c} U_{c'} v_{c'} w_{c'}^T
                    # Residual_{ijk} = T_{ijk} - sum_{c' != c} U_{i,c'} V_{j,c'} W_{k,c'}
                    # We want to find U_{:,c} to minimize || Residual - U_{:,c} v_c w_c^T ||^2
                    # Let A = v_c w_c^T (shape dim_V x dim_W)
                    # We want to minimize || T - U_{:,c} A ||^2_F
                    # This is a least squares problem: T (reshaped) = U_{:,c} A
                    # T is dim_U x dim_V x dim_W.
                    # Reshape T to (dim_U, dim_V*dim_W).
                    # A is dim_V x dim_W -> reshape to (dim_V*dim_W, 1)? No, A is a matrix.
                    # The term U_{:,c} v_c w_c^T is an outer product of U_{:,c} and (v_c w_c^T).
                    # Let M = v_c w_c^T.
                    # We want to approximate T by U_{:,c} M.
                    # This is not a standard matrix multiplication unless M is a vector.
                    # Wait, U_{:,c} is a vector. v_c is a vector. w_c is a vector.
                    # The term is U_{:,c} (v_c w_c^T). This is a rank-1 tensor.
                    # To solve for U_{:,c}, we can use the gradient.
                    # grad_U = 2 * (T - U_{:,c} v_c w_c^T) * (v_c w_c^T) ? No.
                    # The gradient of || T - u v w^T ||^2 with respect to u is:
                    # -2 * (v w^T) (T - u v w^T) ?
                    # Let's use the normal equations.
                    # Minimize || T - u (v w^T) ||^2
                    # This is equivalent to minimizing || T_{i,:, :} - u_i (v w^T) ||^2 for each i? No.
                    # It's a single vector u.
                    # The term is u_i v_j w_k.
                    # The error is sum_{i,j,k} (T_{ijk} - u_i v_j w_k)^2.
                    # Derivative wrt u_i: -2 sum_{j,k} (T_{ijk} - u_i v_j w_k) v_j w_k = 0
                    # sum_{j,k} T_{ijk} v_j w_k = u_i sum_{j,k} v_j^2 w_k^2
                    # u_i = (sum_{j,k} T_{ijk} v_j w_k) / (||v||^2 ||w||^2)
                    
                    v_c = V[:, c]
                    w_c = W[:, c]
                    denom = np.dot(v_c, v_c) * np.dot(w_c, w_c)
                    if denom < 1e-10:
                        continue
                    # Compute numerator for each i
                    # sum_{j,k} T_{ijk} v_j w_k
                    # T is sparse. We can precompute T_vw for each i?
                    # T_vw[i] = sum_{j,k} T_{ijk} v_j w_k
                    # This is a contraction of T with v and w.
                    # T is shape (dim_U, dim_V, dim_W).
                    # np.einsum('ijk,j,k->i', T, v_c, w_c)
                    num = np.einsum('ijk,j,k->i', T, v_c, w_c)
                    U[:, c] = num / denom
                    
                # Similarly for V and W
                for c in range(rank):
                    u_c = U[:, c]
                    w_c = W[:, c]
                    denom = np.dot(u_c, u_c) * np.dot(w_c, w_c)
                    if denom < 1e-10:
                        continue
                    num = np.einsum('ijk,i,k->j', T, u_c, w_c)
                    V[:, c] = num / denom
                    
                for c in range(rank):
                    u_c = U[:, c]
                    v_c = V[:, c]
                    denom = np.dot(u_c, u_c) * np.dot(v_c, v_c)
                    if denom < 1e-10:
                        continue
                    num = np.einsum('ijk,i,j->k', T, u_c, v_c)
                    W[:, c] = num / denom
                    
            # Check error
            # Compute T_approx = sum_c U[:,c] v_c w_c^T
            # T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
            T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
            error = np.linalg.norm(T - T_approx)
            
            if error < 1e-6:
                # Found a solution. Try to round to integers/half-integers.
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                T_approx_r = np.einsum('ic,jc,kc->ijk', U_rounded, V_rounded, W_rounded)
                error_r = np.linalg.norm(T - T_approx_r)
                
                if error_r < 1e-9:
                    # Valid solution with half-integers
                    best_result = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    best_rank = rank
                    found = True
                    break
                else:
                    # Maybe integers?
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    T_approx_i = np.einsum('ic,jc,kc->ijk', U_int, V_int, W_int)
                    error_i = np.linalg.norm(T - T_approx_i)
                    if error_i < 1e-9:
                        best_result = {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
                        best_rank = rank
                        found = True
                        break
                        # If not rounded, we can still return the float solution if the evaluator accepts it?
                        # "integer/half-integer entries" is specified. So we must round.
                        # If rounding fails, we can try to find a better numerical solution or accept a higher rank.
                        
            # If we found a valid rounded solution, we can stop searching for this rank
            if found:
                break
                
        if found:
            break
            
    if best_result is not None:
        return best_result
    else:
        # Fallback to schoolbook
        return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
