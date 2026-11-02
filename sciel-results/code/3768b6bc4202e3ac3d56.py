"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145  # Safety margin before 150s kill

    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1

    def _verify(U, V, W):
        """Verify decomposition: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def _snap_to_half_int(arr):
        """Snap values to nearest half-integer"""
        return np.round(2 * arr) / 2.0

    def _solve_rank(R, max_iters=200, n_restarts=5):
        """
        Attempt to find a rank-R decomposition using Gauss-Newton refinement
        starting from random initializations.
        """
        num_u = n * m
        num_v = m * p
        num_w = p * n
        total_vars = (num_u + num_v + num_w) * R
        num_eqs = n * m * p
        
        for restart in range(n_restarts):
            if time.time() - start_time > time_limit:
                return None

            # Random initialization with small integers
            # Scale matters for convergence
            U = np.random.randint(-2, 3, size=(num_u, R)).astype(float)
            V = np.random.randint(-2, 3, size=(num_v, R)).astype(float)
            W = np.random.randint(-2, 3, size=(num_w, R)).astype(float)
            
            # Ensure some non-zero structure
            if np.all(U == 0): U[0,0] = 1
            if np.all(V == 0): V[0,0] = 1
            if np.all(W == 0): W[0,0] = 1

            x = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            for it in range(max_iters):
                if time.time() - start_time > time_limit:
                    return None
                
                # Compute residuals f[i,j,k]
                f = np.zeros(num_eqs)
                # Compute Jacobian J (num_eqs x total_vars)
                J = np.zeros((num_eqs, total_vars))
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Residual
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            
                            # f = sum_r U[u,r] V[v,r] W[w,r] - 1
                            val = 0.0
                            for r in range(R):
                                u_val = U[u_idx, r]
                                v_val = V[v_idx, r]
                                w_val = W[w_idx, r]
                                term = u_val * v_val * w_val
                                val += term
                                
                                # Jacobian derivatives
                                # d/dU[u,r] = v_val * w_val
                                # d/dV[v,r] = u_val * w_val
                                # d/dW[w,r] = u_val * v_val
                                
                                base_u = r * num_u + u_idx
                                base_v = (num_u + r * num_v) + v_idx
                                base_w = (num_u + num_v + r * num_w) + w_idx
                                
                                J[eq_idx, base_u] = v_val * w_val
                                J[eq_idx, base_v] = u_val * w_val
                                J[eq_idx, base_w] = u_val * v_val
                            
                            f[eq_idx] = val - 1.0
                            eq_idx += 1

                # Check convergence
                if np.max(np.abs(f)) < 1e-8:
                    # Snap to half-integers
                    U_snap = _snap_to_half_int(U)
                    V_snap = _snap_to_half_int(V)
                    W_snap = _snap_to_half_int(W)
                    
                    # Verify snapped solution
                    if _verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                        return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                    else:
                        # If snap failed, maybe try integer snap?
                        U_int = np.round(U)
                        V_int = np.round(V)
                        W_int = np.round(W)
                        if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                            return U_int.tolist(), V_int.tolist(), W_int.tolist()
                        # Else, continue refining or break
                        break

                # Gauss-Newton step
                # Solve J delta_x = -f
                # Use least squares since J is wide (under-determined)
                # delta_x = J^T (J J^T)^{-1} (-f)
                try:
                    JJT = J @ J.T
                    # Add regularization if singular
                    JJT += 1e-10 * np.eye(num_eqs)
                    # Solve for y
                    y = np.linalg.solve(JJT, -f)
                    delta_x = J.T @ y
                except np.linalg.LinAlgError:
                    # Fallback to simple gradient step or break
                    break

                # Damping
                alpha = 1.0
                # Simple line search to ensure decrease
                for _ in range(10):
                    x_new = x + alpha * delta_x
                    U_new = x_new[:num_u*R].reshape(num_u, R)
                    V_new = x_new[num_u*R:2*num_u*R].reshape(num_v, R)
                    W_new = x_new[2*num_u*R:].reshape(num_w, R)
                    
                    # Compute new residual norm
                    f_new = np.zeros(num_eqs)
                    eq_idx = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                u_idx = i * m + j
                                v_idx = j * p + k
                                w_idx = k * n + i
                                val = np.sum(U_new[u_idx] * V_new[v_idx] * W_new[w_idx])
                                f_new[eq_idx] = val - 1.0
                                eq_idx += 1
                    
                    if np.sum(f_new**2) < np.sum(f**2):
                        x = x_new
                        U = U_new
                        V = V_new
                        W = W_new
                        break
                    else:
                        alpha *= 0.5
                
                if alpha < 1e-4:
                    break

        return None

    best_rank = rank_schoolbook
    best_result = (U_sb, V_sb, W_sb)
    
    # Try ranks from 23 down to 12
    # We prioritize finding *any* improvement.
    for r in range(rank_schoolbook - 1, 11, -1):
        if time.time() - start_time > time_limit:
            break
        
        result = _solve_rank(r, max_iters=100, n_restarts=10)
        if result is not None:
            best_rank = r
            best_result = result
            # Since we are going downwards, the first valid one is the best for this loop
            # But we want the MINIMAL rank, so we should continue?
            # No, the loop is `for r in range(23, 11, -1)`.
            # So we find rank 23, then 22, etc.
            # We should stop if we find a very low rank?
            # The problem asks for "rank-minimal".
            # So we should keep searching for lower ranks until time runs out.
            # However, if we find rank 12, we can stop.
            pass
        
    # If we found a better solution, return it
    if best_rank < rank_schoolbook:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    
    # Otherwise return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
