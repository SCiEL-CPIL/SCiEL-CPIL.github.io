import numpy as np
import time
import sys
from fractions import Fraction
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Indices:
    # U: (n*m, R) -> index i*m + j
    # V: (m*p, R) -> index j*p + k
    # W: (p*n, R) -> index k*n + i
    
    size_u = n * m  # 12
    size_v = m * p  # 16
    size_w = p * n  # 12
    
    # Target Tensor T[u, v, w]
    TARGET_TENSOR = np.zeros((size_u, size_v, size_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                TARGET_TENSOR[u, v, w] = 1.0

    def tensor_contraction(U, V, W):
        return np.einsum('ur,vr,wr->uvw', U, V, W)

    def verify_solution(U, V, W):
        """Verify that U, V, W correctly decompose the tensor with exact arithmetic."""
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    total = 0
                    for c in range(len(U[0])):
                        total += U_frac[u_idx][c] * V_frac[v_idx][c] * W_frac[w_idx][c]
                    if total != 1:
                        return False
        return True

    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((size_u, rank_school))
    V_school = np.zeros((size_v, rank_school))
    W_school = np.zeros((size_w, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_solution = None
    best_rank = rank_school

    def objective(params, R):
        U = params[:size_u * R].reshape(size_u, R)
        V = params[size_u * R : size_u * R + size_v * R].reshape(size_v, R)
        W = params[size_u * R + size_v * R :].reshape(size_w, R)
        
        T = tensor_contraction(U, V, W)
        
        # Error: emphasize non-zero entries, penalize non-zero errors in zero entries
        mask = TARGET_TENSOR != 0
        error_nonzero = np.sum((T[mask] - TARGET_TENSOR[mask])**2)
        error_zero = np.sum(T[~mask]**2)
        
        # Weighting: we want exact match.
        return error_nonzero + 0.5 * error_zero

    def gradient(params, R):
        U = params[:size_u * R].reshape(size_u, R)
        V = params[size_u * R : size_u * R + size_v * R].reshape(size_v, R)
        W = params[size_u * R + size_v * R :].reshape(size_w, R)
        
        T = tensor_contraction(U, V, W)
        error = T - TARGET_TENSOR
        # Weight the zero entries in the gradient as well?
        # The objective was sum((T-1)^2) for mask + 0.5*sum(T^2) for ~mask
        # grad T wrt U is einsum('uvw,vr,wr->ur', error, V, W)
        # But the weight for zero entries is different.
        # Let's just use the standard gradient for sum((T-T_target)^2)
        # and rely on the fact that the minimum of the weighted sum is likely close.
        # Actually, let's compute the exact gradient of the weighted objective.
        # L = sum_mask (T-1)^2 + 0.5 sum_~mask T^2
        # dL/dT = 2(T-1) on mask, T on ~mask
        # dL/dU = einsum('uvw,vr,wr->ur', dL/dT, V, W)
        
        dL_dT = np.zeros_like(T)
        dL_dT[mask] = 2 * (T[mask] - 1)
        dL_dT[~mask] = T[~mask]
        
        grad_U = np.einsum('uvw,vr,wr->ur', dL_dT, V, W)
        grad_V = np.einsum('uvw,ur,wr->vr', dL_dT, U, W)
        grad_W = np.einsum('uvw,ur,vr->wr', dL_dT, U, V)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    # Try ranks from 47 down to 40
    for R in range(47, 39, -1):
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts
        num_restarts = 10
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + R * 100 + restart)
            
            # Initialize with random values
            # Scale initialization to be reasonable
            params = np.random.randn(size_u * R + size_v * R + size_w * R) * 1.0
            
            try:
                res = minimize(objective, params, args=(R,), method='L-BFGS-B', 
                               jac=gradient,
                               options={'maxiter': 100, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
                
                if res.fun < 1e-6:
                    U_opt = res.x[:size_u * R].reshape(size_u, R)
                    V_opt = res.x[size_u * R : size_u * R + size_v * R].reshape(size_v, R)
                    W_opt = res.x[size_u * R + size_v * R :].reshape(size_w, R)
                    
                    # Snap to half-integers
                    U_snap = np.round(U_opt * 2) / 2
                    V_snap = np.round(V_opt * 2) / 2
                    W_snap = np.round(W_opt * 2) / 2
                    
                    # Verify snapped solution
                    T_snap = tensor_contraction(U_snap, V_snap, W_snap)
                    if np.allclose(T_snap, TARGET_TENSOR, atol=1e-9):
                        # Verify with exact arithmetic
                        if verify_solution(U_snap, V_snap, W_snap):
                            best_rank = R
                            best_solution = {
                                "U": U_snap.tolist(),
                                "V": V_snap.tolist(),
                                "W": W_snap.tolist()
                            }
                            break  # Found a better solution
            except Exception:
                pass
        
        if best_solution and best_rank < rank_school:
            break  # Found a solution with rank < 48

    if best_solution:
        return best_solution
    else:
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
