"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _rank17_solution():
    n, m, p = DIMENSIONS
    n_a = n * m
    n_b = m * p
    n_c = p * n
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    terms = []
    # Strassen
    terms.append(([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]))
    terms.append(([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]))
    terms.append(([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]))
    terms.append(([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]))
    terms.append(([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]))
    terms.append(([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]))
    terms.append(([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]))
    # Direct
    terms.append(([0], [1], [2], [1], [4], [1]))
    terms.append(([1], [1], [5], [1], [4], [1]))
    terms.append(([3], [1], [2], [1], [5], [1]))
    terms.append(([4], [1], [5], [1], [5], [1]))
    terms.append(([2], [1], [6], [1], [0], [1]))
    terms.append(([2], [1], [7], [1], [2], [1]))
    terms.append(([2], [1], [8], [1], [4], [1]))
    terms.append(([5], [1], [6], [1], [1], [1]))
    terms.append(([5], [1], [7], [1], [3], [1]))
    terms.append(([5], [1], [8], [1], [5], [1]))
    
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for a, val in zip(u_idx, u_val):
            U[a, r] = val
        for b, val in zip(v_idx, v_val):
            V[b, r] = val
        for c, val in zip(w_idx, w_val):
            W[c, r] = val
    return U, V, W

def _try_rank16_als(start_U, start_V, start_W, T, deadline):
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = start_U.shape[0], start_V.shape[0], start_W.shape[0]
    rank = 16
    
    # We try to drop each of the 17 columns and see if ALS converges
    for drop_idx in range(17):
        if time.time() > deadline:
            return None
            
        # Initialize with the 16 remaining columns
        U = np.delete(start_U, drop_idx, axis=1).copy()
        V = np.delete(start_V, drop_idx, axis=1).copy()
        W = np.delete(start_W, drop_idx, axis=1).copy()
        
        # Add small random noise to break symmetry and allow movement
        U += np.random.randn(n_a, rank) * 0.01
        V += np.random.randn(n_b, rank) * 0.01
        W += np.random.randn(n_c, rank) * 0.01
        
        for it in range(200):
            if time.time() > deadline:
                return None
                
            # Solve for W
            # T = sum U_r V_r W_r
            # Reshape T to (n_a * n_b, n_c)
            T_mat = T.reshape(n_a * n_b, n_c)
            
            # Build matrix M_W of shape (n_a * n_b, rank)
            # M_W[:, r] = (U[:, r] * V[:, r]).flatten()
            M_W = np.zeros((n_a * n_b, rank))
            for r in range(rank):
                M_W[:, r] = (U[:, r] * V[:, r]).flatten()
                
            # Solve M_W W = T_mat
            # Least squares
            W_sol, residuals, rank_w, sv = np.linalg.lstsq(M_W, T_mat, rcond=None)
            W = W_sol
            
            # Solve for V
            # T_mat is (n_a * n_b, n_c). We need to solve for V.
            # This is trickier. We can iterate over c.
            # For each c, T[:, c] = sum_r (U_r * V_r) * W_{c,r}
            # Let W_c be the vector of W[:, c].
            # We want to find V such that sum_r W_{c,r} U_r V_r = T[:, c]
            # This is a bilinear problem. We can linearize by fixing U.
            # But we are solving for V.
            # We can use the fact that T[:, c] = sum_r (W_{c,r} U_r) V_r
            # Let A_c = sum_r W_{c,r} U_r? No, U_r is a vector.
            # We have a sum of outer products: sum_r (W_{c,r} U_r) (V_r)^T? No.
            # T_{ab,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
            # For fixed c, T_{ab,c} = sum_r (U_{a,r} W_{c,r}) V_{b,r}
            # This is a linear system for V_r if we treat U_{a,r} W_{c,r} as coefficients.
            # But V_r is a vector.
            # We can solve for V by flattening.
            # Let's use the same trick:
            # We want to find V (n_b x rank).
            # T (n_a x n_b x n_c) = sum_r U_r V_r W_r^T
            # Reshape T to (n_a * n_c, n_b)
            T_mat_vc = T.transpose(0, 2, 1).reshape(n_a * n_c, n_b)
            
            M_V = np.zeros((n_a * n_c, rank))
            for r in range(rank):
                M_V[:, r] = (U[:, r] * W[:, r]).flatten()
                
            V_sol, residuals, rank_v, sv = np.linalg.lstsq(M_V, T_mat_vc, rcond=None)
            V = V_sol
            
            # Solve for U
            # Reshape T to (n_b * n_c, n_a)
            T_mat_ua = T.transpose(1, 2, 0).reshape(n_b * n_c, n_a)
            
            M_U = np.zeros((n_b * n_c, rank))
            for r in range(rank):
                M_U[:, r] = (V[:, r] * W[:, r]).flatten()
                
            U_sol, residuals, rank_u, sv = np.linalg.lstsq(M_U, T_mat_ua, rcond=None)
            U = U_sol
            
            # Check convergence
            error = np.linalg.norm(T - np.einsum('ar,br,cr->abc', U, V, W))
            if error < 1e-9:
                # Try to snap to integers/half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                error_snap = np.linalg.norm(T - np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap))
                if error_snap < 1e-9:
                    return U_snap, V_snap, W_snap
                    
                # If snap fails, maybe we can try to refine with integer constraints?
                # For now, return the real solution if it's very close, 
                # but the task requires integer/half-integer.
                # If we can't snap, this attempt failed.
                pass
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Get rank 17 solution
    U17, V17, W17 = _rank17_solution()
    
    # Try to find rank 16
    result = _try_rank16_als(U17, V17, W17, T, deadline)
    
    if result is not None:
        U, V, W = result
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to rank 17
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
# EVOLVE-BLOCK-END
