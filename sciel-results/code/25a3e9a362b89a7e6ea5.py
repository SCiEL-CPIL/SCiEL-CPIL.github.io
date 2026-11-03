import time
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
_DEADLINE = 140

def _build_target_tensor(n, m, p):
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-8):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    if np.max(np.abs(T_recon - T)) > tol:
        return False
    U_h = U * 2
    V_h = V * 2
    W_h = W * 2
    if not np.allclose(U_h, np.round(U_h), atol=1e-5):
        return False
    if not np.allclose(V_h, np.round(V_h), atol=1e-5):
        return False
    if not np.allclose(W_h, np.round(W_h), atol=1e-5):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_block(j0, j1, k0, k1, n=2, m=3, p=4):
    terms = []
    
    def make_term(u_coeffs, v_coeffs, w_coeffs):
        U = np.zeros(n * m)
        V = np.zeros(m * p)
        W = np.zeros(p * n)
        
        for (i, j), val in u_coeffs.items():
            U[i * m + j] = val
        for (j, k), val in v_coeffs.items():
            V[j * p + k] = val
        for (k, i), val in w_coeffs.items():
            W[k * n + i] = val
            
        return U, V, W

    terms.append(make_term(
        {(0, j0): 1, (1, j1): 1},
        {(j0, k0): 1, (j1, k1): 1},
        {(k0, 0): 1, (k1, 1): 1}
    ))
    
    terms.append(make_term(
        {(1, j0): 1, (1, j1): 1},
        {(j0, k0): 1},
        {(k0, 1): 1, (k1, 1): -1}
    ))
    
    terms.append(make_term(
        {(0, j0): 1},
        {(j1, k0): 1, (j1, k1): -1},
        {(k1, 0): 1, (k1, 1): 1}
    ))
    
    terms.append(make_term(
        {(1, j1): 1},
        {(j0, k1): 1, (j0, k0): -1},
        {(k0, 0): 1, (k0, 1): 1}
    ))
    
    terms.append(make_term(
        {(0, j0): 1, (0, j1): 1},
        {(j1, k1): 1},
        {(k0, 0): -1, (k1, 0): 1}
    ))
    
    terms.append(make_term(
        {(1, j0): 1, (0, j0): -1},
        {(j0, k0): 1, (j0, k1): 1},
        {(k1, 1): 1}
    ))
    
    terms.append(make_term(
        {(0, j1): 1, (1, j1): -1},
        {(j1, k0): 1, (j1, k1): 1},
        {(k0, 0): 1}
    ))
    
    return terms

def _rational_reconstruct(x, max_den=1000000):
    """Reconstruct a rational number from a float using continued fractions."""
    if abs(x) < 1e-10:
        return Fraction(0, 1)
    
    x_frac = Fraction(x).limit_denominator(max_den)
    return x_frac

def _try_rank_with_ls(n, m, p, R, T, max_iter=1000, tol=1e-8):
    """
    Try to find a rank-R decomposition using iterative linear least squares.
    For each rank-1 term, fix the other two factors and solve for the third.
    """
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Initialize with random or structured values
    U = np.random.randn(dim_U, R)
    V = np.random.randn(dim_V, R)
    W = np.random.randn(dim_W, R)
    
    # Normalize columns to avoid scaling issues
    for r in range(R):
        nU = np.linalg.norm(U[:, r])
        nV = np.linalg.norm(V[:, r])
        nW = np.linalg.norm(W[:, r])
        if nU > 1e-10:
            U[:, r] /= nU
        if nV > 1e-10:
            V[:, r] /= nV
        if nW > 1e-10:
            W[:, r] /= nW
    
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE - 30:
            break
        
        # For each rank-1 term, update each factor
        for r in range(R):
            # Update U[:, r] given V[:, r] and W[:, r]
            # T = sum_r U_r ⊗ V_r ⊗ W_r
            # For fixed r, the contribution is U_r ⊗ V_r ⊗ W_r
            # We want to minimize ||T - sum_s U_s ⊗ V_s ⊗ W_s||
            # This is linear in U[:, r]
            
            # Residual without term r
            T_res = T.copy()
            for s in range(R):
                if s != r:
                    T_res -= np.einsum('a,b,c->abc', U[:, s], V[:, s], W[:, s])
            
            # We need to solve: minimize ||T_res - U_r ⊗ V_r ⊗ W_r||_F
            # This is equivalent to: for each (a,b,c), T_res[a,b,c] ≈ U[a,r] * V[b,r] * W[c,r]
            # Vectorize: let v_r = V[:, r] ⊗ W[:, r] (size dim_V * dim_W)
            # Then T_res reshaped to (dim_U, dim_V*dim_W) ≈ U[:, r] ⊗ v_r
            # So U[:, r] = argmin ||T_res_reshaped - U[:, r] v_r^T||
            # This is a rank-1 approximation: U[:, r] = (T_res_reshaped @ v_r) / (v_r @ v_r)
            
            v_r = np.outer(V[:, r], W[:, r])
            T_res_reshaped = T_res.reshape(dim_U, -1)
            
            denom = np.dot(v_r, v_r)
            if denom > 1e-10:
                U[:, r] = T_res_reshaped @ v_r / denom
            
            # Similarly for V
            T_res = T.copy()
            for s in range(R):
                if s != r:
                    T_res -= np.einsum('a,b,c->abc', U[:, s], V[:, s], W[:, s])
            
            u_r = np.outer(U[:, r], W[:, r])
            T_res_reshaped = T_res.reshape(dim_V, dim_U, dim_W).transpose(1, 0, 2).reshape(dim_U, -1).T
            # Actually, let's do it more carefully
            # T_res[a,b,c] ≈ U[a,r] * V[b,r] * W[c,r]
            # For fixed a,c: T_res[a,:,c] ≈ V[:,r] * (U[a,r]*W[c,r])
            # So V[:,r] = sum_{a,c} T_res[a,:,c] * U[a,r] * W[c,r] / sum_{a,c} (U[a,r]*W[c,r])^2
            
            coeff = np.outer(U[:, r], W[:, r])
            T_res_reshaped = T_res.reshape(dim_U, dim_V, dim_W)
            # Sum over a and c: V[:,r] = sum_a sum_c T_res[a,:,c] * coeff[a,c]
            V[:, r] = np.tensordot(T_res_reshaped, coeff, axes=([0, 2], [0, 1]))
            denom = np.sum(coeff * coeff)
            if denom > 1e-10:
                V[:, r] /= denom
            
            # Similarly for W
            T_res = T.copy()
            for s in range(R):
                if s != r:
                    T_res -= np.einsum('a,b,c->abc', U[:, s], V[:, s], W[:, s])
            
            coeff = np.outer(U[:, r], V[:, r])
            T_res_reshaped = T_res.reshape(dim_U, dim_V, dim_W)
            W[:, r] = np.tensordot(T_res_reshaped, coeff, axes=([0, 1], [0, 1]))
            denom = np.sum(coeff * coeff)
            if denom > 1e-10:
                W[:, r] /= denom
        
        # Check residual
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        residual = np.linalg.norm(T_recon - T)
        
        if residual < best_residual:
            best_residual = residual
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if residual < tol:
            break
    
    return best_U, best_V, best_W, best_residual

def _try_rank_with_ls_v2(n, m, p, R, T, max_iter=500, tol=1e-8):
    """
    Improved ALS with better initialization and convergence check.
    """
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Use structured initialization based on schoolbook
    U = np.zeros((dim_U, R))
    V = np.zeros((dim_V, R))
    W = np.zeros((dim_W, R))
    
    # Initialize with first R schoolbook terms
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= R:
                    break
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
        if col >= R:
            break
    
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE - 30:
            break
        
        for r in range(R):
            # Update U[:, r]
            T_res = T.copy()
            for s in range(R):
                if s != r:
                    T_res -= np.einsum('a,b,c->abc', U[:, s], V[:, s], W[:, s])
            
            v_r = np.outer(V[:, r], W[:, r])
            T_res_reshaped = T_res.reshape(dim_U, -1)
            denom = np.dot(v_r, v_r)
            if denom > 1e-10:
                U[:, r] = T_res_reshaped @ v_r / denom
            
            # Update V[:, r]
            T_res = T.copy()
            for s in range(R):
                if s != r:
                    T_res -= np.einsum('a,b,c->abc', U[:, s], V[:, s], W[:, s])
            
            coeff = np.outer(U[:, r], W[:, r])
            T_res_reshaped = T_res.reshape(dim_U, dim_V, dim_W)
            V[:, r] = np.tensordot(T_res_reshaped, coeff, axes=([0, 2], [0, 1]))
            denom = np.sum(coeff * coeff)
            if denom > 1e-10:
                V[:, r] /= denom
            
            # Update W[:, r]
            T_res = T.copy()
            for s in range(R):
                if s != r:
                    T_res -= np.einsum('a,b,c->abc', U[:, s], V[:, s], W[:, s])
            
            coeff = np.outer(U[:, r], V[:, r])
            T_res_reshaped = T_res.reshape(dim_U, dim_V, dim_W)
            W[:, r] = np.tensordot(T_res_reshaped, coeff, axes=([0, 1], [0, 1]))
            denom = np.sum(coeff * coeff)
            if denom > 1e-10:
                W[:, r] /= denom
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        residual = np.linalg.norm(T_recon - T)
        
        if residual < best_residual:
            best_residual = residual
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if residual < tol:
            break
    
    return best_U, best_V, best_W, best_residual

def _apply_rational_reconstruction(U, V, W, max_den=1000000):
    """Apply rational reconstruction to all entries."""
    U_frac = np.zeros_like(U)
    V_frac = np.zeros_like(V)
    W_frac = np.zeros_like(W)
    
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            U_frac[i, j] = float(_rational_reconstruct(U[i, j], max_den))
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            V_frac[i, j] = float(_rational_reconstruct(V[i, j], max_den))
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_frac[i, j] = float(_rational_reconstruct(W[i, j], max_den))
    
    return U_frac, V_frac, W_frac

def construct():
    n, m, p = 2, 3, 4
    T = _build_target_tensor(n, m, p)
    
    # First, get the current best known rank (22 from Strassen + schoolbook)
    U_list = []
    V_list = []
    W_list = []
    
    terms1 = _strassen_block(0, 1, 0, 1, n, m, p)
    for U, V, W in terms1:
        U_list.append(U)
        V_list.append(V)
        W_list.append(W)
        
    terms2 = _strassen_block(0, 1, 2, 3, n, m, p)
    for U, V, W in terms2:
        U_list.append(U)
        V_list.append(V)
        W_list.append(W)
        
    for i in range(n):
        for k in range(p):
            U = np.zeros(n * m)
            V = np.zeros(m * p)
            W = np.zeros(p * n)
            U[i * m + 2] = 1
            V[2 * p + k] = 1
            W[k * n + i] = 1
            U_list.append(U)
            V_list.append(V)
            W_list.append(W)
            
    U_best = np.column_stack(U_list)
    V_best = np.column_stack(V_list)
    W_best = np.column_stack(W_list)
    best_rank = 22
    
    if _verify(U_best, V_best, W_best, T):
        # Try to find lower rank
        for R in range(15, best_rank):
            if time.time() > _DEADLINE - 60:
                break
            
            # Try multiple random initializations
            for attempt in range(3):
                if time.time() > _DEADLINE - 30:
                    break
                
                U, V, W, residual = _try_rank_with_ls_v2(n, m, p, R, T, max_iter=200)
                
                if residual < 1e-6:
                    # Apply rational reconstruction
                    U_r, V_r, W_r = _apply_rational_reconstruction(U, V, W)
                    
                    if _verify(U_r, V_r, W_r, T):
                        best_rank = R
                        U_best, V_best, W_best = U_r, V_r, W_r
                        break
                elif residual < best_residual if 'best_residual' in locals() else residual < 1e-3:
                    if 'best_residual' not in locals() or residual < best_residual:
                        best_residual = residual
                        U_best, V_best, W_best = U, V, W
        
        if _verify(U_best, V_best, W_best, T):
            return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
    
    # Fallback to schoolbook
    U_s, V_s, W_s = _schoolbook(n, m, p)
    return {"U": U_s, "V": V_s, "W": W_s}
