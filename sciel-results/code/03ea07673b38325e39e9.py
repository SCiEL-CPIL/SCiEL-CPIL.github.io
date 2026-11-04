"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import sys

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    """Build the schoolbook decomposition for matrix multiplication."""
    n, m, p = DIMENSIONS
    r = n * m * p
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Compute T_decomp[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Use efficient computation
    T_decomp = np.zeros((n*m, m*p, p*n))
    for r in range(U.shape[1]):
        T_decomp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    # Check exactness (with tolerance for floating point)
    diff = np.abs(T_decomp - T)
    return np.max(diff) < 1e-9

def _round_to_half_integer(arr):
    """Round array entries to nearest half-integer."""
    return np.round(arr * 2.0) / 2.0

def _build_strassen_2x2():
    """
    Build a Strassen-like decomposition for 2x2 matrix multiplication.
    Returns U, V, W for multiplying two 2x2 matrices.
    """
    # Strassen's algorithm uses 7 multiplications for 2x2 matrices
    # For A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[a*e + b*g, a*f + b*h], [c*e + d*g, c*f + d*h]]
    # 
    # Strassen's 7 products:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # 
    # C[0,0] = p1 + p4 - p5 + p7
    # C[0,1] = p3 + p5
    # C[1,0] = p2 + p4
    # C[1,1] = p1 - p2 + p3 + p6
    #
    # In our tensor decomposition framework:
    # A has indices: a_idx = i*2 + j for 2x2
    # B has indices: b_idx = j*2 + k for 2x2
    # C has indices: c_idx = k*2 + i for 2x2 (trace convention: W[k*n+i,r])
    
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n*m, r))  # (4, 7)
    V = np.zeros((m*p, r))  # (4, 7)
    W = np.zeros((p*n, r))  # (4, 7)
    
    # A indices: 0=a(0,0), 1=b(0,1), 2=c(1,0), 3=d(1,1)
    # B indices: 0=e(0,0), 1=f(0,1), 2=g(1,0), 3=h(1,1)
    # C indices: 0=C[0,0](k=0,i=0), 1=C[0,1](k=0,i=1), 2=C[1,0](k=1,i=0), 3=C[1,1](k=1,i=1)
    
    # p1 = (a+d)(e+h)
    U[0, 0] = 1.0  # a
    U[3, 0] = 1.0  # d
    V[0, 0] = 1.0  # e
    V[3, 0] = 1.0  # h
    W[0, 0] = 1.0  # C[0,0]
    W[3, 0] = 1.0  # C[1,1]
    
    # p2 = (c+d)e
    U[2, 1] = 1.0  # c
    U[3, 1] = 1.0  # d
    V[0, 1] = 1.0  # e
    W[2, 1] = 1.0  # C[1,0]
    
    # p3 = a(f-h)
    U[0, 2] = 1.0  # a
    V[1, 2] = 1.0  # f
    V[3, 2] = -1.0  # h
    W[1, 2] = 1.0  # C[0,1]
    W[3, 2] = 1.0  # C[1,1]
    
    # p4 = d(g-e)
    U[3, 3] = 1.0  # d
    V[0, 3] = -1.0  # e
    V[2, 3] = 1.0  # g
    W[0, 3] = 1.0  # C[0,0]
    W[2, 3] = 1.0  # C[1,0]
    
    # p5 = (a+b)h
    U[0, 4] = 1.0  # a
    U[1, 4] = 1.0  # b
    V[3, 4] = 1.0  # h
    W[0, 4] = -1.0  # C[0,0]
    W[1, 4] = 1.0  # C[0,1]
    
    # p6 = (c-a)(e+f)
    U[0, 5] = -1.0  # a
    U[2, 5] = 1.0  # c
    V[0, 5] = 1.0  # e
    V[1, 5] = 1.0  # f
    W[3, 5] = 1.0  # C[1,1]
    
    # p7 = (b-d)(g+h)
    U[1, 6] = 1.0  # b
    U[3, 6] = -1.0  # d
    V[2, 6] = 1.0  # g
    V[3, 6] = 1.0  # h
    W[0, 6] = 1.0  # C[0,0]
    
    return U, V, W

def _build_hybrid(n, m, p):
    """
    Build a hybrid decomposition using block structure.
    Try to exploit the structure of 4x5 * 5x6 multiplication.
    """
    # For now, return schoolbook as fallback
    U, V, W = _build_schoolbook()
    return U, V, W

def _optimize_decomposition_fast(U, V, W, target_rank, max_iters=20, tol=1e-10):
    """
    Fast optimization to try to reduce the rank of the decomposition.
    Uses alternating least squares with early termination.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 100.0  # Leave 40s safety margin
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    r = target_rank
    U = U[:, :r].copy()
    V = V[:, :r].copy()
    W = W[:, :r].copy()
    
    def compute_residual():
        """Compute squared residual efficiently."""
        T_decomp = np.zeros((n*m, m*p, p*n))
        for r_idx in range(r):
            T_decomp += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.sum((T_decomp - T) ** 2)
    
    prev_res = compute_residual()
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Update U: for each a, solve M_a @ U[a,:] = T[a,:,:].ravel()
        # M_a[(b,c), r] = V[b,r] * W[c,r]
        for a in range(n*m):
            M_a = np.einsum('br,cr->bcr', V, W).reshape(m*p*p*n, r)
            t_a = T[a].reshape(-1)
            try:
                U[a, :] = np.linalg.lstsq(M_a, t_a, rcond=None)[0]
            except:
                pass
        
        # Update V: for each b, solve M_b @ V[b,:] = T[:,b,:].ravel()
        for b in range(m*p):
            M_b = np.einsum('ar,cr->acr', U, W).reshape(n*m*p*n, r)
            t_b = T[:, b, :].reshape(-1)
            try:
                V[b, :] = np.linalg.lstsq(M_b, t_b, rcond=None)[0]
            except:
                pass
        
        # Update W: for each c, solve M_c @ W[c,:] = T[:,:,c].ravel()
        for c in range(p*n):
            M_c = np.einsum('ar,br->abr', U, V).reshape(n*m*m*p, r)
            t_c = T[:, :, c].reshape(-1)
            try:
                W[c, :] = np.linalg.lstsq(M_c, t_c, rcond=None)[0]
            except:
                pass
        
        res = compute_residual()
        if abs(prev_res - res) < tol:
            break
        prev_res = res
    
    return U, V, W, prev_res

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    deadline = time.time() + 140.0  # 10s safety margin before 150s kill
    
    # Start with schoolbook decomposition
    U0, V0, W0 = _build_schoolbook()
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_schoolbook
    best_res = 0.0
    
    # Verify schoolbook is valid
    if _verify_decomposition(U0, V0, W0):
        best_res = 0.0
    
    # Try to reduce rank
    # Try various target ranks, starting from lower values
    for target_rank in [80, 90, 100, 110]:
        if time.time() > deadline - 20:
            break
        
        U_t, V_t, W_t, res = _optimize_decomposition_fast(U0, V0, W0, target_rank)
        
        # Round to half-integers and verify
        U_rounded = _round_to_half_integer(U_t)
        V_rounded = _round_to_half_integer(V_t)
        W_rounded = _round_to_half_integer(W_t)
        
        if _verify_decomposition(U_rounded, V_rounded, W_rounded):
            best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
            best_rank = target_rank
            best_res = 0.0
            break
        
        # Also try without rounding
        if res < 1e-8 and _verify_decomposition(U_t, V_t, W_t):
            best_U, best_V, best_W = U_t, V_t, W_t
            best_rank = target_rank
            best_res = res
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
