"""Matrix multiplication 2,3,10 - search for rank-minimal decomposition."""
import numpy as np
import time
import random
import sys

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, tol=1e-6):
    """Verify tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                expected = 1.0
                if abs(val - expected) > tol:
                    return False
    return True

def _verify_fast(U, V, W, tol=1e-6):
    """Faster verification using numpy."""
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Reshape: U is (n*m, r), V is (m*p, r), W is (p*n, r)
    # For each c, the contribution is outer product of reshaped U[:,c], V[:,c], W[:,c]
    # U reshaped: (n, m, r), V reshaped: (m, p, r), W reshaped: (p, n, r)
    Ur = U.reshape(n, m, -1)
    Vr = V.reshape(m, p, -1)
    Wr = W.reshape(p, n, -1)
    # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # Use einsum: sum over c of Ur[i,j,c]*Vr[j,k,c]*Wr[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.allclose(T, 1.0, atol=tol)

def _try_rank(r, max_iter=500, seed=42):
    """Try to find rank-r decomposition using iterative refinement."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    best = None
    
    for trial in range(max_iter):
        # Initialize randomly
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        # Iterative refinement: fix two, solve for third
        converged = False
        for it in range(200):
            # Solve for W given U, V
            # For each (k,i): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all j
            # This is a linear system in W
            # For fixed (k,i): sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*V[j*p+k,c]) = 1
            # Define M[k*n+i, c] = sum_j U[i*m+j,c] * V[j*p+k,c]
            # Then sum_c M[k*n+i,c] * W[k*n+i,c] = 1
            # This is underdetermined (p*n equations, r unknowns per row... wait)
            # Actually each row of W is independent: for row (k,i), we need sum_c W[row,c]*M[row,c] = 1
            # So W[row,c] = M[row,c] / sum_c M[row,c]^2 (least squares)
            
            # Compute M: M[k*n+i, c] = sum_j U[i*m+j,c] * V[j*p+k,c]
            # U shape: (n*m, r), V shape: (m*p, r)
            # For (k,i,c): sum_j U[i*m+j, c] * V[j*p+k, c]
            # Let's compute this: for each c, M[:,:,c] = ?
            # U reshaped (n,m,r), V reshaped (m,p,r)
            Ur = U.reshape(n, m, r)
            Vr = V.reshape(m, p, r)
            # M[k,i,c] = sum_j Ur[i,j,c] * Vr[j,k,c]
            M = np.einsum('ijc,jkc->kic', Ur, Vr)  # (p, n, r)
            M = M.reshape(p * n, r)
            
            # For each row, W[row,:] should satisfy dot(W[row,:], M[row,:]) = 1
            # Least squares: W[row,:] = M[row,:] / (M[row,:]·M[row,:])
            norms_sq = np.sum(M * M, axis=1)
            norms_sq[norms_sq < 1e-12] = 1e-12
            W = M / norms_sq[:, None]
            
            # Similarly solve for U
            # For (i,j): sum_c U[i*m+j,c] * (sum_k V[j*p+k,c]*W[k*n+i,c]) = 1
            # N[i*m+j, c] = sum_k V[j*p+k,c] * W[k*n+i,c]
            Vr2 = V.reshape(m, p, r)
            Wr2 = W.reshape(p, n, r)
            N = np.einsum('jkc,kic->ijc', Vr2, Wr2)  # (n, m, r)
            N = N.reshape(n * m, r)
            norms_sq_u = np.sum(N * N, axis=1)
            norms_sq_u[norms_sq_u < 1e-12] = 1e-12
            U = N / norms_sq_u[:, None]
            
            # Solve for V
            # For (j,k): sum_c V[j*p+k,c] * (sum_i U[i*m+j,c]*W[k*n+i,c]) = 1
            Ur2 = U.reshape(n, m, r)
            Wr3 = W.reshape(p, n, r)
            P = np.einsum('ijc,kic->jkc', Ur2, Wr3)  # (m, p, r)
            P = P.reshape(m * p, r)
            norms_sq_v = np.sum(P * P, axis=1)
            norms_sq_v[norms_sq_v < 1e-12] = 1e-12
            V = P / norms_sq_v[:, None]
            
            # Check convergence
            if _verify_fast(U, V, W, tol=1e-8):
                converged = True
                break
        
        if converged:
            # Try to clean up to integers/half-integers
            best = (U, V, W)
            break
    
    return best

def _clean_to_rational(U, V, W, max_denom=20):
    """Try to round to rational numbers with small denominators."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    
    # Try rounding to multiples of 1/max_denom
    for denom in [1, 2, 3, 4, 5, 6, 8, 10, 12, max_denom]:
        scale = float(denom)
        Uc = np.round(U * scale) / scale
        Vc = np.round(V * scale) / scale
        Wc = np.round(W * scale) / scale
        if _verify_fast(Uc, Vc, Wc, tol=1e-9):
            return Uc, Vc, Wc
    
    # Try just rounding to nearest 0.5
    Uc = np.round(U * 2) / 2
    Vc = np.round(V * 2) / 2
    Wc = np.round(W * 2) / 2
    if _verify_fast(Uc, Vc, Wc, tol=1e-9):
        return Uc, Vc, Wc
    
    return None

def _structured_search(r, deadline):
    """Search with structure: exploit the small dimensions."""
    n, m, p = DIMENSIONS
    best = None
    
    for seed in range(100):
        if time.time() > deadline:
            break
        result = _try_rank(r, max_iter=50, seed=seed)
        if result is not None:
            U, V, W = result
            cleaned = _clean_to_rational(U, V, W)
            if cleaned is not None:
                Uc, Vc, Wc = cleaned
                if _verify_fast(Uc, Vc, Wc, tol=1e-9):
                    best = (Uc, Vc, Wc)
                    break
            else:
                # Use the floating point result
                best = (U, V, W)
                break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    # Try progressively lower ranks
    # Lower bound: max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # But for matrix mult, the tensor rank is at least max(n*m, m*p, p*n) / something
    # Actually the trivial lower bound is max(n*m, m*p, p*n) = 30
    # Let's try from 30 upward
    
    # First, try a few specific ranks with random search
    for r in range(30, min(schoolbook_rank, 55)):
        if time.time() > deadline - 30:
            break
        
        # Try structured search
        result = _structured_search(r, deadline - 30)
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = [[float(U[i, c]) for c in range(r)] for i in range(n * m)]
            V_list = [[float(V[i, c]) for c in range(r)] for i in range(m * p)]
            W_list = [[float(W[i, c]) for c in range(r)] for i in range(p * n)]
            
            # Final verification
            if _verify(U_list, V_list, W_list, tol=1e-6):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # If we didn't find anything better, return schoolbook
    if time.time() > deadline - 5:
        return _schoolbook()
    
    # Try a few more random attempts at lower ranks
    for r in range(30, 45):
        if time.time() > deadline - 5:
            break
        result = _try_rank(r, max_iter=200, seed=999 + r)
        if result is not None:
            U, V, W = result
            U_list = [[float(U[i, c]) for c in range(r)] for i in range(n * m)]
            V_list = [[float(V[i, c]) for c in range(r)] for i in range(m * p)]
            W_list = [[float(W[i, c]) for c in range(r)] for i in range(p * n)]
            if _verify(U_list, V_list, W_list, tol=1e-6):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return _schoolbook()
# EVOLVE-BLOCK-END
