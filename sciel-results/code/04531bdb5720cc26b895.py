"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(recon, T, atol=1e-9):
        return False
    return True

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _als_decompose(T, rank, max_iter=200, tol=1e-10, seed=42):
    """ALS decomposition of tensor T into rank-many rank-1 tensors."""
    rng = np.random.RandomState(seed)
    nA, nB, nC = T.shape
    U = rng.randn(nA, rank) * 0.1
    V = rng.randn(nB, rank) * 0.1
    W = rng.randn(nC, rank) * 0.1
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Update U: for each a, minimize ||T[a,:,:] - sum_r U[a,r]*V[:,r]⊗W[:,r]||
        # Residual R = T - einsum('ar,br,cr->abc', U, V, W)
        # For fixed V,W: U[a,r] = <T[a,:,:], V[:,r]⊗W[:,r]> / ||V[:,r]⊗W[:,r]||^2
        VW = np.einsum('br,cr->bcr', V, W)  # shape (nB, nC, rank)
        denom_U = np.einsum('bcr,bcr->r', VW, VW)
        for r in range(rank):
            if denom_U[r] > 1e-15:
                U[:, r] = np.einsum('abc,ab->c', T, VW[:,:,r]) / denom_U[r]
        
        # Update V
        UW = np.einsum('ar,cr->acr', U, W)  # shape (nA, nC, rank)
        denom_V = np.einsum('acr,acr->r', UW, UW)
        for r in range(rank):
            if denom_V[r] > 1e-15:
                V[:, r] = np.einsum('abc,ac->c', T, UW[:,:,r]) / denom_V[r]
        
        # Update W
        UV = np.einsum('ar,br->abr', U, V)  # shape (nA, nB, rank)
        denom_W = np.einsum('abr,abr->r', UV, UV)
        for r in range(rank):
            if denom_W[r] > 1e-15:
                W[:, r] = np.einsum('abc,ab->c', T, UV[:,:,r]) / denom_W[r]
        
        # Compute error
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(recon - T) / (np.linalg.norm(T) + 1e-15)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < tol:
            break
    
    return best_U, best_V, best_W, best_err

def _round_and_verify(U, V, W, n, m, p, max_denom=4):
    """Try rounding to rationals with small denominators and verify."""
    T = _build_tensor(n, m, p)
    for denom in [1, 2, 3, 4, max_denom]:
        U_r = np.round(U * denom) / denom
        V_r = np.round(V * denom) / denom
        W_r = np.round(W * denom) / denom
        recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(recon, T, atol=1e-9):
            return U_r, V_r, W_r, denom
    return None, None, None, None

def _build_rank44_decomposition(n, m, p):
    """Build a rank-44 decomposition for 2x3 x 3x8 matrix multiplication."""
    n, m, p = 2, 3, 8
    rank = 44
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for s in range(4):
        base_rank = s * 11
        k0 = 2 * s
        k1 = 2 * s + 1
        
        # M1
        c = base_rank + 0
        U[0*3+0, c] = 1.0
        U[1*3+1, c] = 1.0
        V[0*8+k0, c] = 1.0
        V[1*8+k1, c] = 1.0
        W[k0*2+0, c] = 1.0
        W[k1*2+1, c] = 1.0
        
        # M2
        c = base_rank + 1
        U[1*3+0, c] = 1.0
        U[1*3+1, c] = 1.0
        V[0*8+k0, c] = 1.0
        W[k0*2+1, c] = 1.0
        
        # M3
        c = base_rank + 2
        U[0*3+0, c] = 1.0
        V[0*8+k1, c] = 1.0
        V[1*8+k1, c] = -1.0
        W[k1*2+0, c] = 1.0
        
        # M4
        c = base_rank + 3
        U[1*3+1, c] = 1.0
        V[1*8+k0, c] = 1.0
        V[0*8+k0, c] = -1.0
        W[k0*2+0, c] = 1.0
        W[k0*2+1, c] = 1.0
        
        # M5
        c = base_rank + 4
        U[0*3+0, c] = 1.0
        U[0*3+1, c] = 1.0
        V[1*8+k1, c] = 1.0
        W[k0*2+0, c] = -1.0
        W[k1*2+0, c] = 1.0
        
        # M6
        c = base_rank + 5
        U[1*3+0, c] = 1.0
        U[0*3+0, c] = -1.0
        V[0*8+k0, c] = 1.0
        V[0*8+k1, c] = 1.0
        W[k1*2+1, c] = 1.0
        
        # M7
        c = base_rank + 6
        U[0*3+1, c] = 1.0
        U[1*3+1, c] = -1.0
        V[1*8+k0, c] = 1.0
        V[1*8+k1, c] = 1.0
        W[k0*2+0, c] = 1.0

        # Schoolbook for j=2
        c = base_rank + 7
        U[0*3+2, c] = 1.0
        V[2*8+k0, c] = 1.0
        W[k0*2+0, c] = 1.0
        
        c = base_rank + 8
        U[0*3+2, c] = 1.0
        V[2*8+k1, c] = 1.0
        W[k1*2+0, c] = 1.0
        
        c = base_rank + 9
        U[1*3+2, c] = 1.0
        V[2*8+k0, c] = 1.0
        W[k0*2+1, c] = 1.0
        
        c = base_rank + 10
        U[1*3+2, c] = 1.0
        V[2*8+k1, c] = 1.0
        W[k1*2+1, c] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    """
    Find a low-rank decomposition of 2x3*3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # First, get the rank-44 baseline
    result44 = _build_rank44_decomposition(n, m, p)
    if _verify(result44["U"], result44["V"], result44["W"], n, m, p):
        best_result = result44
        best_rank = 44
    else:
        # Fallback schoolbook
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
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
        return best_result
    
    # Try to find lower rank via ALS
    T = _build_tensor(n, m, p)
    
    for target_rank in range(43, 30, -1):
        if time.time() > deadline:
            break
        
        found = False
        for seed in range(5):
            if time.time() > deadline:
                break
            
            U_a, V_a, W_a, err = _als_decompose(T, target_rank, max_iter=100, tol=1e-12, seed=seed)
            
            if err < 1e-8:
                # Try rounding
                U_r, V_r, W_r, denom = _round_and_verify(U_a, V_a, W_a, n, m, p)
                if U_r is not None:
                    result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    if _verify(result["U"], result["V"], result["W"], n, m, p):
                        best_result = result
                        best_rank = target_rank
                        found = True
                        break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
