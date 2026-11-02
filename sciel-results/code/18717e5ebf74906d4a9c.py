"""Matrix multiplication 3,3,7 rank minimization."""
import time
import itertools
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [3, 3, 7]

# Known best rank for 3x3x7 is 23 (from literature: Strassen-like constructions)
# We attempt to find a rank-23 decomposition via linear programming.

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i, j, k] = 1 if the 
    output element (i,k) depends on input A[i,j] * B[j,k].
    Actually, the tensor T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k].
    We reshape into a 2D matrix for linear algebra."""
    # The multiplication tensor has entries T[i, j, k] = 1 for all i,j,k
    # We want to decompose: C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[r]... 
    # No, standard form: C[i,k] = sum_r A[i,j] U[j,r] * B[j,k] V[r,k]... 
    # Let me use the standard tensor decomposition form.
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r] where T[i,j,k]=1 for all i,j,k
    # This is the all-ones tensor of shape (n*m, m*p, p*n) mapped appropriately.
    # Actually for matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: T[i, j, k] where i is row of A (and C), j is col of A/row of B, k is col of B (and C)
    # T[i,j,k] = 1 for all valid i,j,k
    # We flatten: index a = i*m+j (size n*m), index b = j*p+k (size m*p), index c = k*n+i (size p*n)
    # Then T[a,b,c] = 1 if a,b,c are consistent (same j)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _try_rank_r(T, r, n, m, p, timeout=100):
    """Try to find rank-r decomposition via random initialization + least squares.
    Returns (U, V, W) as lists of lists, or None if failed."""
    start = time.time()
    nm = n * m
    mp = m * p
    pn = p * n
    
    # The decomposition: T[a,b,c] = sum_{r=1}^{R} U[a,r] * V[b,r] * W[c,r]
    # This is a 3-way tensor decomposition (CP decomposition).
    # We can try alternating least squares or random init + refinement.
    
    # Try multiple random initializations
    best_err = np.inf
    best_factors = None
    
    n_restarts = 50
    for attempt in range(n_restarts):
        if time.time() - start > timeout - 5:
            break
        
        # Random initialization with small values
        np.random.seed(attempt * 12345 + 42)
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        
        # Alternating least squares
        for it in range(200):
            if time.time() - start > timeout - 5:
                break
            
            # Update U: for each a, T[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r])
            # T[a,:,:] is a vector of length mp*pn
            # Let M_r = V[:,r] * W[:,r] (element-wise), length mp*pn
            # T_flat[a] = U[a,:] @ M where M[r, (b,c)] = V[b,r]*W[c,r]
            # Actually: T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed a: T[a, b, c] over b,c = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])[b,c]
            # So define M[r, idx] where idx = b*pn + c, M[r, b*pn+c] = V[b,r]*W[c,r]
            # Then T_flat[a, idx] = sum_r U[a,r] M[r, idx]
            # So T_flat = U @ M where M is (r, mp*pn)
            
            # Build M
            M = np.zeros((r, mp * pn))
            for b in range(mp):
                for c in range(pn):
                    for rr in range(r):
                        M[rr, b * pn + c] = V[b, rr] * W[c, rr]
            
            T_flat = T.reshape(nm, mp * pn)
            
            # Solve U @ M^T = T_flat^T  =>  M U^T = T_flat^T
            # U^T = (M^T M)^{-1} M^T T_flat^T
            # Use lstsq
            try:
                U_new, _, _, _ = np.linalg.lstsq(M, T_flat.T, rcond=None)
                U = U_new.T  # (nm, r)
            except:
                pass
            
            # Update V: for each b, T[:, b, :] = sum_r V[b,r] * (U[:,r] ⊗ W[:,r])
            # T[a, b, c] = sum_r V[b,r] U[a,r] W[c,r]
            # For fixed b: T[a, c] over a,c = sum_r V[b,r] * (U[:,r] * W[:,r])[a,c]
            # N[r, a*pn+c] = U[a,r] * W[c,r]
            N = np.zeros((r, nm * pn))
            for a in range(nm):
                for c in range(pn):
                    for rr in range(r):
                        N[rr, a * pn + c] = U[a, rr] * W[c, rr]
            
            T_flat2 = T.reshape(nm, mp, pn).transpose(1, 0, 2).reshape(mp, nm * pn)
            
            try:
                V_new, _, _, _ = np.linalg.lstsq(N, T_flat2.T, rcond=None)
                V = V_new.T  # (mp, r)
            except:
                pass
            
            # Update W: for each c, T[:, :, c] = sum_r W[c,r] * (U[:,r] ⊗ V[:,r])
            # T[a, b, c] = sum_r W[c,r] U[a,r] V[b,r]
            # For fixed c: T[a, b] = sum_r W[c,r] * (U[:,r] * V[:,r])[a,b]
            P_mat = np.zeros((r, nm * mp))
            for a in range(nm):
                for b in range(mp):
                    for rr in range(r):
                        P_mat[rr, a * mp + b] = U[a, rr] * V[b, rr]
            
            T_flat3 = T.reshape(nm, mp, pn).transpose(2, 0, 1).reshape(pn, nm * mp)
            
            try:
                W_new, _, _, _ = np.linalg.lstsq(P_mat, T_flat3.T, rcond=None)
                W = W_new.T  # (pn, r)
            except:
                pass
            
            # Check convergence
            if it % 10 == 0:
                err = np.sum((T - np.einsum('ar,br,cr->abc', U, V, W))**2)
                if err < best_err:
                    best_err = err
                    best_factors = (U.copy(), V.copy(), W.copy())
        
        # Final error check
        err = np.sum((T - np.einsum('ar,br,cr->abc', U, V, W))**2)
        if err < best_err:
            best_err = err
            best_factors = (U.copy(), V.copy(), W.copy())
        
        if best_err < 1e-10:
            break
    
    if best_factors is not None and best_err < 1e-8:
        U, V, W = best_factors
        # Convert to list of lists with rounding to half-integers
        U_list = [[float(round(x * 2) / 2) for x in row] for row in U]
        V_list = [[float(round(x * 2) / 2) for x in row] for row in V]
        W_list = [[float(round(x * 2) / 2) for x in row] for row in W]
        
        # Verify
        U_arr = np.array(U_list, dtype=np.float64)
        V_arr = np.array(V_list, dtype=np.float64)
        W_arr = np.array(W_list, dtype=np.float64)
        recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        if np.max(np.abs(recon - T)) < 1e-9:
            return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140  # 10s safety margin
    
    # Try ranks from 23 upward (known lower bound for 3x3x7 is around 23)
    for r in range(23, 64):
        if time.time() > deadline - 30:
            break
        
        result = _try_rank_r(_build_tensor(n, m, p), r, n, m, p, timeout=deadline - time.time() - 5)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 63
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
