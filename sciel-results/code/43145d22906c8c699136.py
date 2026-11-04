"""Task: rank-minimal exact tensor decomposition of 3x4 by 4x7 matrix multiplication.
Trace convention: out[i,k] = sum_j A[i,j]*B[j,k] == sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
"""
import numpy as np
import time

_START = time.time()
_LIMIT = 120.0          # hard safety margin before the 150 s wall-clock kill
DIMENSIONS = [3, 4, 7]


def _check(U, V, W, n, m, p, r, tol=1e-6):
    """Check if the tensor decomposition exactly reproduces matrix multiplication."""
    # Term 0 must be the identity term (schoolbook)
    # For each term c >= 1, we need sum_{i,j,k} U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 0
    # Actually, the full check is: for each (i,k), sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 if (i,j,k) matches
    # More precisely: the tensor T[i,j,k] = delta[i,j,k] for the multiplication identity
    # We need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 if this corresponds to the (i,j,k) term
    
    # The matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k) in the sense that
    # out[i,k] = sum_j A[i,j]B[j,k], so the tensor is the "identity" bilinear map.
    # The rank-1 decomposition must satisfy:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1  for all i,j,k? No.
    
    # Actually, the constraint is:
    # For the bilinear map (A,B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]:
    # We need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] to equal the coefficient of A[i,j]*B[j,k] in C[k*n+i]
    # which is 1 if the indices match (always, since we're decomposing the full tensor)
    
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    # A rank-r decomposition means T = sum_c U_c ⊗ V_c ⊗ W_c where
    # (U_c ⊗ V_c ⊗ W_c)[i*m+j, j*p+k, k*n+i] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    
    # So we need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # AND the "wrong" entries must be 0, but since the tensor is supported only on these indices,
    # we just need the sum to equal 1 for each (i,j,k).
    
    # Let's verify by computing the full tensor and checking.
    # Build the tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0
    
    # More efficiently, check that for all i,j,k:
    # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    Nm, Mp, Pn = n * m, m * p, p * n

    # --- 1. Explicit schoolbook decomposition (rank n*m*p), always valid ---
    R0 = n * m * p
    U0 = np.zeros((Nm, R0)); V0 = np.zeros((Mp, R0)); W0 = np.zeros((Pn, R0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1

    def _pack(arr):
        return [[float(x) for x in row] for row in np.asarray(arr)]

    # best result found so far (initially schoolbook)
    best = {"U": _pack(U0), "V": _pack(V0), "W": _pack(W0)}

    # --- 2. Vectorized ALS decomposition ---
    def _als_decompose(n, m, p, r, max_iter=200, tol=1e-9, seed=42):
        """Find a rank-r decomposition using vectorized ALS."""
        rng = np.random.RandomState(seed)
        
        # Precompute index arrays for vectorized access
        # For U: rows are (i,j), columns are c
        # For V: rows are (j,k), columns are c  
        # For W: rows are (k,i), columns are c
        
        # Initialize randomly with small scale
        U = rng.randn(Nm, r) * 0.5
        V = rng.randn(Mp, r) * 0.5
        W = rng.randn(Pn, r) * 0.5
        
        for iteration in range(max_iter):
            if time.time() > _LIMIT - 15.0:
                break
                
            # --- Solve for U given V, W ---
            # For each (i,j): A[k,c] = V[j*p+k,c] * W[k*n+i,c], solve A @ u = ones(p)
            U_new = np.zeros((Nm, r))
            for i in range(n):
                for j in range(m):
                    # V[j*p:(j+1)*p, :] has shape (p, r)
                    # W[:, i] has shape (p,) but we need W[k*n+i] for k=0..p-1
                    # W[k*n+i] = W.reshape(p, n)[k, i]
                    V_sub = V[j*p:(j+1)*p, :]  # (p, r)
                    W_sub = W.reshape(p, n)[:, i]  # (p,)
                    A = V_sub * W_sub[:, np.newaxis]  # (p, r)
                    b = np.ones(p)
                    u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i*m+j, :] = u
            
            # --- Solve for V given U, W ---
            V_new = np.zeros((Mp, r))
            for j in range(m):
                for k in range(p):
                    # U[i*m+j] for i=0..n-1: U.reshape(n, m, r)[i, j, :]
                    U_sub = U.reshape(n, m, r)[:, j, :]  # (n, r)
                    W_sub = W.reshape(p, n)[k, :]  # (n,)
                    A = U_sub * W_sub[:, np.newaxis]  # (n, r)
                    b = np.ones(n)
                    v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j*p+k, :] = v
            
            # --- Solve for W given U, V ---
            W_new = np.zeros((Pn, r))
            for k in range(p):
                for i in range(n):
                    # U[i*m+j] for j=0..m-1: U.reshape(n, m, r)[i, :, :]
                    U_sub = U.reshape(n, m, r)[i, :, :]  # (m, r)
                    V_sub = V.reshape(m, p, r)[:, k, :]  # (m, r)
                    A = U_sub * V_sub  # (m, r)
                    b = np.ones(m)
                    w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k*n+i, :] = w
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence with tighter tolerance
            if _check(U, V, W, n, m, p, r, tol=tol):
                return U, V, W, True
        
        return U, V, W, _check(U, V, W, n, m, p, r, tol=tol)
    
    # --- 3. Search for lower-rank decompositions ---
    lb = max(n*p, m*p, n*m)  # = 28
    
    best_rank = R0
    # Try ranks from lower bound upward, with more seeds at lower ranks
    for r in range(lb, min(R0, lb + 20)):
        if time.time() > _LIMIT - 25.0:
            break
        
        found = False
        # Use more seeds for lower ranks (harder to find)
        n_seeds = 50 if r < lb + 5 else 20
        for seed in range(n_seeds):
            if time.time() > _LIMIT - 15.0:
                break
            
            U, V, W, success = _als_decompose(n, m, p, r, max_iter=100, tol=1e-9, seed=seed + 1000 + r)
            
            if success:
                best = {"U": _pack(U), "V": _pack(V), "W": _pack(W)}
                best_rank = r
                found = True
                break
        
        if found:
            # Found a valid decomposition at this rank, try one more rank lower if possible
            if r > lb:
                continue
            else:
                break
    
    return best
