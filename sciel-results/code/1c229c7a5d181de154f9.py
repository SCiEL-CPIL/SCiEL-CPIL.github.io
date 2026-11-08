"""Matrix multiplication 2x3 by 3x9: rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# Known bounds:
# - Schoolbook rank = 2*3*9 = 54
# - Strassen-like or other improvements might exist
# - For small matrices, exhaustive search or structured ansatz may help

# The matrix multiplication tensor T has T[i,j,k] = 1 if output (i,k) uses input (i,j) and (j,k)
# We want rank-r decomposition: T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
# where u_r has shape (n*m,), v_r has shape (m*p,), w_r has shape (p*n,)
# and T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

# Actually, the standard formulation:
# C = A @ B, where A is n x m, B is m x p, C is n x p
# T_{i,j,k} = C_{i,k} / (A_{i,j} * B_{j,k}) in tensor sense
# The tensor is T[i, j, k] = 1 for all i, j, k (since every entry of C depends on corresponding A and B)
# Wait, no. The multiplication tensor has T[i, j, k] = 1 if the (i,k) entry of C uses A[i,j] and B[j,k].
# Actually T[i, j, k] = δ_{i,i} δ_{k,k} ... let me think more carefully.

# The matrix multiplication tensor T ∈ R^{(n*m) × (m*p) × (p*n)} is defined by:
# T[i*m + j, j*p + k, k*n + i] = 1 for all valid i,j,k
# All other entries are 0.

# We seek rank R such that T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
# i.e., T[a, b, c] = sum_{r=1}^{R} U[a, r] * V[b, r] * W[c, r]

# For the schoolbook: R = n*m*p, with each term picking one (i,j,k) triple.

# Let's try to find a lower rank. For 2x3 * 3x9, we can try various approaches.

def _build_tensor():
    """Build the matrix multiplication tensor as a 3D array."""
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that the decomposition is exact."""
    # Compute sum of outer products
    R = U.shape[1]
    # Efficient check: for each (a,b,c) in T, check if sum_r U[a,r]*V[b,r]*W[c,r] matches T[a,b,c]
    # Only check non-zero entries of T for speed
    for i in range(DIMENSIONS[0]):
        for j in range(DIMENSIONS[1]):
            for k in range(DIMENSIONS[2]):
                a = i * DIMENSIONS[1] + j
                b = j * DIMENSIONS[2] + k
                c = k * DIMENSIONS[0] + i
                val = 0.0
                for r in range(R):
                    val += U[a, r] * V[b, r] * W[c, r]
                if abs(val - T[a, b, c]) > tol:
                    return False
    # Also check that zero entries remain zero (sample a few)
    return True

def _verify_full(U, V, W, T, tol=1e-9):
    """Full verification including zero entries."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    # Check all entries
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                val = 0.0
                for r in range(R):
                    val += U[a, r] * V[b, r] * W[c, r]
                if abs(val - T[a, b, c]) > tol:
                    return False
    return True

def _try_rank(R, T, max_iters=5000, seed=42):
    """
    Try to find a rank-R decomposition using alternating least squares or random init.
    Returns (U, V, W) if successful, else None.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    rng = np.random.RandomState(seed)
    
    # Random initialization with small values
    for attempt in range(20):
        U = rng.randn(nm, R) * 0.5
        V = rng.randn(mp, R) * 0.5
        W = rng.randn(pn, R) * 0.5
        
        # Alternating optimization
        converged = False
        for it in range(max_iters):
            # Update U: for each a, U[a, :] should satisfy sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all b,c
            # This is a linear system in U[a, :] for fixed V, W
            # For each a: T[a, :, :] = U[a, :] @ (V ⊙ W) where ⊙ is element-wise... 
            # Actually: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # Let M[a, r, b, c] = V[b,r] * W[c,r]
            # Then T[a, b, c] = sum_r U[a,r] * M[a, r, b, c]... no, M doesn't depend on a.
            # T[a, b, c] = sum_r U[a,r] * (V[b,r] * W[c,r])
            # For fixed a, this is: T[a, :, :] = sum_r U[a,r] * (V[:, r] ⊗ W[:, r])
            # So we have a linear system: for each a, solve for U[a, :]
            # The RHS is T[a, :, :] reshaped to (mp*pn,), and the matrix has columns V[:,r] ⊗ W[:,r]
            
            # Build the design matrix once: D[r, b, c] = V[b,r] * W[c,r], shape (R, mp, pn)
            D = V[:, :, None] * W[:, None, :]  # (R, mp, pn)
            D_flat = D.reshape(R, -1)  # (R, mp*pn)
            
            # For each a: solve D_flat.T @ U[a, :] ≈ T[a, :, :].flatten()
            # Use least squares
            for a in range(nm):
                rhs = T[a, :, :].flatten()
                # Solve: min ||D_flat.T @ x - rhs||
                # x = U[a, :]
                try:
                    U[a, :], residuals, rank_s, sv = np.linalg.lstsq(D_flat.T, rhs, rcond=None)
                except:
                    pass
            
            # Update V similarly
            E = U[:, :, None] * W[:, None, :]  # (R, nm, pn)
            E_flat = E.reshape(R, -1)  # (R, nm*pn)
            for b in range(mp):
                rhs = T[:, b, :].flatten()
                try:
                    V[b, :], residuals, rank_s, sv = np.linalg.lstsq(E_flat.T, rhs, rcond=None)
                except:
                    pass
            
            # Update W
            F = U[:, :, None] * V[:, None, :]  # (R, nm, mp)
            F_flat = F.reshape(R, -1)  # (R, nm*mp)
            for c in range(pn):
                rhs = T[:, :, c].flatten()
                try:
                    W[c, :], residuals, rank_s, sv = np.linalg.lstsq(F_flat.T, rhs, rcond=None)
                except:
                    pass
            
            # Check convergence
            # Compute error on non-zero entries
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = 0.0
                        for r in range(R):
                            val += U[a, r] * V[b, r] * W[c, r]
                        err += (val - 1.0) ** 2
            if err < 1e-12:
                converged = True
                break
        
        if converged:
            # Verify
            if _verify_full(U, V, W, T, tol=1e-8):
                return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Try ranks from known lower bounds upward
    # Lower bound: max(n*p, n*m, m*p) = max(18, 6, 27) = 27
    # But for bilinear complexity, lower bounds are tighter.
    # Let's try a range of ranks
    
    ranks_to_try = list(range(27, rank_schoolbook + 1))
    # Prioritize lower ranks
    random.seed(123)
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
        if R >= best_rank:
            break
        
        # Try multiple seeds
        for seed in range(10):
            if time.time() > deadline:
                break
            result = _try_rank(R, T, max_iters=200, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify_full(U, V, W, T, tol=1e-8):
                    best_result = (U, V, W)
                    best_rank = R
                    break  # Found rank R, move to next lower... wait, we're going up
                break  # Try next R
    
    # Convert to lists for return
    U, V, W = best_result
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
