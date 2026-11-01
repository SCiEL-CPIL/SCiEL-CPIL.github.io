"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.

The schoolbook rank is n*m*p = 2*2*5 = 20.
We attempt to find a lower rank via:
  1. Algebraic construction using the specific structure of small matrix multiplication.
  2. Integer/half-integer search with constraint solving.
  3. ALS with rationalization as a fallback.

For 2x2 * 2x5, the known lower bounds and constructions suggest rank 20 may be minimal,
but we try to verify and also attempt rank 19, 18, ... down to a few values.

However, a well-known result: for multiplying an (n x m) by (m x p) matrix, the
schoolbook gives n*m*p. For small cases, sometimes one can save a term.
For 2x2 * 2x5: the tensor has 20 non-zero entries. The rank of the matrix multiplication
tensor is at least the border rank, which for these small sizes often equals the number
of terms.

Given the difficulty, we'll focus on:
  - Providing the schoolbook decomposition (rank 20) as a guaranteed valid answer.
  - Attempting a quick search for rank < 20 using structured initialization and
    exact rational verification.

The key insight for potential improvement:
  In matrix multiplication C = A*B where A is 2x2, B is 2x5, C is 2x5.
  C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
  C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
  C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
  C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3]
  C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
  C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
  ... etc.

  The standard decomposition uses one term per (i,j,k) triple: 2*2*5=20 terms.
  
  Can we share terms? For example, if we could express A[0,0]*B[0,0] + A[0,1]*B[1,0]
  using a shared factor... but that's already one term per (i,j,k).
  
  Actually, the standard decomposition IS one term per (i,j,k), so rank 20.
  
  To get rank < 20, we'd need to combine multiple (i,j,k) contributions into fewer
  rank-1 tensors. This is generally not possible for matrix multiplication tensors
  of these sizes without specific algebraic identities (like Strassen for 2x2*2x2).
  
  For 2x2 * 2x5, I'm not aware of a known algorithm with rank < 20.
  
  Therefore, the best we can do is rank 20. But let's still attempt a search
  for rank 19-18 to be thorough, and fall back to 20.
"""

import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 2, 5]


def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1
    return T


def _check_decomposition(U, V, W, T):
    """Check if sum_r U_u,r * V_v,r * W_w,r == T_u,v,w for all u,v,w."""
    R = U.shape[1]
    result = np.zeros_like(T, dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return np.allclose(result, T, atol=1e-9)


def _schoolbook_decomposition():
    """Return the standard schoolbook decomposition with rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u, col] = 1
                V[v, col] = 1
                W[w, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()


def _try_small_rank(R, T, deadline):
    """
    Attempt to find a rank-R decomposition using structured random initialization
    and ALS, then check for exact rational solution.
    """
    n, m, p = DIMENSIONS
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    # Precompute the constraint matrices for ALS
    # For U: given V, W, solve for U
    # T[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
    
    for seed in [42, 123, 7, 999, 2024, 555, 777, 1111, 2222, 3333]:
        if time.time() > deadline:
            break
        np.random.seed(seed)
        
        U = np.random.randn(U_rows, R) * 0.5
        V = np.random.randn(V_rows, R) * 0.5
        W = np.random.randn(W_rows, R) * 0.5
        
        prev_err = np.inf
        for it in range(100):
            if time.time() > deadline:
                break
            
            # Update U
            # For each u: sum_r U[u,r] * (V[v,r]*W[w,r]) = T[u,v,w] for all v,w
            # Build matrix M_u of shape (V_rows*W_rows, R) where M_u[(v,w), r] = V[v,r]*W[w,r]
            VW = np.einsum('vr,wr->vwr', V, W)  # (V_rows, W_rows, R)
            VW_flat = VW.reshape(-1, R)  # (V_rows*W_rows, R)
            
            U_new = np.zeros_like(U)
            for u in range(U_rows):
                b = T[u].reshape(-1).astype(np.float64)
                sol, _, _, _ = np.linalg.lstsq(VW_flat, b, rcond=None)
                U_new[u] = sol
            U = U_new
            
            # Update V
            UW = np.einsum('ur,wr->uwr', U, W)  # (U_rows, W_rows, R)
            UW_flat = UW.reshape(-1, R)  # (U_rows*W_rows, R)
            
            V_new = np.zeros_like(V)
            for v in range(V_rows):
                b = T[:, v, :].reshape(-1).astype(np.float64)
                sol, _, _, _ = np.linalg.lstsq(UW_flat, b, rcond=None)
                V_new[v] = sol
            V = V_new
            
            # Update W
            UV = np.einsum('ur,vr->uvr', U, V)  # (U_rows, V_rows, R)
            UV_flat = UV.reshape(-1, R)  # (U_rows*V_rows, R)
            
            W_new = np.zeros_like(W)
            for w in range(W_rows):
                b = T[:, :, w].reshape(-1).astype(np.float64)
                sol, _, _, _ = np.linalg.lstsq(UV_flat, b, rcond=None)
                W_new[w] = sol
            W = W_new
            
            # Compute error
            err = 0.0
            for r in range(R):
                err += np.sum((np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r]) - T.astype(np.float64))**2)
            err = np.sqrt(err)
            
            if abs(prev_err - err) < 1e-12:
                break
            prev_err = err
        
        # Check for exact solution
        if _check_decomposition(U, V, W, T):
            # Try to rationalize to half-integers
            for max_denom in [2, 4, 8]:
                U_s = np.round(U * max_denom) / max_denom
                V_s = np.round(V * max_denom) / max_denom
                W_s = np.round(W * max_denom) / max_denom
                if _check_decomposition(U_s, V_s, W_s, T):
                    return U_s.tolist(), V_s.tolist(), W_s.tolist()
            # Return as-is if close enough
            return U.tolist(), V.tolist(), W.tolist()
    
    return None


def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    T = _build_target()
    
    # Try ranks from 19 down to 15 (quick attempts)
    for R in range(19, 14, -1):
        if time.time() > deadline:
            break
        result = _try_small_rank(R, T, deadline)
        if result is not None:
            U_l, V_l, W_l = result
            return {"U": U_l, "V": V_l, "W": W_l}
    
    # Fall back to schoolbook (rank 20)
    U_l, V_l, W_l = _schoolbook_decomposition()
    return {"U": U_l, "V": V_l, "W": W_l}
