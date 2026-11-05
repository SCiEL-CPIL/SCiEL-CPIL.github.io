import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if U,V,W exactly decompose the tensor (within tol)."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _construct_rank44_decomposition():
    """
    Construct a rank-44 decomposition using Strassen for 2x2x2 blocks
    and schoolbook for 1x2x2 blocks.
    Returns U, V, W as numpy arrays of shape (n*m, 44), (m*p, 44), (p*n, 44).
    """
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(rA, cA, rB, cB, rC):
        """Add Strassen decomposition for 2x2x2 block."""
        nonlocal col
        # M1 = (A11+A22)(B11+B22)
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rC+0), col] = 1
        W[(cB+1)*n + (rC+1), col] = 1
        col += 1
        
        # M2 = (A21+A22)B11
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rC+1), col] = 1
        W[(cB+1)*n + (rC+1), col] = -1
        col += 1
        
        # M3 = A11(B12-B22)
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = -1
        W[(cB+1)*n + (rC+0), col] = 1
        W[(cB+1)*n + (rC+1), col] = 1
        col += 1
        
        # M4 = A22(B21-B11)
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = -1
        W[(cB+0)*n + (rC+0), col] = 1
        W[(cB+0)*n + (rC+1), col] = 1
        col += 1
        
        # M5 = (A11+A12)B22
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rC+0), col] = 1
        W[(cB+0)*n + (rC+0), col] = -1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+0), col] = -1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rC+1), col] = 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[(rA+0)*m + (cA+1), col] = 1
        U[(rA+1)*m + (cA+1), col] = -1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rC+0), col] = 1
        col += 1

    def add_schoolbook_1x2x2(rA, cA, rB, cB, rC):
        """Add schoolbook for 1x2x2 block."""
        nonlocal col
        # C0 = A0*B00 + A1*B10
        U[(rA)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rC), col] = 1
        col += 1
        
        # A1*B10 -> C0
        U[(rA)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rC), col]  = 1
        col += 1
        
        # A0*B01 -> C1
        U[(rA)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rC), col] = 1
        col += 1
        
        # A1*B11 -> C1
        U[(rA)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rC), col] = 1
        col += 1

    # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    add_strassen(0, 0, 0, 0, 0)
    add_strassen(0, 2, 2, 0, 0)
    
    # C[0:2, 2:4] = A[0:2, 0:2]*B[0:2, 2:4] + A[0:2, 2:4]*B[2:4, 2:4]
    add_strassen(0, 0, 0, 2, 0)
    add_strassen(0, 2, 2, 2, 0)
    
    # C[2, 0:2] = A[2, 0:2]*B[0:2, 0:2] + A[2, 2:4]*B[2:4, 0:2]
    add_schoolbook_1x2x2(2, 0, 0, 0, 2)
    add_schoolbook_1x2x2(2, 2, 2, 0, 2)
    
    # C[2, 2:4] = A[2, 0:2]*B[0:2, 2:4] + A[2, 2:4]*B[2:4, 2:4]
    add_schoolbook_1x2x2(2, 0, 0, 2, 2)
    add_schoolbook_1x2x2(2, 2, 2, 2, 2)
    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, max_reduction=5, deadline=None):
    """
    Try to reduce rank by finding linear dependencies among tensor terms.
    """
    if deadline is not None and time.time() > deadline:
        return U, V, W
    
    rank = U.shape[1]
    nmp = n * m * p
    
    # Build the matrix of all rank-1 tensors flattened
    M = np.zeros((nmp, rank))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T = np.einsum('ij,jk,ki->ijk', u, v, w)
        M[:, r] = T.flatten()
    
    # Check rank of M
    try:
        U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return U, V, W
    
    tol = 1e-10
    num_rank = np.sum(S > tol)
    
    if num_rank >= rank:
        return U, V, W
    
    # Try eliminating each term
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    for r_elim in range(rank):
        if deadline is not None and time.time() > deadline:
            break
        
        M_other = np.delete(M, r_elim, axis=1)
        
        try:
            x, residuals, rank_other, _ = np.linalg.lstsq(M_other, M[:, r_elim], rcond=1e-10)
        except np.linalg.LinAlgError:
            continue
        
        residual = np.linalg.norm(M_other @ x - M[:, r_elim])
        if residual < 1e-8:
            # Term r_elim is a linear combination of others
            # We can eliminate it
            new_U = np.delete(U, r_elim, axis=1)
            new_V = np.delete(V, r_elim, axis=1)
            new_W = np.delete(W, r_elim, axis=1)
            
            if _verify_decomposition(new_U, new_V, new_W, n, m, p, tol=1e-6):
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = new_U.shape[1]
                U, V, W = new_U, new_V, new_W
                rank = best_rank
                # Rebuild M
                M = np.zeros((nmp, rank))
                for r in range(rank):
                    u = U[:, r].reshape(n, m)
                    v = V[:, r].reshape(m, p)
                    w = W[:, r].reshape(p, n)
                    T = np.einsum('ij,jk,ki->ijk', u, v, w)
                    M[:, r] = T.flatten()
    
    return best_U, best_V, best_W

def _construct():
    """Main construction with rank reduction attempt."""
    n, m, p = 3, 4, 4
    deadline = time.time() + 100  # Safety margin
    
    # Build rank-44 decomposition
    U, V, W = _construct_rank44_decomposition()
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        # Fallback to schoolbook
        rank = n * m * p
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Try to reduce rank
    if time.time() < deadline:
        U_red, V_red, W_red = _try_reduce_rank(U, V, W, n, m, p, max_reduction=5, deadline=deadline)
        if U_red.shape[1] < U.shape[1]:
            U, V, W = U_red, V_red, W_red
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    return _construct()
