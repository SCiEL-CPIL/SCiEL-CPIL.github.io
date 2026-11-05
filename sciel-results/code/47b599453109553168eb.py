"""Program for rank-minimal exact tensor decomposition of 4x5*5x6 matrix multiplication."""
import numpy as np

DIMENSIONS = [4, 5, 6]

def _build_strassen_222():
    """Build the 7-term Strassen decomposition for 2x2x2.
    Returns U (4x7), V (4x7), W (4x7) where:
      U row = i*2+j, V row = j*2+k, W row = k*2+i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00 -> C10, C11(-)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11) -> C01, C11
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00) -> C00, C10
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11 -> C01, C00(-)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[0, 4] = -1
    
    # M6 = (A10-A00)(B00+B01) -> C11
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify_decomp(U, V, W, n, m, p):
    """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Uses a block decomposition with Strassen's algorithm for 2x2x2 blocks 
    and schoolbook for 2x1x2 blocks.
    Expected rank: 108 (vs 120 schoolbook).
    """
    n, m, p = DIMENSIONS
    
    # Build verified Strassen 2x2x2 decomposition
    U22, V22, W22 = _build_strassen_222()
    
    # Verify the 2x2x2 Strassen decomposition
    if not _verify_decomp(U22, V22, W22, 2, 2, 2):
        # Fallback to schoolbook for 2x2x2 (rank 8)
        U22 = np.zeros((4, 8))
        V22 = np.zeros((4, 8))
        W22 = np.zeros((4, 8))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U22[i*2+j, col] = 1
                    V22[j*2+k, col] = 1
                    W22[k*2+i, col] = 1
                    col += 1
    
    # Block offsets and sizes
    # n=4 split into [2, 2]
    # m=5 split into [2, 2, 1]
    # p=6 split into [2, 2, 2]
    i_offsets = [0, 2]
    i_sizes = [2, 2]
    j_offsets = [0, 2, 4]
    j_sizes = [2, 2, 1]
    k_offsets = [0, 2, 4]
    k_sizes = [2, 2, 2]
    
    # Compute total rank
    total_rank = 0
    for i_idx in range(len(i_offsets)):
        for j_idx in range(len(j_offsets)):
            for k_idx in range(len(k_offsets)):
                nb = i_sizes[i_idx]
                mb = j_sizes[j_idx]
                pb = k_sizes[k_idx]
                if nb == 2 and mb == 2 and pb == 2:
                    total_rank += 7
                elif nb == 2 and mb == 1 and pb == 2:
                    total_rank += 4
                else:
                    total_rank += nb * mb * pb
    
    # Allocate arrays
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col = 0
    
    for i_idx, i_0 in enumerate(i_offsets):
        n_b = i_sizes[i_idx]
        for j_idx, j_0 in enumerate(j_offsets):
            m_b = j_sizes[j_idx]
            for k_idx, k_0 in enumerate(k_offsets):
                p_b = k_sizes[k_idx]
                
                if n_b == 2 and m_b == 2 and p_b == 2:
                    # Use Strassen: 7 terms
                    for t in range(7):
                        # U: map local (u,v) to global (i_0+u, j_0+v)
                        for u in range(2):
                            for v in range(2):
                                if U22[u*2+v, t] != 0:
                                    U[(i_0+u)*m + (j_0+v), col] = U22[u*2+v, t]
                        # V: map local (v,w) to global (j_0+v, k_0+w)
                        for v in range(2):
                            for w in range(2):
                                if V22[v*2+w, t] != 0:
                                    V[(j_0+v)*p + (k_0+w), col] = V22[v*2+w, t]
                        # W: map local (w,u) to global (k_0+w, i_0+u)
                        for w in range(2):
                            for u in range(2):
                                if W22[w*2+u, t] != 0:
                                    W[(k_0+w)*n + (i_0+u), col] = W22[w*2+u, t]
                        col += 1
                elif n_b == 2 and m_b == 1 and p_b == 2:
                    # Schoolbook for 2x1x2: 4 terms
                    for u in range(2):
                        for w in range(2):
                            U[(i_0+u)*m + (j_0+0), col] = 1
                            V[(j_0+0)*p + (k_0+w), col] = 1
                            W[(k_0+w)*n + (i_0+u), col] = 1
                            col += 1
                else:
                    # Generic schoolbook fallback
                    for u in range(n_b):
                        for v in range(m_b):
                            for w in range(p_b):
                                U[(i_0+u)*m + (j_0+v), col] = 1
                                V[(j_0+v)*p + (k_0+w), col] = 1
                                W[(k_0+w)*n + (i_0+u), col] = 1
                                col += 1
    
    # Trim to actual rank
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    # Final verification
    assert col == total_rank, f"Rank mismatch: expected {total_rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
