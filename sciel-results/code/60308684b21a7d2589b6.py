"""Rewritten program for 3x5x5 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_222():
    """
    Return (U, V, W) for 2x2x2 Strassen multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook multiplication."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _build_rank71():
    """
    Build a rank-71 decomposition for 3x5x5 matrix multiplication.
    
    Partition:
    - i: {0}, {1,2}
    - j: {0,1}, {2,3}, {4}
    - k: {0,1}, {2,3}, {4}
    
    2x2x2 blocks (use Strassen, rank 7):
    - i={1,2}, j={0,1}, k={0,1}
    - i={1,2}, j={0,1}, k={2,3}
    - i={1,2}, j={2,3}, k={0,1}
    - i={1,2}, j={2,3}, k={2,3}
    
    All other blocks: schoolbook
    """
    n, m, p = 3, 5, 5
    
    # Precompute Strassen factors
    su, sv, sw = _strassen_222()
    
    # Total rank: 4*7 + remaining schoolbook terms
    # i={0}: 1*5*5 = 25 terms
    # i={1,2}, j={0,1}, k={4}: 2*2*1 = 4 terms
    # i={1,2}, j={2,3}, k={4}: 2*2*1 = 4 terms
    # i={1,2}, j={4}, k={0,1}: 2*1*2 = 4 terms
    # i={1,2}, j={4}, k={2,3}: 2*1*2 = 4 terms
    # i={1,2}, j={4}, k={4}: 2*1*1 = 2 terms
    # Total schoolbook: 25 + 4 + 4 + 4 + 4 + 2 = 43
    # Total rank: 4*7 + 43 = 28 + 43 = 71
    total_rank = 71
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Define blocks
    i_blocks = [(0,), (1, 2)]
    j_blocks = [(0, 1), (2, 3), (4,)]
    k_blocks = [(0, 1), (2, 3), (4,)]
    
    for i_b in range(len(i_blocks)):
        for j_b in range(len(j_blocks)):
            for k_b in range(len(k_blocks)):
                i_set = i_blocks[i_b]
                j_set = j_blocks[j_b]
                k_set = k_blocks[k_b]
                
                if len(i_set) == 2 and len(j_set) == 2 and len(k_set) == 2:
                    # Use Strassen
                    # Map local indices to global
                    # Local i: 0 -> i_set[0], 1 -> i_set[1]
                    # Local j: 0 -> j_set[0], 1 -> j_set[1]
                    # Local k: 0 -> k_set[0], 1 -> k_set[1]
                    for c in range(7):
                        for li in range(2):
                            gi = i_set[li]
                            for lj in range(2):
                                gj = j_set[lj]
                                U[gi * m + gj][col + c] = su[li * 2 + lj][c]
                        for lj in range(2):
                            gj = j_set[lj]
                            for lk in range(2):
                                gk = k_set[lk]
                                V[gj * p + gk][col + c] = sv[lj * 2 + lk][c]
                        for lk in range(2):
                            gk = k_set[lk]
                            for li in range(2):
                                gi = i_set[li]
                                W[gk * n + gi][col + c] = sw[lk * 2 + li][c]
                    col += 7
                else:
                    # Schoolbook
                    for gi in i_set:
                        for gj in j_set:
                            for gk in k_set:
                                U[gi * m + gj][col] = 1
                                V[gj * p + gk][col] = 1
                                W[gk * n + gi][col] = 1
                                col += 1
    
    if col != total_rank:
        raise ValueError(f"Column count mismatch: {col} != {total_rank}")
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, rank):
    """
    Verify that the tensor identity holds:
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_{r=0}^{rank-1} W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] == 1
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(W_np[k*n + i] * U_np[i*m + j] * V_np[j*p + k])
                if not np.isclose(s, 1.0, atol=1e-10):
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Try rank-71 decomposition
    try:
        U, V, W = _build_rank71()
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p, 71):
            return {"U": U, "V": V, "W": W}
        else:
            # Verification failed, fall back to schoolbook
            U, V, W = _schoolbook_factors(n, m, p)
            return {"U": U, "V": V, "W": W}
    except Exception:
        # Fall back to schoolbook
        U, V, W = _schoolbook_factors(n, m, p)
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
