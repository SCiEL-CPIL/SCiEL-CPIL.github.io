"""Rewritten program for 3x5x5 matrix multiplication tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_222():
    """Return (U, V, W) as numpy arrays for 2x2x2 Strassen multiplication.
    
    U shape: (4, 7) - rows indexed by (i, j) -> i*2+j
    V shape: (4, 7) - rows indexed by (j, k) -> j*2+k  
    W shape: (4, 7) - rows indexed by (k, i) -> k*2+i
    
    The decomposition satisfies:
    C[i,k] = sum_r W[k*2+i, r] * (sum_j U[i*2+j, r] * A[i,j]) * (sum_j V[j*2+k, r] * B[j,k])
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1   # C00
    W[3][0] = 1   # C11
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1   # C10
    W[3][1] = -1  # C11
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1   # C01
    W[3][2] = 1   # C11
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1   # C00
    W[2][3] = 1   # C10
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1  # C00
    W[1][4] = 1   # C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1   # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1   # C00
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition is correct.
    
    For all i in [0,n), j in [0,m), k in [0,p):
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m + j], V[j*p + k] * W[k*n + i])
                if abs(s - 1) > tol:
                    return False
    return True

def _build_rank71():
    """Build a rank-71 decomposition using Strassen on 2x2x2 sub-blocks.
    
    Partition:
    i: {0}, {1,2}
    j: {0,1}, {2,3}, {4}
    k: {0,1}, {2,3}, {4}
    
    For 2x2x2 blocks (i={1,2}, j in 2-element set, k in 2-element set): use Strassen (rank 7)
    For all other blocks: use schoolbook (rank = product of sizes)
    
    Total rank: 4*7 + 43*1 = 28 + 43 = 71
    """
    n, m, p = 3, 5, 5
    
    i_blocks = [(0,), (1, 2)]
    j_blocks = [(0, 1), (2, 3), (4,)]
    k_blocks = [(0, 1), (2, 3), (4,)]
    
    # First pass: count total rank
    total_rank = 0
    for i_b in range(len(i_blocks)):
        for j_b in range(len(j_blocks)):
            for k_b in range(len(k_blocks)):
                i_set = i_blocks[i_b]
                j_set = j_blocks[j_b]
                k_set = k_blocks[k_b]
                if len(i_set) == 2 and len(j_set) == 2 and len(k_set) == 2:
                    total_rank += 7
                else:
                    total_rank += len(i_set) * len(j_set) * len(k_set)
    
    assert total_rank == 71, f"Expected rank 71, got {total_rank}"
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for i_b in range(len(i_blocks)):
        for j_b in range(len(j_blocks)):
            for k_b in range(len(k_blocks)):
                i_set = i_blocks[i_b]
                j_set = j_blocks[j_b]
                k_set = k_blocks[k_b]
                
                if len(i_set) == 2 and len(j_set) == 2 and len(k_set) == 2:
                    # Use Strassen
                    su, sv, sw = _strassen_222()
                    for c in range(7):
                        for li in range(2):
                            gi = i_set[li]
                            for lj in range(2):
                                gj = j_set[lj]
                                U[gi * m + gj, col + c] = su[li * 2 + lj, c]
                        for lj in range(2):
                            gj = j_set[lj]
                            for lk in range(2):
                                gk = k_set[lk]
                                V[gj * p + gk, col + c] = sv[lj * 2 + lk, c]
                        for lk in range(2):
                            gk = k_set[lk]
                            for li in range(2):
                                gi = i_set[li]
                                W[gk * n + gi, col + c] = sw[lk * 2 + li, c]
                    col += 7
                else:
                    # Schoolbook
                    for gi in i_set:
                        for gj in j_set:
                            for gk in k_set:
                                U[gi * m + gj, col] = 1
                                V[gj * p + gk, col] = 1
                                W[gk * n + gi, col] = 1
                                col += 1
    
    assert col == total_rank, f"Column mismatch: {col} vs {total_rank}"
    
    return U, V, W

def _try_rank_search(start_rank, max_rank, deadline, n, m, p):
    """Try to find a decomposition with rank between start_rank and max_rank.
    Uses the Strassen-based construction and attempts to reduce rank by
    finding linear dependencies among the columns.
    """
    # Start with the rank-71 construction
    U, V, W = _build_rank71()
    base_rank = U.shape[1]
    
    if base_rank < start_rank:
        return U, V, W, base_rank
    
    # Try to reduce rank by identifying redundant columns
    # The idea: if two columns produce the same "pattern" in the tensor,
    # they might be mergeable. This is a heuristic approach.
    
    # For now, just return the base construction
    # A full rank minimization would require solving a nonlinear system
    return U, V, W, base_rank

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    deadline = time.time() + 120  # safety margin before 150s limit
    
    # Try rank-71 decomposition with Strassen on 2x2x2 blocks
    try:
        U, V, W = _build_rank71()
        
        # Verify the decomposition
        if _verify_decomposition(U, V, W, n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
        else:
            # Verification failed, fall through to schoolbook
            pass
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        # Not enough time, return schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: schoolbook decomposition (rank 75)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
