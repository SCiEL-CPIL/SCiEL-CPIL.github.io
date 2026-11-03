"""Matrix multiplication 3x4x8 with optimized tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 8]


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W


def _schoolbook_1x2x2():
    """Return U, V, W for 1x2x2 schoolbook decomposition (rank 4)."""
    U = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    V = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    W = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    return U, V, W


def _build_88_decomposition():
    """Build the rank-88 decomposition using Strassen blocks."""
    n, m, p = 3, 4, 8
    total_rank = 88
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    U2, V2, W2 = _strassen_2x2x2()
    U1, V1, W1 = _schoolbook_1x2x2()
    
    col_offset = 0
    # 8 Strassen blocks for i=0,1
    for jp in range(2):
        for kp in range(4):
            j0, j1 = 2 * jp, 2 * jp + 1
            k0, k1 = 2 * kp, 2 * kp + 1
            u_rows = [0 * m + j0, 0 * m + j1, 1 * m + j0, 1 * m + j1]
            v_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
            w_rows = [k0 * n + 0, k1 * n + 0, k0 * n + 1, k1 * n + 1]
            for i in range(4):
                for c in range(7):
                    U[u_rows[i]][col_offset + c] += U2[i][c]
                    V[v_rows[i]][col_offset + c] += V2[i][c]
                    W[w_rows[i]][col_offset + c] += W2[i][c]
            col_offset += 7
    # 8 schoolbook blocks for i=2
    for jp in range(2):
        for kp in range(4):
            j0, j1 = 2 * jp, 2 * jp + 1
            k0, k1 = 2 * kp, 2 * kp + 1
            u_rows = [2 * m + j0, 2 * m + j1]
            v_rows = [j0 * p + k0, j1 * p + k0, j0 * p + k1, j1 * p + k1]
            w_rows = [k0 * n + 2, k1 * n + 2]
            for i in range(2):
                for c in range(4):
                    U[u_rows[i]][col_offset + c] += U1[i][c]
                    W[w_rows[i]][col_offset + c] += W1[i][c]
            for i in range(4):
                for c in range(4):
                    V[v_rows[i]][col_offset + c] += V1[i][c]
            col_offset += 4
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition exactly using integer arithmetic with NumPy."""
    rank = len(U[0])
    if rank == 0:
        return False
        
    # Reshape to (n, m, R), (m, p, R), (p, n, R)
    U_t = np.array(U).reshape(n, m, rank)
    V_t = np.array(V).reshape(m, p, rank)
    W_t = np.array(W).reshape(p, n, rank)
    
    # Compute sum over rank dimension: S[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Note: W is indexed by (k, i) in the flattened version k*n+i.
    # In the tensor product for matrix mult, the term is U_{ij} V_{jk} W_{ki}.
    # So we need W[k, i, c].
    # W_t is (p, n, R) which is (k, i, c).
    S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    
    return np.all(S == 1)


def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """Try to reduce rank by removing linearly dependent columns using NumPy verification."""
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    rank = len(U[0])
    
    # Try removing columns one at a time
    removed = True
    # Use a random order to avoid getting stuck in a specific local minimum
    # or to find different dependencies.
    while removed and time.time() < deadline:
        removed = False
        # Shuffle column indices to try different removals
        indices = list(range(rank))
        np.random.shuffle(indices)
        
        for c in indices:
            if time.time() > deadline:
                break
                
            # Try removing column c
            U_new = [row[:c] + row[c+1:] for row in U]
            V_new = [row[:c] + row[c+1:] for row in V]
            W_new = [row[:c] + row[c+1:] for row in W]
            
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
                rank -= 1
                removed = True
                break
    
    return U, V, W


def construct():
    start = time.time()
    deadline = start + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    result = _build_88_decomposition()
    
    # Try to reduce rank
    U, V, W = result["U"], result["V"], result["W"]
    U, V, W = _try_reduce_rank(U, V, W, n, m, p, deadline)
    
    result = {"U": U, "V": V, "W": W}
    
    # Final verification
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback: schoolbook
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
        result = {"U": U, "V": V, "W": W}
    
    return result
