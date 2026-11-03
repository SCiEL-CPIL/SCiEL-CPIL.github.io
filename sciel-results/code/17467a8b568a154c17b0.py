"""Matrix multiplication 3x4x8 with optimized tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    # Standard Strassen factors
    # U[i, r] corresponds to A[i, j] contribution
    # V[j, r] corresponds to B[j, k] contribution
    # W[k, i, r] corresponds to C[k, i] contribution? 
    # Contract: W[k*n+i, r]
    # In 2x2x2, n=2, m=2, p=2.
    # U is 4x7, V is 4x7, W is 4x7.
    # Indices:
    # U: i*m+j -> 0,1,2,3 (i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1)
    # V: j*p+k -> 0,1,2,3 (j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1)
    # W: k*n+i -> 0,1,2,3 (k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1)
    
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
    # n=1, m=2, p=2
    # U size: 1*2 = 2 rows
    # V size: 2*2 = 4 rows
    # W size: 2*1 = 2 rows
    # Rank 4
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
    """Build the rank-88 decomposition using Strassen and Schoolbook blocks."""
    n, m, p = 3, 4, 8
    total_rank = 88
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    U2, V2, W2 = _strassen_2x2x2()
    U1, V1, W1 = _schoolbook_1x2x2()
    
    col_offset = 0
    
    # Loop over j-pairs (rows of B, cols of A)
    # m=4, so j in 0..3. Pairs: (0,1) and (2,3). jp in 0..1
    for jp in range(2):
        j0, j1 = 2 * jp, 2 * jp + 1
        
        # Loop over k-pairs (cols of B, cols of C)
        # p=8, so k in 0..7. Pairs: (0,1), (2,3), (4,5), (6,7). kp in 0..3
        for kp in range(4):
            k0, k1 = 2 * kp, 2 * kp + 1
            
            # Case 1: i in {0, 1} -> 2x2x2 Strassen
            # U rows: i*m+j for i in {0,1}, j in {j0,j1}
            # Order: (0,j0), (0,j1), (1,j0), (1,j1)
            u_rows = [0 * m + j0, 0 * m + j1, 1 * m + j0, 1 * m + j1]
            
            # V rows: j*p+k for j in {j0,j1}, k in {k0,k1}
            # Order: (j0,k0), (j0,k1), (j1,k0), (j1,k1)
            v_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
            
            # W rows: k*n+i for k in {k0,k1}, i in {0,1}
            # Order: (k0,0), (k0,1), (k1,0), (k1,1)
            w_rows = [k0 * n + 0, k0 * n + 1, k1 * n + 0, k1 * n + 1]
            
            for idx in range(4):
                for c in range(7):
                    U[u_rows[idx]][col_offset + c] += U2[idx][c]
                    V[v_rows[idx]][col_offset + c] += V2[idx][c]
                    W[w_rows[idx]][col_offset + c] += W2[idx][c]
            
            col_offset += 7
            
            # Case 2: i = 2 -> 1x2x2 Schoolbook
            # U rows: i*m+j for i=2, j in {j0,j1}
            # Order: (2,j0), (2,j1)
            u_rows_1 = [2 * m + j0, 2 * m + j1]
            
            # V rows: same as above
            # v_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
            
            # W rows: k*n+i for k in {k0,k1}, i=2
            # Order: (k0,2), (k1,2)
            w_rows_1 = [k0 * n + 2, k1 * n + 2]
            
            for idx in range(2):
                for c in range(4):
                    U[u_rows_1[idx]][col_offset + c] += U1[idx][c]
                    W[w_rows_1[idx]][col_offset + c] += W1[idx][c]
            
            for idx in range(4):
                for c in range(4):
                    V[v_rows[idx]][col_offset + c] += V1[idx][c]
            
            col_offset += 4
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if val != 1:
                    return False
    return True


def construct():
    start = time.time()
    deadline = start + 140  # safety margin before 150s limit
    
    # Directly construct the known rank-88 solution
    result = _build_88_decomposition()
    
    n, m, p = DIMENSIONS
    
    # Verify to be safe
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback: schoolbook (rank 96)
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
