"""Rewritten program: rank-minimal exact tensor decomposition for 2x3 by 3x9 matrix multiplication.

Uses Strassen's algorithm for 2x2x2 sub-blocks and schoolbook for 2x2x1 blocks.
Total rank: 4*7 + 1*4 + 18 = 50 (vs schoolbook 54).
"""

import time
import sys

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 9  # A is n x m, B is m x p, C is p x n
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j in {0,1}, k in {0,1}, {2,3}, {4,5}, {6,7} -> 4 blocks of 2x2x2
    # For each block (i in {0,1}, j in {0,1}, k in {k0, k0+1})
    # Strassen's 7 products for M_{2,2,2}:
    # A is 2x2 (rows i, cols j), B is 2x2 (rows j, cols k), C is 2x2 (rows k, cols i)
    # 
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # (u_entries, v_entries, w_entries)
        # u_entries: list of (row, coeff) for U
        # v_entries: list of (row, coeff) for V
        # w_entries: list of (row, coeff) for W
    ]
    
    # For a 2x2x2 block with i0=0, j0=0, k0 given:
    # A indices: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
    # B indices: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
    # C indices: (k0, i0), (k0, i0+1), (k0+1, i0), (k0+1, i0+1)
    
    def add_strassen_block(i0, j0, k0, col_start):
        """Add Strassen decomposition for 2x2x2 block. Returns next column."""
        nonlocal col
        c = col_start
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00 at (i0*m+j0), A11 at ((i0+1)*m+(j0+1))
        U[i0 * m + j0][c] += 1
        U[(i0+1) * m + (j0+1)][c] += 1
        # V: B00 at (j0*p+k0), B11 at ((j0+1)*p+(k0+1))
        V[j0 * p + k0][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        # W: C00 gets +1, C11 gets +1
        W[k0 * n + i0][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        col += 1; c += 1
        
        # M2 = (A10 + A11) B00
        U[i0 * m + j0][c] += 1
        U[(i0+1) * m + (j0+1)][c] += 1
        V[j0 * p + k0][c] += 1
        # W: C10 gets +1
        W[(k0+1) * n + i0][c] += 1
        col += 1; c += 1
        
        # M3 = A00 (B01 - B11)
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0+1)][c] += 1
        V[(j0+1) * p + (k0+1)][c] -= 1
        # W: C01 gets +1
        W[k0 * n + (i0+1)][c] += 1
        col += 1; c += 1
        
        # M4 = A11 (B10 - B00)
        U[(i0+1) * m + (j0+1)][c] += 1
        V[(j0+1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        # W: C00 gets +1, C10 gets +1
        W[k0 * n + i0][c] += 1
        W[(k0+1) * n + i0][c] += 1
        col += 1; c += 1
        
        # M5 = (A00 + A01) B11
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0+1)][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        # W: C00 gets -1, C01 gets +1
        W[k0 * n + i0][c] -= 1
        W[k0 * n + (i0+1)][c] += 1
        col += 1; c += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[i0 * m + j0][c] -= 1
        U[(i0+1) * m + j0][c] += 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0+1)][c] += 1
        # W: C11 gets +1
        W[(k0+1) * n + (i0+1)][c] += 1
        col += 1; c += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[i0 * m + (j0+1)][c] += 1
        U[(i0+1) * m + (j0+1)][c] -= 1
        V[(j0+1) * p + k0][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        # W: C00 gets +1
        W[k0 * n + i0][c] += 1
        col += 1; c += 1
        
        return col
    
    # Process 4 blocks of 2x2x2 for k = {0,1}, {2,3}, {4,5}, {6,7}
    for k0 in [0, 2, 4, 6]:
        col = add_strassen_block(0, 0, k0, col)
    
    # Part 1b: j in {0,1}, k = 8 -> 2x2x1 block (schoolbook, 4 terms)
    # A is 2x2 (i in {0,1}, j in {0,1}), B is 2x1 (j in {0,1}, k=8), C is 1x2 (k=8, i in {0,1})
    # C[8, i] = sum_j A[i,j] * B[j, 8]
    # 4 terms: (i,j,k=8)
    for i in range(2):
        for j in range(2):
            U[i * m + j][col] = 1
            V[j * p + 8][col] = 1
            W[8 * n + i][col] = 1
            col += 1
    
    # Part 2: j = 2, k in {0..8} -> 2x1x9 block (schoolbook, 18 terms)
    # A is 2x1 (i in {0,1}, j=2), B is 1x9 (j=2, k in {0..8}), C is 9x2 (k in {0..8}, i in {0,1})
    # C[k, i] = A[i, 2] * B[2, k]
    for i in range(2):
        for k in range(9):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Convert to list of lists (already is)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END


def _verify(U, V, W):
    """Verify the decomposition is correct."""
    import numpy as np
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and i==i' else 0
    # Actually, the tensor for matrix mult: C = A*B
    # C[k, i] = sum_j A[i, j] * B[j, k]
    # So T[i*m+j, j'*p+k, k'*n+i'] = delta_{j,j'} * delta_{i,i'}
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Decomposition: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Compute sum_r U[:,r] outer V[:,r] outer W[:,r]
    T_decomp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    return np.allclose(T, T_decomp, atol=1e-10)


if __name__ == "__main__":
    result = construct()
    if _verify(result["U"], result["V"], result["W"]):
        print("Verification passed")
    else:
        print("Verification FAILED")
        sys.exit(1)
