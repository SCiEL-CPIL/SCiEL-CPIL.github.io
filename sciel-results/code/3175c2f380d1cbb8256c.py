"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition manually to achieve rank 78.
    # U shape: (n*m, rank) = (12, 78)
    # V shape: (m*p, rank) = (28, 78)
    # W shape: (p*n, rank) = (21, 78)
    
    # We'll use a list of tuples (u_idx, v_idx, w_idx, coeff) to build the columns
    # Or just fill arrays directly.
    
    # Rank calculation:
    # 2 blocks of (3x2x7)
    # Each 3x2x7:
    #   1x2x7 (rank 14)
    #   2x2x7 (rank 25)
    #     2x2x2 (rank 7)
    #     2x2x2 (rank 7)
    #     2x2x3 (rank 11) -> 2x2x2 (7) + 2x2x1 (4)
    # Total per 3x2x7: 14 + 7 + 7 + 7 + 4 = 39
    # Total rank: 39 * 2 = 78
    
    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        if coeff != 0:
            U[u_idx][col] = coeff
            V[v_idx][col] = coeff
            W[w_idx][col] = coeff
        col += 1

    def add_strassen_2x2x2(i_offset, j_offset, k_offset):
        # Adds 7 terms for 2x2x2 Strassen
        # i_offset: global i start (0 or 1 for the 2x2 block)
        # j_offset: global j start
        # k_offset: global k start
        nonlocal col
        # Local indices 0,1 map to global i_offset, i_offset+1 etc.
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00, A11. V: B00, B11. W: C00, C11
        # This is not a single rank-1 term. It's a product of sums.
        # In tensor decomposition, we need to expand this into rank-1 terms?
        # No, Strassen's algorithm replaces 8 multiplications with 7.
        # Each "multiplication" in Strassen is a scalar multiplication of two linear combinations.
        # This corresponds to 7 rank-1 terms in the tensor decomposition.
        # The coefficients in U, V, W can be -1, 0, 1.
        
        # Term 1: (A00+A11) * (B00+B11)
        # U entries: A00 (1), A11 (1)
        # V entries: B00 (1), B11 (1)
        # W entries: C00 (1), C11 (1) ... wait.
        # The contribution to C is:
        # C00 += M1
        # C11 += M1
        # So W00 += 1, W11 += 1.
        
        # We need to add a column to U, V, W.
        # U[i_offset, j_offset] += 1
        # U[i_offset+1, j_offset+1] += 1
        # V[j_offset, k_offset] += 1
        # V[j_offset+1, k_offset+1] += 1
        # W[k_offset, i_offset] += 1  (C00)
        # W[k_offset+1, i_offset+1] += 1 (C11)
        
        # But we must be careful with the indices.
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # M1
        U[i_offset*m + j_offset][col] += 1
        U[(i_offset+1)*m + (j_offset+1)][col] += 1
        V[j_offset*p + k_offset][col] += 1
        V[(j_offset+1)*p + (k_offset+1)][col] += 1
        W[k_offset*n + i_offset][col] += 1
        W[(k_offset+1)*n + (i_offset+1)][col] += 1
        col += 1
        
        # M2 = (A10+A11) * B00
        U[(i_offset+1)*m + j_offset][col] += 1
        U[(i_offset+1)*m + (j_offset+1)][col] += 1
        V[j_offset*p + k_offset][col] += 1
        W[k_offset*n + (i_offset+1)][col] += 1
        col += 1
        
        # M3 = A00 * (B01-B11)
        U[i_offset*m + j_offset][col] += 1
        V[j_offset*p + (k_offset+1)][col] += 1
        V[(j_offset+1)*p + (k_offset+1)][col] -= 1
        W[(k_offset+1)*n + i_offset][col] += 1
        col += 1
        
        # M4 = A11 * (B10-B00)
        U[(i_offset+1)*m + (j_offset+1)][col] += 1
        V[(j_offset+1)*p + k_offset][col] += 1
        V[j_offset*p + k_offset][col] -= 1
        W[k_offset*n + (i_offset+1)][col] += 1
        col += 1
        
        # M5 = (A00+A01) * B11
        U[i_offset*m + j_offset][col] += 1
        U[i_offset*m + (j_offset+1)][col] += 1
        V[(j_offset+1)*p + (k_offset+1)][col] += 1
        W[(k_offset+1)*n + i_offset][col] += 1
        col += 1
        
        # M6 = (A10-A00) * (B00+B01)
        U[(i_offset+1)*m + j_offset][col] += 1
        U[i_offset*m + j_offset][col] -= 1
        V[j_offset*p + k_offset][col] += 1
        V[j_offset*p + (k_offset+1)][col] += 1
        W[k_offset*n + (i_offset+1)][col] += 1
        W[(k_offset+1)*n + (i_offset+1)][col] += 1
        col += 1
        
        # M7 = (A01-A11) * (B10+B11)
        U[i_offset*m + (j_offset+1)][col] += 1
        U[(i_offset+1)*m + (j_offset+1)][col] -= 1
        V[(j_offset+1)*p + k_offset][col] += 1
        V[(j_offset+1)*p + (k_offset+1)][col] += 1
        W[k_offset*n + i_offset][col] += 1
        W[(k_offset+1)*n + i_offset][col] += 1
        col += 1

    def add_schoolbook_2x2x1(i_offset, j_offset, k_offset):
        # Adds 4 terms for 2x2x1 schoolbook
        nonlocal col
        for di in range(2):
            for dj in range(2):
                U[(i_offset+di)*m + (j_offset+dj)][col] = 1
                V[(j_offset+dj)*p + k_offset][col] = 1
                W[k_offset*n + (i_offset+di)][col] = 1
                col += 1

    def add_schoolbook_1x2x7(i_offset, j_offset, k_offset):
        # Adds 14 terms for 1x2x7 schoolbook
        # i is fixed (size 1)
        nonlocal col
        for dj in range(2):
            for dk in range(7):
                U[i_offset*m + (j_offset+dj)][col] = 1
                V[(j_offset+dj)*p + (k_offset+dk)][col] = 1
                W[(k_offset+dk)*n + i_offset][col] = 1
                col += 1

    # Block 1: j in {0, 1}
    # 1x2x7 part: i=0
    add_schoolbook_1x2x7(0, 0, 0)
    
    # 2x2x7 part: i in {1, 2}
    # Split k into {0,1}, {2,3}, {4,5,6}
    add_strassen_2x2x2(1, 0, 0) # k 0,1
    add_strassen_2x2x2(1, 0, 2) # k 2,3
    add_strassen_2x2x2(1, 0, 4) # k 4,5
    add_schoolbook_2x2x1(1, 0, 6) # k 6
    
    # Block 2: j in {2, 3}
    # 1x2x7 part: i=0
    add_schoolbook_1x2x7(0, 2, 0)
    
    # 2x2x7 part: i in {1, 2}
    add_strassen_2x2x2(1, 2, 0)
    add_strassen_2x2x2(1, 2, 2)
    add_strassen_2x2x2(1, 2, 4)
    add_schoolbook_2x2x1(1, 2, 6)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
