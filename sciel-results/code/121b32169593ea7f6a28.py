import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    Strategy: View the 2x14 multiplication as a 2x2 matrix multiplication over the
    ring F[x]/(x^14), where the 14 columns of B form coefficients of a polynomial vector.
    Apply the 7 Strassen bilinear forms as scalar multipliers, yielding global rank 7.
    
    The tensor T_{i,j,k} = delta_{i,k} for matrix multiplication C = A * B where
    A is 2x2, B is 2x14, C is 2x14.
    
    Indexing convention:
    - A: row i (0..1), col j (0..1) -> index a = i*m + j = i*2 + j
    - B: row j (0..1), col k (0..13) -> index b = j*p + k = j*14 + k
    - C: row i (0..1), col k (0..13) -> index c = k*n + i = k*2 + i (official trace convention W[k*n+i, r])
    
    Rank-7 Strassen decomposition:
    For each Strassen term s (0..6), we have:
    - U[:, s]: coefficients for A
    - V[:, s]: coefficients for B  
    - W[:, s]: coefficients for C
    
    The key insight: since B has 14 columns and the ring is F[x]/(x^14),
    the Strassen bilinear forms extend naturally. Each Strassen term involves
    linear combinations of A entries and B entries, and the result is a linear
    combination of C entries.
    
    For the 2x2 case, Strassen's algorithm uses 7 multiplications:
    M0 = (A00 + A11)(B00 + B11)
    M1 = (A10 + A11)B00
    M2 = A00(B10 - B00)
    M3 = A11(B01 - B11)
    M4 = (A00 + A01)B11
    M5 = (A10 - A00)(B00 + B01)
    M6 = (A01 - A11)(B10 + B11)
    
    C00 = M0 + M3 - M5 - M6
    C01 = M2 + M6
    C10 = M1 + M5
    C11 = M0 - M1 + M2 + M4
    
    For the 2x14 case, each "entry" of B is actually a vector of 14 coefficients
    (one for each column). The Strassen bilinear forms apply component-wise
    across the 14 columns.
    
    So for each Strassen term s:
    - U[:, s] has non-zero entries only in the 2x2 A block (indices 0..3)
    - V[:, s] has non-zero entries in the 2x14 B block (indices 0..27), with
      the same pattern as the 2x2 case but replicated across all 14 columns
    - W[:, s] has non-zero entries in the 2x14 C block (indices 0..27), with
      the same pattern as the 2x2 case but replicated across all 14 columns
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28
    
    rank = 7
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)
    
    # Strassen terms: (a_indices, a_vals, b_indices, b_vals, c_indices, c_vals)
    # For 2x2: A indices 0..3 (00,01,10,11), B indices 0..3 (00,01,10,11), C indices 0..3 (00,01,10,11)
    strassen_terms = [
        # s=0: M0 = (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # s=1: M1 = (A10+A11)B00 -> C10 += M1, C11 -= M1
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        # s=2: M2 = A00(B10-B00) -> C01 += M2, C11 += M2
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
        # s=3: M3 = A11(B01-B11) -> C00 += M3, C11 += M3
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),
        # s=4: M4 = (A00+A01)B11 -> C11 += M4
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),
        # s=5: M5 = (A10-A00)(B00+B01) -> C00 -= M5, C10 += M5
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        # s=6: M6 = (A01-A11)(B10+B11) -> C00 -= M6, C01 += M6
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]
    
    for s in range(7):
        a_idx_list, a_val_list, b_idx_list, b_val_list, c_idx_list, c_val_list = strassen_terms[s]
        
        # Fill U: A has shape 2x2, index = i*m + j
        for a_idx, coeff in zip(a_idx_list, a_val_list):
            a_i = a_idx // m
            a_j = a_idx % m
            U[a_i * m + a_j, s] = coeff
        
        # Fill V: B has shape 2x14, index = j*p + k
        # The 2x2 B pattern applies to each column k independently
        for b_idx, coeff in zip(b_idx_list, b_val_list):
            b_j = b_idx // 2  # row in 2x2 (0 or 1)
            b_col_2x2 = b_idx % 2  # col in 2x2 (0 or 1)
            # For each of the 14 columns k, the coefficient is the same
            for k in range(p):
                V[b_j * p + k, s] = coeff
        
        # Fill W: C has shape 2x14, index = k*n + i (official trace convention)
        # The 2x2 C pattern applies to each column k independently
        for c_idx, coeff in zip(c_idx_list, c_val_list):
            c_i = c_idx // 2  # row in 2x2 (0 or 1)
            c_col_2x2 = c_idx % 2  # col in 2x2 (0 or 1)
            # For each of the 14 columns k, the coefficient is the same
            for k in range(p):
                W[k * n + c_i, s] = coeff
    
    U_list = U.astype(int).tolist()
    V_list = V.astype(int).tolist()
    W_list = W.astype(int).tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
