"""Tensor decomposition for 2x3 * 3x9 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # We'll build rank-1 terms for the decomposition.
    # Each term is (u_vec, v_vec, w_vec) where:
    #   u_vec has length n*m (row index into A: i*m + j)
    #   v_vec has length m*p (row index into B: j*p + k)
    #   w_vec has length p*n (row index into C: k*n + i)
    
    terms = []
    
    # Part 1: A[0:2, 0:2] * B[0:2, 0:9] -> C[0:9, 0:2]
    # Split B columns into blocks of size 2 (and one of size 1):
    #   B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8], B[0:2, 8:9]
    
    # For each 2x2x2 block, use Strassen (rank 7)
    # For the 2x2x1 block, use schoolbook (rank 4)
    
    def add_strassen_222(a_rows, b_cols, c_cols, i_off, j_start, k_start):
        """
        A rows: a_rows = [i0, i1] (indices into A's rows, i.e., 0 and 1)
        B cols: b_cols = [k0, k1] (indices into B's columns, offset by k_start)
        C cols: c_cols = [k0, k1] (indices into C's columns, offset by k_start)
        
        A submatrix: A[i0:j0+1, j_start:j_start+2] where j0 = j_start, j1 = j_start+1
        B submatrix: B[j_start:j_start+2, k_start:k_start+2]
        C submatrix: C[k_start:k_start+2, i_off:i_off+2]
        
        Wait, let me reconsider the indexing.
        
        A is n x m = 2 x 3. Row index i in {0,1}, col index j in {0,1,2}.
        B is m x p = 3 x 9. Row index j in {0,1,2}, col index k in {0,...,8}.
        C is p x n = 9 x 2. Row index k in {0,...,8}, col index i in {0,1}.
        
        For the sub-problem A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] = C[k0:k0+2, i0:i0+2]:
        
        A sub has rows (i0, i0+1) and cols (j0, j0+1). But wait, A only has 2 rows total (n=2).
        So i0 must be 0, and i0+1 = 1. The sub-problem uses both rows of A.
        
        U vector indices: i*m + j for i in {0,1}, j in {j0, j0+1}
        V vector indices: j*p + k for j in {j0, j0+1}, k in {k0, k0+1}
        W vector indices: k*n + i for k in {k0, k0+1}, i in {0,1}
        
        Strassen for 2x2 * 2x2:
        Let A_sub = [[a00, a01], [a10, a11]] where a_ij = A[i+j0_offset, j0+j]
        Let B_sub = [[b00, b01], [b10, b11]] where b_ij = B[j0+i, k0+j]
        Let C_sub = [[c00, c01], [c10, c11]] where c_ij = C[k0+i, i0+j]
        
        M1 = (a00 + a11)(b10 + b11)
        M2 = (a10 + a11)b00
        M3 = a00(b01 - b11)
        M4 = a11(b10 - b00)
        M5 = (a00 + a01)b11
        M6 = (a10 - a00)(b00 + b01)
        M7 = (a01 - a11)(b10 + b11)
        
        c00 = M1 + M4 - M5 + M7
        c01 = M3 + M5
        c10 = M2 + M4
        c11 = M1 - M2 + M3 + M6
        
        For each Mk, we add a rank-1 term.
        """
        # U indices for A sub: (0, j0), (0, j0+1), (1, j0), (1, j0+1)
        # i.e., u_idx[0] = 0*m + j0, u_idx[1] = 0*m + (j0+1), u_idx[2] = 1*m + j0, u_idx[3] = 1*m + (j0+1)
        u00 = 0 * m + j0      # a00
        u01 = 0 * m + (j0 + 1)  # a01
        u10 = 1 * m + j0      # a10
        u11 = 1 * m + (j0 + 1)  # a11
        
        # V indices for B sub: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
        v00 = j0 * p + k0          # b00
        v01 = j0 * p + (k0 + 1)    # b01
        v10 = (j0 + 1) * p + k0    # b10
        v11 = (j0 + 1) * p + (k0 + 1)  # b11
        
        # W indices for C sub: (k0, 0), (k0, 1), (k0+1, 0), (k0+1, 1)
        w00 = k0 * n + 0     # c00
        w01 = k0 * n + 1     # c01
        w10 = (k0 + 1) * n + 0   # c10
        w11 = (k0 + 1) * n + 1   # c11
        
        # M1 = (a00 + a11)(b10 + b11) -> c00 += 1, c11 += 1
        terms.append(([u00, u11], [v10, v11], [w00, w11], [(1,1), (1,1), (1,1)]))
        
        # M2 = (a10 + a11)b00 -> c10 += 1, c11 -= 1
        terms.append(([u10, u11], [v00], [w10, w11], [(1,1), (1,1), (1,-1)]))
        
        # M3 = a00(b01 - b11) -> c01 += 1, c11 += 1
        terms.append(([u00], [v01, v11], [w01, w11], [(1,1), (1,-1), (1,1)]))
        
        # M4 = a11(b10 - b00) -> c00 += 1, c10 += 1
        terms.append(([u11], [v10, v00], [w00, w10], [(1,1), (1,-1), (1,1)]))
        
        # M5 = (a00 + a01)b11 -> c00 -= 1, c01 += 1
        terms.append(([u00, u01], [v11], [w00, w01], [(1,1), (1,1), (-1,1)]))
        
        # M6 = (a10 - a00)(b00 + b01) -> c11 += 1
        terms.append(([u10, u00], [v00, v01], [w11], [(1,-1), (1,1), (1,1)]))
        
        # M7 = (a01 - a11)(b10 + b11) -> c00 += 1
        terms.append(([u01, u11], [v10, v11], [w00], [(1,-1), (1,1), (1,1)]))
    
    def add_schoolbook_221(a_rows, b_cols, c_cols, i_off, j0, k0):
        """
        A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+1] = C[k0:k0+1, i0:i0+2]
        This is 2x2 * 2x1 = 2x1, rank 4 by schoolbook.
        """
        u00 = 0 * m + j0
        u01 = 0 * m + (j0 + 1)
        u10 = 1 * m + j0
        u11 = 1 * m + (j0 + 1)
        
        v00 = j0 * p + k0
        v10 = (j0 + 1) * p + k0
        
        w00 = k0 * n + 0
        w01 = k0 * n + 1
        
        # c0 = a00*b00 + a01*b10
        # c1 = a10*b00 + a11*b10
        
        # Term 1: a00 * b00 -> c0
        terms.append(([u00], [v00], [w00], [(1,1), (1,1), (1,1)]))
        # Term 2: a01 * b10 -> c0
        terms.append(([u01], [v10], [w00], [(1,1), (1,1), (1,1)]))
        # Term 3: a10 * b00 -> c1
        terms.append(([u10], [v00], [w01], [(1,1), (1,1), (1,1)]))
        # Term 4: a11 * b10 -> c1
        terms.append(([u11], [v10], [w01], [(1,1), (1,1), (1,1)]))
    
    # Part 1: j0=0, using A cols 0,1
    # B col blocks: k0=0,2,4,6 (size 2), k0=8 (size 1)
    add_strassen_222(None, None, None, 0, 0, 0)
    add_strassen_222(None, None, None, 0, 0, 2)
    add_strassen_222(None, None, None, 0, 0, 4)
    add_strassen_222(None, None, None, 0, 0, 6)
    add_schoolbook_221(None, None, None, 0, 0, 8)
    
    # Part 2: j0=2, using A col 2 (so 2x1 * 1x9)
    # A[0:2, 2:3] * B[2:3, 0:9] = C[0:9, 0:2]
    # This is 2x1 * 1x9, schoolbook rank = 2*1*9 = 18
    for k in range(9):
        # A[0,2] * B[2,k] -> C[k,0]
        u_idx = 0 * m + 2
        v_idx = 2 * p + k
        w_idx = k * n + 0
        terms.append(([u_idx], [v_idx], [w_idx], [(1,1), (1,1), (1,1)]))
        # A[1,2] * B[2,k] -> C[k,1]
        u_idx = 1 * m + 2
        v_idx = 2 * p + k
        w_idx = k * n + 1
        terms.append(([u_idx], [v_idx], [w_idx], [(1,1), (1,1), (1,1)]))
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_indices, v_indices, w_indices, coeffs) in enumerate(terms):
        for idx, val in zip(u_indices, coeffs[0]):
            U[idx][col] = val
        for idx, val in zip(v_indices, coeffs[1]):
            V[idx][col] = val
        for idx, val in zip(w_indices, coeffs[2]):
            W[idx][col] = val
    
    return {"U": U, "V": V, "W": W}
