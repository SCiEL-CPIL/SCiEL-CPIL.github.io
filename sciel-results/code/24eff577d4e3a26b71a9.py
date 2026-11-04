"""Optimized tensor decomposition for 2x4 * 4x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Construct a rank-36 exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    
    Uses Strassen's algorithm on 2x2 sub-blocks for the 2x2*2x2 parts and
    schoolbook for the 2x2*2x1 part.
    
    Returns:
        dict with keys 'U', 'V', 'W' containing lists of factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Strategy:
    # Split the middle dimension j into two blocks of size 2: j in {0,1} and j in {2,3}.
    # Split the output dimension k into: {0,1}, {2,3}, {4}.
    # 
    # For each j-block (size 2) and k-block:
    # - If k-block size is 2: use Strassen (rank 7)
    # - If k-block size is 1: use schoolbook (rank 4)
    #
    # Total rank: 2 (j-blocks) * (2 (k-blocks of size 2) * 7 + 1 (k-block of size 1) * 4)
    #           = 2 * (14 + 4) = 2 * 18 = 36
    
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_term(idx, u_coeffs, v_coeffs, w_coeffs):
        """
        Add a rank-1 term to U, V, W.
        u_coeffs: dict {(i, j): val}
        v_coeffs: dict {(j, k): val}
        w_coeffs: dict {(k, i): val}
        """
        for (i, j), val in u_coeffs.items():
            U[i*m + j, idx] += val
        for (j, k), val in v_coeffs.items():
            V[j*p + k, idx] += val
        for (k, i), val in w_coeffs.items():
            W[k*n + i, idx] += val

    def strassen_block(a_row_offset, b_row_offset, c_col_offset, idx_start):
        """
        Add Strassen decomposition for a 2x2x2 block.
        a_row_offset: column offset for A (0 or 2)
        b_row_offset: row offset for B (0 or 2)
        c_col_offset: col offset for C (0 or 2)
        idx_start: starting index in U, V, W
        """
        # Local indices:
        # A_local[i, j] = A[i, a_row_offset + j] for i in {0,1}, j in {0,1}
        # B_local[j, k] = B[b_row_offset + j, c_col_offset + k] for j in {0,1}, k in {0,1}
        # C_local[i, k] = C[i, c_col_offset + k] for i in {0,1}, k in {0,1}
        #
        # Note: The problem statement says W[k*n+i, r].
        # So W index is (k, i).
        
        # Strassen products:
        # P1 = (A00 + A11) * (B00 + B11)
        # P2 = (A10 + A11) * B00
        # P3 = A00 * (B01 - B11)
        # P4 = A11 * (B10 - B00)
        # P5 = (A00 + A01) * B11
        # P6 = (A10 - A00) * (B00 + B01)
        # P7 = (A01 - A11) * (B10 + B11)
        
        # Results:
        # C00 = P1 - P4 + P5 + P6  (Wait, standard Strassen is:
        # C00 = P1 - P4 + P5 + P6? No.
        # Let's check standard Strassen:
        # P1 = (a+d)(e+h)
        # P2 = (c+d)e
        # P3 = a(f-h)
        # P4 = d(g-e)
        # P5 = (a+b)h
        # P6 = (c-a)(e+f)
        # P7 = (b-d)(g+h)
        #
        # c = P1 + P4 - P5 + P7
        # d = P3 + P5
        # e = P2 + P4
        # f = P1 - P2 + P3 + P6
        #
        # Mapping to our indices:
        # A = [[A00, A01], [A10, A11]] -> a=A00, b=A01, c=A10, d=A11
        # B = [[B00, B01], [B10, B11]] -> e=B00, f=B01, g=B10, h=B11
        # C = [[C00, C01], [C10, C11]]
        #
        # C00 (i=0, k=0) = P1 + P4 - P5 + P7
        # C01 (i=0, k=1) = P3 + P5
        # C10 (i=1, k=0) = P2 + P4
        # C11 (i=1, k=1) = P1 - P2 + P3 + P6
        
        # Term 1: P1
        add_term(idx_start,
                 {(0, a_row_offset): 1, (1, a_row_offset+1): 1},  # A00, A11
                 {(b_row_offset, c_col_offset): 1, (b_row_offset+1, c_col_offset+1): 1},  # B00, B11
                 {(c_col_offset, 0): 1, (c_col_offset+1, 1): 1})  # C00 += 1, C11 += 1
        
        # Term 2: P2
        add_term(idx_start+1,
                 {(1, a_row_offset): 1, (1, a_row_offset+1): 1},  # A10, A11
                 {(b_row_offset, c_col_offset): 1},  # B00
                 {(c_col_offset, 1): 1, (c_col_offset+1, 1): -1}) # C10 += 1, C11 -= 1
        
        # Term 3: P3
        add_term(idx_start+2,
                 {(0, a_row_offset): 1},  # A00
                 {(b_row_offset, c_col_offset+1): 1, (b_row_offset+1, c_col_offset+1): -1},  # B01, -B11
                 {(c_col_offset+1, 0): 1, (c_col_offset+1, 1): 1}) # C01 += 1, C11 += 1
        
        # Term 4: P4
        add_term(idx_start+3,
                 {(1, a_row_offset+1): 1},  # A11
                 {(b_row_offset+1, c_col_offset): 1, (b_row_offset, c_col_offset): -1},  # B10, -B00
                 {(c_col_offset, 0): 1, (c_col_offset, 1): 1}) # C00 += 1, C10 += 1
        
        # Term 5: P5
        add_term(idx_start+4,
                 {(0, a_row_offset): 1, (0, a_row_offset+1): 1},  # A00, A01
                 {(b_row_offset+1, c_col_offset+1): 1},  # B11
                 {(c_col_offset, 0): -1, (c_col_offset+1, 0): 1}) # C00 -= 1, C01 += 1
        
        # Term 6: P6
        add_term(idx_start+5,
                 {(1, a_row_offset): 1, (0, a_row_offset): -1},  # A10, -A00
                 {(b_row_offset, c_col_offset): 1, (b_row_offset, c_col_offset+1): 1},  # B00, B01
                 {(c_col_offset, 0): 1}) # C00 += 1
        
        # Term 7: P7
        add_term(idx_start+6,
                 {(0, a_row_offset+1): 1, (1, a_row_offset+1): -1},  # A01, -A11
                 {(b_row_offset+1, c_col_offset): 1, (b_row_offset+1, c_col_offset+1): 1},  # B10, B11
                 {(c_col_offset+1, 1): 1}) # C11 += 1

    def schoolbook_2x2x1_block(a_row_offset, b_row_offset, c_col_offset, idx_start):
        """
        Add schoolbook decomposition for a 2x2x1 block.
        A is 2x2, B is 2x1, C is 2x1.
        """
        # C00 += A00 B00 + A01 B10
        # C10 += A10 B00 + A11 B10
        
        # Term 1: A00 B00 -> C00
        add_term(idx_start,
                 {(0, a_row_offset): 1},
                 {(b_row_offset, c_col_offset): 1},
                 {(c_col_offset, 0): 1})
        
        # Term 2: A01 B10 -> C00
        add_term(idx_start+1,
                 {(0, a_row_offset+1): 1},
                 {(b_row_offset+1, c_col_offset): 1},
                 {(c_col_offset, 0): 1})
        
        # Term 3: A10 B00 -> C10
        add_term(idx_start+2,
                 {(1, a_row_offset): 1},
                 {(b_row_offset, c_col_offset): 1},
                 {(c_col_offset, 1): 1})
        
        # Term 4: A11 B10 -> C10
        add_term(idx_start+3,
                 {(1, a_row_offset+1): 1},
                 {(b_row_offset+1, c_col_offset): 1},
                 {(c_col_offset, 1): 1})

    idx = 0
    # Block 1: j in {0,1} (a_row_offset=0, b_row_offset=0)
    strassen_block(0, 0, 0, idx)       # k in {0,1}
    idx += 7
    strassen_block(0, 0, 2, idx)       # k in {2,3}
    idx += 7
    schoolbook_2x2x1_block(0, 0, 4, idx) # k in {4}
    idx += 4
    
    # Block 2: j in {2,3} (a_row_offset=2, b_row_offset=2)
    strassen_block(2, 2, 0, idx)       # k in {0,1}
    idx += 7
    strassen_block(2, 2, 2, idx)       # k in {2,3}
    idx += 7
    schoolbook_2x2x1_block(2, 2, 4, idx) # k in {4}
    idx += 4
    
    if idx != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {idx}")
    
    # Verify
    def verify(U, V, W, tol=1e-9):
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    assert verify(U, V, W), "Decomposition failed verification"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
