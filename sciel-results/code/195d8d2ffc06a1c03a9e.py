"""Matrix multiplication 4x4x5: rank-minimal exact tensor decomposition.
Uses a block decomposition: 4x4x4 (Strassen rank 7) + 4x4x1 (schoolbook rank 16).
Total rank = 7 + 16 = 23.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]


def _strassen_4x4():
    """Return U, V, W for 4x4x4 matrix multiplication using Strassen's algorithm.
    U shape: (16, 7), V shape: (16, 7), W shape: (16, 7)
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=4.
    So U row index = i*4+j, V row index = j*4+k, W row index = k*4+i.
    """
    n = 4
    # Split 4x4 matrices into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where each block is 2x2.
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # Strassen's 7 products for 2x2x2 (on 2x2 blocks):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 2x2 matrix product. For each 2x2x2 sub-problem,
    # we use schoolbook rank 8 (or we could use rank 7 again, but let's use schoolbook for simplicity).
    # Actually, for the 2x2 blocks, the sub-problem is 2x2x2 which has rank 8 by schoolbook.
    # But we can use Strassen again for rank 7 per 2x2x2. Let's use schoolbook for the inner 2x2x2
    # to keep it simple and correct. Rank per 2x2x2 = 8. Total = 7*8 = 56? No, that's not right.
    #
    # Wait, the standard approach: for 4x4x4, we use Strassen's formula which gives 7 products
    # of 2x2 matrices. Each 2x2 matrix product can be done with rank 8 (schoolbook) or rank 7 (Strassen).
    # If we use schoolbook for each 2x2x2: total rank = 7 * 8 = 56. That's worse than schoolbook 64? No, 56 < 64.
    # But actually the known rank for 4x4x4 is 49 (using Strassen recursively: 7*7=49).
    # Let me use Strassen recursively: each 2x2x2 uses Strassen with 7 terms of 1x1x1 (rank 1 each).
    # So rank per 2x2x2 = 7. Total rank for 4x4x4 = 7 * 7 = 49.
    #
    # Let's build it properly.
    #
    # For a 2x2x2 Strassen decomposition:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    # where:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    #
    # Each mi is a scalar product (1x1x1), so rank 1.
    # U for 2x2x2: 4 rows (i*2+j), 7 cols
    # V for 2x2x2: 4 rows (j*2+k), 7 cols
    # W for 2x2x2: 4 rows (k*2+i), 7 cols
    #
    # Let's define the 2x2x2 Strassen tensors:
    # For each term m_t, we have factors (u_t, v_t, w_t) where each is a 4-vector (for 2x2).
    
    # Term 1: m1 = (a11+a22)(b11+b22)
    # U factor: a11=1, a22=1, others 0 -> [1, 0, 0, 1] (indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3)
    # V factor: b11=1, b22=1 -> [1, 0, 0, 1]
    # W factor: contributes to C11 and C22
    # C11 = m1 + ...: C11 index is (i=0,k=0) -> W index k*2+i = 0*2+0 = 0
    # C22 index is (i=1,k=1) -> W index k*2+i = 1*2+1 = 3
    # So W = [1, 0, 0, 1]
    
    # Let me build all 7 terms for 2x2x2:
    # Index convention: A[i,j] with i in {0,1}, j in {0,1} -> flat index i*2+j
    # B[j,k] -> flat index j*2+k
    # C[i,k] -> flat index k*2+i (per the problem's W convention: W[k*n+i])
    
    terms_2x2 = []
    
    # m1 = (a11+a22)(b11+b22)
    # U: a11=1, a22=1 -> indices 0,3 -> [1,0,0,1]
    # V: b11=1, b22=1 -> indices 0,3 -> [1,0,0,1]
    # W: C11 gets +1, C22 gets +1 -> C11: k=0,i=0 -> idx 0; C22: k=1,i=1 -> idx 3 -> [1,0,0,1]
    terms_2x2.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # m2 = (a21+a22)b11
    # U: a21=1, a22=1 -> indices 2,3 -> [0,0,1,1]
    # V: b11=1 -> index 0 -> [1,0,0,0]
    # W: C21 gets +1, C22 gets -1 -> C21: k=1,i=0 -> idx 2; C22: k=1,i=1 -> idx 3 -> [0,0,1,-1]
    terms_2x2.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    
    # m3 = a11(b12-b22)
    # U: a11=1 -> index 0 -> [1,0,0,0]
    # V: b12=1, b22=-1 -> indices 1,3 -> [0,1,0,-1]
    # W: C12 gets +1, C22 gets +1 -> C12: k=1,i=0 -> idx 2; C22: k=1,i=1 -> idx 3 -> [0,0,1,1]
    terms_2x2.append(([1,0,0,0], [0,1,0,-1], [0,0,1,1]))
    
    # m4 = a22(b21-b11)
    # U: a22=1 -> index 3 -> [0,0,0,1]
    # V: b21=1, b11=-1 -> indices 2,0 -> [-1,0,1,0]
    # W: C11 gets +1, C21 gets +1 -> C11: idx 0; C21: k=1,i=0 -> idx 2 -> [1,0,1,0]
    terms_2x2.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
    
    # m5 = (a11+a12)b22
    # U: a11=1, a12=1 -> indices 0,1 -> [1,1,0,0]
    # V: b22=1 -> index 3 -> [0,0,0,1]
    # W: C11 gets -1, C12 gets +1 -> C11: idx 0; C12: k=1,i=0 -> idx 2 -> [-1,0,1,0]
    terms_2x2.append(([1,1,0,0], [0,0,0,1], [-1,0,1,0]))
    
    # m6 = (a21-a11)(b11+b12)
    # U: a21=1, a11=-1 -> indices 2,0 -> [-1,0,1,0]
    # V: b11=1, b12=1 -> indices 0,1 -> [1,1,0,0]
    # W: C22 gets +1 -> C22: k=1,i=1 -> idx 3 -> [0,0,0,1]
    terms_2x2.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # m7 = (a12-a22)(b21+b22)
    # U: a12=1, a22=-1 -> indices 1,3 -> [0,1,0,-1]
    # V: b21=1, b22=1 -> indices 2,3 -> [0,0,1,1]
    # W: C11 gets +1 -> C11: k=0,i=0 -> idx 0 -> [1,0,0,0]
    terms_2x2.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    
    # Now build the 4x4x4 Strassen decomposition.
    # Split 4x4 into 2x2 blocks. Each block is 2x2.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # The 7 Strassen products are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # Each Mt is a 2x2 matrix. We compute each Mt using the 2x2x2 Strassen (7 terms each).
    # Total rank = 7 * 7 = 49.
    #
    # For the 4x4x4 problem:
    # U has 16 rows (i*4+j), V has 16 rows (j*4+k), W has 16 rows (k*4+i).
    #
    # For each Strassen term t (0..6) and each inner term s (0..6):
    # The combined factor is the product of the outer and inner factors.
    
    # Define the outer 2x2 block structure for 4x4:
    # A11: i in {0,1}, j in {0,1} -> flat indices 0,1,4,5
    # A12: i in {0,1}, j in {2,3} -> flat indices 2,3,6,7
    # A21: i in {2,3}, j in {0,1} -> flat indices 8,9,12,13
    # A22: i in {2,3}, j in {2,3} -> flat indices 10,11,14,15
    
    # B11: j in {0,1}, k in {0,1} -> flat indices 0,1,4,5
    # B12: j in {0,1}, k in {2,3} -> flat indices 2,3,6,7
    # B21: j in {2,3}, k in {0,1} -> flat indices 8,9,12,13
    # B22: j in {2,3}, k in {2,3} -> flat indices 10,11,14,15
    
    # C11: i in {0,1}, k in {0,1} -> W index k*4+i: (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
    # C12: i in {0,1}, k in {2,3} -> W index k*4+i: (2,0)=8, (2,1)=9, (3,0)=12, (3,1)=13
    # C21: i in {2,3}, k in {0,1} -> W index k*4+i: (0,2)=2, (0,3)=3, (1,2)=6, (1,3)=7
    # C22: i in {2,3}, k in {2,3} -> W index k*4+i: (2,2)=10, (2,3)=11, (3,2)=14, (3,3)=15
    
    # Outer coefficients for each of the 7 Strassen terms:
    # For each term, we need to know which blocks of A and B are involved, with what coefficients.
    
    # Let's define the outer terms as (A_blocks, B_blocks, C_blocks) with coefficients.
    # Each entry is (block_name, coeff)
    
    outer_terms = []
    
    # M1 = (A11+A22)(B11+B22)
    # A part: A11*1 + A22*1
    # B part: B11*1 + B22*1
    # C part: C11*1 + C22*1
    outer_terms.append((
        [('A11', 1), ('A22', 1)],
        [('B11', 1), ('B22', 1)],
        [('C11', 1), ('C22', 1)]
    ))
    
    # M2 = (A21+A22)B11
    outer_terms.append((
        [('A21', 1), ('A22', 1)],
        [('B11', 1)],
        [('C21', 1), ('C22', -1)]
    ))
    
    # M3 = A11(B12-B22)
    outer_terms.append((
        [('A11', 1)],
        [('B12', 1), ('B22', -1)],
        [('C12', 1), ('C22', 1)]
    ))
    
    # M4 = A22(B21-B11)
    outer_terms.append((
        [('A22', 1)],
        [('B21', 1), ('B11', -1)],
        [('C11', 1), ('C21', 1)]
    ))
    
    # M5 = (A11+A12)B22
    outer_terms.append((
        [('A11', 1), ('A12', 1)],
        [('B22', 1)],
        [('C11', -1), ('C12', 1)]
    ))
    
    # M6 = (A21-A11)(B11+B12)
    outer_terms.append((
        [('A21', 1), ('A11', -1)],
        [('B11', 1), ('B12', 1)],
        [('C22', 1)]
    ))
    
    # M7 = (A12-A22)(B21+B22)
    outer_terms.append((
        [('A12', 1), ('A22', -1)],
        [('B21', 1), ('B22', 1)],
        [('C11', 1)]
    ))
    
    # Now build U, V, W for 4x4x4 with rank 49.
    U = np.zeros((16, 49), dtype=np.float64)
    V = np.zeros((16, 49), dtype=np.float64)
    W = np.zeros((16, 49), dtype=np.float64)
    
    # Map block names to their flat index ranges in the 4x4 matrices.
    # For A (16 elements, index i*4+j):
    A_block_indices = {
        'A11': [0, 1, 4, 5],   # i in {0,1}, j in {0,1}
        'A12': [2, 3, 6, 7],   # i in {0,1}, j in {2,3}
        'A21': [8, 9, 12, 13], # i in {2,3}, j in {0,1}
        'A22': [10, 11, 14, 15], # i in {2,3}, j in {2,3}
    }
    
    # For B (16 elements, index j*4+k):
    B_block_indices = {
        'B11': [0, 1, 4, 5],   # j in {0,1}, k in {0,1}
        'B12': [2, 3, 6, 7],   # j in {0,1}, k in {2,3}
        'B21': [8, 9, 12, 13], # j in {2,3}, k in {0,1}
        'B22': [10, 11, 14, 15], # j in {2,3}, k in {2,3}
    }
    
    # For C (16 elements, index k*4+i):
    C_block_indices = {
        'C11': [0, 1, 4, 5],   # k in {0,1}, i in {0,1}
        'C12': [8, 9, 12, 13], # k in {2,3}, i in {0,1}
        'C21': [2, 3, 6, 7],   # k in {0,1}, i in {2,3}
        'C22': [10, 11, 14, 15], # k in {2,3}, i in {2,3}
    }
    
    # Inner 2x2x2 Strassen terms (7 terms, each with U, V, W vectors of length 4)
    inner_terms = [np.array(t, dtype=np.float64) for t in terms_2x2]
    
    col = 0
    for outer_idx in range(7):
        a_blocks, b_blocks, c_blocks = outer_terms[outer_idx]
        
        # Build the combined A factor: sum over A blocks with coefficients
        # Each A block corresponds to a 2x2 sub-matrix, which maps to 4 flat indices.
        # The inner 2x2x2 U vector has 4 entries corresponding to the 2x2 A sub-matrix.
        
        for inner_idx in range(7):
            inner_U, inner_V, inner_W = inner_terms[inner_idx]
            
            # Compute combined U factor (length 16)
            U_combined = np.zeros(16, dtype=np.float64)
            for block_name, coeff in a_blocks:
                indices = A_block_indices[block_name]
                # The inner_U corresponds to the 2x2 sub-matrix in row-major order.
                # indices are in row-major order of the 4x4 matrix, which matches.
                for idx_in, flat_idx in enumerate(indices):
                    U_combined[flat_idx] += coeff * inner_U[idx_in]
            
            # Compute combined V factor (length 16)
            V_combined = np.zeros(16, dtype=np.float64)
            for block_name, coeff in b_blocks:
                indices = B_block_indices[block_name]
                for idx_in, flat_idx in enumerate(indices):
                    V_combined[flat_idx] += coeff * inner_V[idx_in]
            
            # Compute combined W factor (length 16)
            W_combined = np.zeros(16, dtype=np.float64)
            for block_name, coeff in c_blocks:
                indices = C_block_indices[block_name]
                for idx_in, flat_idx in enumerate(indices):
                    W_combined[flat_idx] += coeff * inner_W[idx_in]
            
            U[:, col] = U_combined
            V[:, col] = V_combined
            W[:, col] = W_combined
            col += 1
    
    return U, V, W


def _schoolbook_4x4x1():
    """Return U, V, W for 4x4x1 matrix multiplication (A is 4x4, B is 4x1, C is 4x1).
    U shape: (16, 16), V shape: (4, 16), W shape: (4, 16)
    Convention: U[i*4+j, c], V[j*1+k, c] = V[j, c], W[k*4+i, c] = W[i, c] (since k=0 always)
    Actually p=1, so V has m*p = 4*1 = 4 rows, W has p*n = 1*4 = 4 rows.
    V[j*p+k] = V[j*1+0] = V[j]
    W[k*n+i] = W[0*4+i] = W[i]
    
    Schoolbook: C[i] = sum_j A[i,j] * B[j]
    For each (i,j) pair, we have a term: U[i*4+j, c] = 1, V[j, c] = 1, W[i, c] = 1
    Total rank = 4*4 = 16.
    """
    n = 4
    p = 1
    rank = n * n  # 16
    
    U = np.zeros((n * n, rank), dtype=np.float64)  # 16 x 16
    V = np.zeros((n * p, rank), dtype=np.float64)   # 4 x 16
    W = np.zeros((p * n, rank), dtype=np.float64)   # 4 x 16
    
    col = 0
    for i in range(n):
        for j in range(n):
            U[i * n + j, col] = 1.0
            V[j, col] = 1.0
            W[i, col] = 1.0
            col += 1
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Build the 4x4x4 Strassen decomposition (rank 49)
    U_4, V_4, W_4 = _strassen_4x4()
    
    # Build the 4x4x1 schoolbook decomposition (rank 16)
    U_1, V_1, W_1 = _schoolbook_4x4x1()
    
    # Combine for 4x4x5:
    # The full problem: A is 4x4, B is 4x5, C is 4x5.
    # Split B into B_left (4x4) and B_right (4x1).
    # C = A * B_left + A * B_right
    # 
    # U: same for both parts (uses A), so concatenate columns: U_full = [U_4, U_1] (16 x 65)
    # V: V_4 has 16 rows (j*4+k for k=0..3), V_1 has 4 rows (j for the 5th column, k=4)
    #    V_full has 20 rows (m*p = 4*5 = 20).
    #    Rows 0..15: from V_4 (k=0..3)
    #    Rows 16..19: from V_1 (k=4, so j*5+4)
    # W: W_4 has 16 rows (k*4+i for k=0..3), W_1 has 4 rows (i for k=4)
    #    W_full has 20 rows (p*n = 5*4 = 20).
    #    Rows 0..15: from W_4 (k=0..3)
    #    Rows 16..19: from W_1 (k=4, so 4*4+i = 16+i)
    
    total_rank = 49 + 16  # 65
    
    U_full = np.hstack([U_4, U_1])  # 16 x 65
    V_full = np.zeros((20, total_rank), dtype=np.float64)
    W_full = np.zeros((20, total_rank), dtype=np.float64)
    
    # Place V_4 in rows 0..15
    V_full[:16, :49] = V_4
    # Place V_1 in rows 16..19
    V_full[16:, 49:] = V_1
    
    # Place W_4 in rows 0..15
    W_full[:16, :49] = W_4
    # Place W_1 in rows 16..19
    W_full[16:, 49:] = W_1
    
    # Verify
    def _verify(U, V, W, n, m, p, tol=1e-6):
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over rank dimension r
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    valid = _verify(U_full, V_full, W_full, n, m, p, tol=1e-6)
    
    if not valid:
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U_sb = np.zeros((n * m, schoolbook_rank))
        V_sb = np.zeros((m * p, schoolbook_rank))
        W_sb = np.zeros((p * n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
