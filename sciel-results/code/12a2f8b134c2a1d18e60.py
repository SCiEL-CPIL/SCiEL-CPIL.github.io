import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Block-Strassen approach
    # A (4x4) split into 2x2 blocks of 2x2: A[i_block][j_block]
    # B (4x8) split into 2x4 blocks of 2x2: B[i_block][j_block]  
    # C (4x8) split into 2x4 blocks of 2x2: C[i_block][j_block]
    #
    # C[i_block][j_block] = A[i_block][0]*B[0][j_block] + A[i_block][1]*B[1][j_block]
    #
    # Each 2x2 * 2x2 product uses Strassen (rank 7)
    # Total: 2 * 4 * 2 * 7 = 112
    
    # Strassen coefficients for 2x2 * 2x2:
    # M1 = (A11+A12)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B11-B12)
    # M5 = (A11+A22)*(B21+B22)
    # M6 = (A11-A21)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M5
    # C22 = M1 - M3 + M2 - M4
    
    # For each 2x2*2x2 product, we have 7 terms
    # Each term is a product of a linear combination of A elements and a linear combination of B elements
    
    # Let's define the Strassen terms for a generic 2x2*2x2:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # 
    # Term 1: (a00+a01) * (b00+b11)
    # Term 2: (a10+a11) * b00
    # Term 3: a00 * (b01-b11)
    # Term 4: a11 * (b00-b01)
    # Term 5: (a00+a11) * (b10+b11)
    # Term 6: (a00-a10) * (b00+b01)
    # Term 7: (a01-a11) * (b10+b11)
    #
    # Result:
    # c00 = t1 + t4 - t5 + t7
    # c01 = t3 + t5
    # c10 = t2 + t5
    # c11 = t1 - t3 + t2 - t4
    
    # Now, for our block structure:
    # A has 4 row-blocks (2 per 2x2 block) and 4 col-blocks
    # Actually, A is 4x4, split into 2x2 blocks of size 2x2
    # A[ib][jb] where ib in {0,1}, jb in {0,1}, each is 2x2
    
    # B is 4x8, split into 2x4 blocks of size 2x2
    # B[ib][jb] where ib in {0,1}, jb in {0,1,2,3}, each is 2x2
    
    # C is 4x8, split into 2x4 blocks of size 2x2
    # C[ib][jb] where ib in {0,1}, jb in {0,1,2,3}, each is 2x2
    
    # C[ib][jb] = A[ib][0]*B[0][jb] + A[ib][1]*B[1][jb]
    
    # For each (ib, jb), we compute two 2x2*2x2 products using Strassen
    
    rank = 2 * 4 * 2 * 7  # = 112
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Index mapping:
    # A[i][k] -> U index: i*m + k = i*4 + k
    # B[k][j] -> V index: k*p + j = k*8 + j
    # C[i][j] -> W index: j*n + i = j*4 + i (trace convention W[k*n+i,r] where k=j, i=i)
    
    # Strassen term definitions for 2x2*2x2:
    # Each term has coefficients for A elements (4) and B elements (4)
    # A elements: [a00, a01, a10, a11]
    # B elements: [b00, b01, b10, b11]
    
    strassen_terms = [
        # (A coeffs, B coeffs)
        ([1, 1, 0, 0], [1, 0, 0, 1]),   # t1: (a00+a01)*(b00+b11)
        ([0, 0, 1, 1], [1, 0, 0, 0]),   # t2: (a10+a11)*b00
        ([1, 0, 0, 0], [0, 1, 0, -1]),  # t3: a00*(b01-b11)
        ([0, 0, 0, 1], [1, -1, 0, 0]),  # t4: a11*(b00-b01)
        ([1, 0, 0, 1], [0, 0, 1, 1]),   # t5: (a00+a11)*(b10+b11)
        ([1, 0, -1, 0], [1, 1, 0, 0]),  # t6: (a00-a10)*(b00+b01)
        ([0, 1, 0, -1], [0, 0, 1, 1]),  # t7: (a01-a11)*(b10+b11)
    ]
    
    # Result coefficients: which terms contribute to each output element
    # c00 = t1 + t4 - t5 + t7
    # c01 = t3 + t5
    # c10 = t2 + t5
    # c11 = t1 - t3 + t2 - t4
    result_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],   # c00
        [0, 0, 1, 0, 1, 0, 0],    # c01
        [0, 1, 0, 0, 1, 0, 0],    # c10
        [1, 0, -1, -1, 0, 0, 0],  # c11
    ]
    
    r = 0
    for ib in range(2):  # row block of A/C
        for jb in range(4):  # col block of B/C
            # C[ib][jb] = A[ib][0]*B[0][jb] + A[ib][1]*B[1][jb]
            for prod_idx in range(2):  # 0: A[ib][0]*B[0][jb], 1: A[ib][1]*B[1][jb]
                # A block: A[ib][prod_idx]
                # A[ib][prod_idx] is a 2x2 matrix
                # Row offset in A: ib*2
                # Col offset in A: prod_idx*2
                a_row_off = ib * 2
                a_col_off = prod_idx * 2
                
                # B block: B[prod_idx][jb]
                # B[prod_idx][jb] is a 2x2 matrix
                # Row offset in B: prod_idx*2
                # Col offset in B: jb*2
                b_row_off = prod_idx * 2
                b_col_off = jb * 2
                
                # C block: C[ib][jb]
                # Row offset in C: ib*2
                # Col offset in C: jb*2
                c_row_off = ib * 2
                c_col_off = jb * 2
                
                for t in range(7):
                    a_coeffs, b_coeffs = strassen_terms[t]
                    
                    for a_r in range(2):
                        for a_c in range(2):
                            a_elem = a_coeffs[a_r * 2 + a_c]
                            if a_elem != 0:
                                a_i = a_row_off + a_r
                                a_k = a_col_off + a_c
                                U[a_i * m + a_k, r] += a_elem
                    
                    for b_r in range(2):
                        for b_c in range(2):
                            b_elem = b_coeffs[b_r * 2 + b_c]
                            if b_elem != 0:
                                b_k = b_row_off + b_r
                                b_j = b_col_off + b_c
                                V[b_k * p + b_j, r] += b_elem
                    
                    # W coefficients
                    for c_r in range(2):
                        for c_c in range(2):
                            c_coeff = result_coeffs[c_r * 2 + c_c][t]
                            if c_coeff != 0:
                                c_i = c_row_off + c_r
                                c_j = c_col_off + c_c
                                W[c_j * n + c_i, r] += c_coeff
                    
                    r += 1
    
    assert r == rank, f"r={r}, rank={rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
