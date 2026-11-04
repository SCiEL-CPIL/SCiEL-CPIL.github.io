"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Strassen's 7 multiplication formulas for 2x2x2
    # M1 = A00 * B00
    # M2 = (A10 + A00) * B00
    # M3 = A01 * B11
    # M4 = (A11 + A01) * B11
    # M5 = (A00 + A11) * (B00 + B11)
    # M6 = (A10 - A11) * (B10 + B11)
    # M7 = (A01 - A00) * (B00 + B10)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M4
    
    # For 4x4x4: apply Strassen to 2x2 blocks
    # A is 4x4, split into 2x2 blocks A00, A01, A10, A11 (each 2x2)
    # B is 4x4, split into 2x2 blocks B00, B01, B10, B11 (each 2x2)
    # C is 4x4, split into 2x2 blocks C00, C01, C10, C11 (each 2x2)
    
    # For each Strassen term, we need 7 scalar multiplications (2x2x2)
    # Total: 7 * 7 = 49 terms for the 4x4x4 part
    
    # Strassen block coefficients for A (4 values: A00, A01, A10, A11)
    # and B (4 values: B00, B01, B10, B11)
    strassen_blocks = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # Each is a dict mapping block index to coefficient
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),    # M1
        ({(1,0): 1, (0,0): 1}, {(0,0): 1}, {(1,0): 1}),  # M2
        ({(0,1): 1}, {(1,1): 1}, {(0,1): 1}),    # M3
        ({(1,1): 1, (0,1): 1}, {(1,1): 1}, {(1,1): 1}),  # M4
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): -1}),  # M5
        ({(1,0): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(1,0): 1}),  # M6
        ({(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(0,0): 1}),  # M7
    ]
    
    # Strassen scalar coefficients within each 2x2 block
    # For 2x2x2 Strassen: 7 terms
    # A and B are 2x2, indexed (row, col)
    strassen_scalar = [
        # (A_coeffs, B_coeffs, C_coeffs) for 2x2 matrices
        # M1 = A[0,0]*B[0,0]
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),
        # M2 = (A[1,0]+A[0,0])*B[0,0]
        ({(1,0): 1, (0,0): 1}, {(0,0): 1}, {(1,0): 1}),
        # M3 = A[0,1]*B[1,1]
        ({(0,1): 1}, {(1,1): 1}, {(0,1): 1}),
        # M4 = (A[1,1]+A[0,1])*B[1,1]
        ({(1,1): 1, (0,1): 1}, {(1,1): 1}, {(1,1): 1}),
        # M5 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): -1}),
        # M6 = (A[1,0]-A[1,1])*(B[1,0]+B[1,1])
        ({(1,0): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(1,0): 1}),
        # M7 = (A[0,1]-A[0,0])*(B[0,0]+B[1,0])
        ({(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(0,0): 1}),
    ]
    
    # Build the 49 Strassen terms for 4x4x4
    U_terms = []  # Each is a dict or list of (index, coeff)
    V_terms = []
    W_terms = []
    
    for blk_idx, (A_blk, B_blk, C_blk) in enumerate(strassen_blocks):
        for scal_idx, (A_scal, B_scal, C_scal) in enumerate(strassen_scalar):
            # Build U vector (16 entries for 4x4 A)
            u_entry = {}
            for (bi, bj), c_blk in A_blk.items():
                for (si, sj), c_scal in A_scal.items():
                    # Global A index: (bi*2 + si) * 4 + (bj*2 + sj)
                    a_idx = (bi * 2 + si) * 4 + (bj * 2 + sj)
                    coeff = c_blk * c_scal
                    u_entry[a_idx] = u_entry.get(a_idx, 0) + coeff
            
            # Build V vector (28 entries for 4x7 B, but we only use first 4 cols)
            v_entry = {}
            for (bi, bj), c_blk in B_blk.items():
                for (si, sj), c_scal in B_scal.items():
                    # B is 4x7, first 4 columns. Index: (bi*2 + si) * 7 + (bj*2 + sj)
                    b_idx = (bi * 2 + si) * 7 + (bj * 2 + sj)
                    coeff = c_blk * c_scal
                    v_entry[b_idx] = v_entry.get(b_idx, 0) + coeff
            
            # Build W vector (28 entries for 7x4 C, but we only use first 4 rows)
            w_entry = {}
            for (ci, cj), c_blk in C_blk.items():
                for (si, sj), c_scal in C_scal.items():
                    # C is 7x4, first 4 rows. W index: (ci*2 + si) * 4 + (cj*2 + sj)
                    c_idx = (ci * 2 + si) * 4 + (cj * 2 + sj)
                    coeff = c_blk * c_scal
                    w_entry[c_idx] = w_entry.get(c_idx, 0) + coeff
            
            U_terms.append(u_entry)
            V_terms.append(v_entry)
            W_terms.append(w_entry)
    
    # Now add 48 schoolbook terms for 4x4x3 (last 3 columns of B, last 3 rows of C)
    for i in range(4):
        for j in range(4):
            for k in range(3):
                # A index: i * 4 + j
                a_idx = i * 4 + j
                # B index: j * 7 + (4 + k)
                b_idx = j * 7 + (4 + k)
                # C index: (4 + k) * 4 + i
                c_idx = (4 + k) * 4 + i
                
                U_terms.append({a_idx: 1})
                V_terms.append({b_idx: 1})
                W_terms.append({c_idx: 1})
    
    total_rank = len(U_terms)
    
    # Convert to list-of-lists format
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for r in range(total_rank):
        for idx, coeff in U_terms[r].items():
            U[idx][r] = coeff
        for idx, coeff in V_terms[r].items():
            V[idx][r] = coeff
        for idx, coeff in W_terms[r].items():
            W[idx][r] = coeff
    
    # Verify decomposition
    # C[i,k] = sum_j sum_r U[i*4+j, r] * V[j*7+k, r] * W[k*4+i, r]
    # Wait, the convention is W[k*n+i, r] where k is column index of C, i is row index
    # So C[i, k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify
    C_expected = {}
    for i in range(4):
        for k in range(7):
            val = 0
            for j in range(4):
                for r in range(total_rank):
                    val += U[i * 4 + j][r] * V[j * 7 + k][r] * W[k * 4 + i][r]
            C_expected[(i, k)] = val
    
    # For matrix multiplication C = A * B, C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor should have T[i,j,k] = 1 if this is the "product" tensor
    # Actually, the matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k
    # and the decomposition satisfies T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So we need: for all i, j, k: sum_r U[i*4+j, r] * V[j*7+k, r] * W[k*4+i, r] = 1
    
    # Let me verify a few
    errors = 0
    for i in range(4):
        for j in range(4):
            for k in range(7):
                s = 0
                for r in range(total_rank):
                    s += U[i * 4 + j][r] * V[j * 7 + k][r] * W[k * 4 + i][r]
                if s != 1:
                    errors += 1
    
    if errors > 0:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
