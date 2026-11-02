import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Total rank calculation:
    # 1x1x1: 1
    # 1x4x1 (dot): 4
    # 1x1x4 (scale row): 1
    # 4x1x1 (scale col): 1
    # 4x1x4 (outer): 1
    # 4x4x4 (Strassen): 49
    total_rank = 1 + 4 + 1 + 1 + 1 + 49
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a term
    def add_term(u_rows, v_rows, w_rows, u_vals, v_vals, w_vals):
        nonlocal col_offset
        # u_rows: list of indices in U
        # v_rows: list of indices in V
        # w_rows: list of indices in W
        # u_vals: list of values
        # v_vals: list of values
        # w_vals: list of values
        
        # This helper is too generic. Let's do it manually for each block.
        pass

    # --- Term 1: A11 * B11 (1x1 * 1x1 -> 1x1) ---
    # A11 is A[0,0]. B11 is B[0,0]. C11 is C[0,0].
    # U[0, col] = 1
    # V[0, col] = 1
    # W[0, col] = 1
    U[0][col_offset] = 1
    V[0][col_offset] = 1
    W[0][col_offset] = 1
    col_offset += 1
    
    # --- Term 2: A12 * B21 (1x4 * 4x1 -> 1x1) ---
    # A12 is A[0, 1..4]. B21 is B[1..4, 0]. C11 is C[0,0].
    # C11 += sum_{k=1..4} A[0,k] * B[k,0]
    # This is dot product. Rank 4.
    for k in range(1, 5):
        # A index: 0 * 5 + k = k
        # B index: k * 5 + 0 = 5*k
        # C index: 0
        U[k][col_offset] = 1
        V[5*k][col_offset] = 1
        W[0][col_offset] = 1
        col_offset += 1
        
    # --- Term 3: A11 * B12 (1x1 * 1x4 -> 1x4) ---
    # A11 is A[0,0]. B12 is B[0, 1..4]. C12 is C[0, 1..4].
    # C[0,j] += A[0,0] * B[0,j]
    # Rank 1.
    # U[0, col] = 1
    # V[0, col] = 1? No, V needs to extract B[0,j] for all j.
    # V index for B[0,j] is 0*5 + j = j.
    # We need V[j, col] = 1 for j=1..4.
    # W index for C[0,j] is 0*5 + j = j.
    # We need W[j, col] = 1 for j=1..4.
    U[0][col_offset] = 1
    for j in range(1, 5):
        V[j][col_offset] = 1
        W[j][col_offset] = 1
    col_offset += 1
    
    # --- Term 4: A21 * B11 (4x1 * 1x1 -> 4x1) ---
    # A21 is A[1..4, 0]. B11 is B[0,0]. C21 is C[1..4, 0].
    # C[i,0] += A[i,0] * B[0,0]
    # Rank 1.
    # U index for A[i,0] is i*5 + 0 = 5*i.
    # U[5*i, col] = 1 for i=1..4.
    # V[0, col] = 1.
    # W index for C[i,0] is i*5 + 0 = 5*i.
    # W[5*i, col] = 1 for i=1..4.
    for i in range(1, 5):
        U[5*i][col_offset] = 1
        W[5*i][col_offset] = 1
    V[0][col_offset] = 1
    col_offset += 1
    
    # --- Term 5: A21 * B12 (4x1 * 1x4 -> 4x4) ---
    # A21 is A[1..4, 0]. B12 is B[0, 1..4]. C22 is C[1..4, 1..4].
    # C[i,j] += A[i,0] * B[0,j]
    # Rank 1.
    # U[5*i, col] = 1 for i=1..4.
    # V[j, col] = 1 for j=1..4.
    # W[5*i + j, col] = 1 for i=1..4, j=1..4.
    for i in range(1, 5):
        U[5*i][col_offset] = 1
    for j in range(1, 5):
        V[j][col_offset] = 1
    for i in range(1, 5):
        for j in range(1, 5):
            W[5*i + j][col_offset] = 1
    col_offset += 1
    
    # --- Term 6: A22 * B22 (4x4 * 4x4 -> 4x4) ---
    # Use Strassen for 4x4. Rank 49.
    # We need to construct the 49 columns for this part.
    
    # First, define Strassen for 2x2.
    # U22, V22, W22 shapes (4, 7), (4, 7), (4, 7)
    U22 = [[0]*7 for _ in range(4)]
    V22 = [[0]*7 for _ in range(4)]
    W22 = [[0]*7 for _ in range(4)]
    
    # Indices in 2x2:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1 = (A00+A11)(B00+B11)
    U22[0][0] = 1; U22[3][0] = 1
    V22[0][0] = 1; V22[3][0] = 1
    W22[0][0] = 1; W22[3][0] = 1 # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)B00
    U22[2][1] = 1; U22[3][1] = 1
    V22[0][1] = 1
    W22[2][1] = 1; W22[3][1] = 1 # C10 += M2, C11 += -M2? No.
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # Let's map carefully.
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will fill U22, V22, W22 for these 7 terms.
    
    # Term 1: M1
    U22[0][0] += 1; U22[3][0] += 1
    V22[0][0] += 1; V22[3][0] += 1
    W22[0][0] += 1; W22[3][0] += 1
    
    # Term 2: M2
    U22[2][1] += 1; U22[3][1] += 1
    V22[0][1] += 1
    W22[2][1] += 1; W22[3][1] += 1
    
    # Term 3: M3
    U22[0][2] += 1
    V22[1][2] += 1; V22[3][2] -= 1
    W22[1][2] += 1; W22[3][2] += 1
    
    # Term 4: M4
    U22[3][3] += 1
    V22[2][3] += 1; V22[0][3] -= 1
    W22[0][3] += 1; W22[2][3] += 1
    
    # Term 5: M5
    U22[0][4] += 1; U22[1][4] += 1
    V22[3][4] += 1
    W22[0][4] -= 1; W22[1][4] += 1
    
    # Term 6: M6
    U22[2][5] += 1; U22[0][5] -= 1
    V22[0][5] += 1; V22[1][5] += 1
    W22[3][5] += 1
    
    # Term 7: M7
    U22[1][6] += 1; U22[3][6] -= 1
    V22[2][6] += 1; V22[3][6] += 1
    W22[0][6] += 1
    
    # Now, construct 4x4 Strassen using 2x2 Strassen primitives.
    # The 4x4 matrix A is divided into 4 2x2 blocks: A00, A01, A10, A11.
    # Similarly for B and C.
    # The 7 "scalar" multiplications in the 2x2 Strassen formula are replaced by 2x2 matrix multiplications.
    # Each 2x2 multiplication has rank 7 (using the U22, V22, W22 defined above).
    # Total rank 7 * 7 = 49.
    
    # We need to map the 2x2 Strassen inputs (which are linear combinations of 2x2 blocks)
    # to the 4x4 indices.
    
    # Let's define the mapping of 2x2 block indices to 4x4 indices.
    # A00 (2x2) corresponds to A[0:2, 0:2] in 4x4.
    # A01 corresponds to A[0:2, 2:4].
    # A10 corresponds to A[2:4, 0:2].
    # A11 corresponds to A[2:4, 2:4].
    
    # We will iterate through the 7 terms of the 2x2 Strassen.
    # For each term t (0..6), we have:
    # U22[i, t] gives the coefficient for A_block_i in the input to M_t.
    # V22[j, t] gives the coefficient for B_block_j in the input to M_t.
    # W22[k, t] gives the coefficient for C_block_k in the output from M_t.
    
    # For each term t, and for each non-zero entry in U22[:, t], V22[:, t], W22[:, t],
    # we need to "expand" the 2x2 block operation into the 4x4 indices using the U22, V22, W22 of the 2x2 Strassen.
    
    # This is essentially a Kronecker product of the 2x2 Strassen tensor with itself?
    # Yes.
    
    # Let's build U44, V44, W44.
    U44 = [[0]*49 for _ in range(16)]
    V44 = [[0]*49 for _ in range(16)]
    W44 = [[0]*49 for _ in range(16)]
    
    for t in range(7):
        # For each term t in the outer Strassen, we perform a 2x2 matrix multiplication.
        # The inputs to this 2x2 multiplication are linear combinations of the 2x2 blocks.
        # The output is a 2x2 matrix which is added to the corresponding 2x2 blocks of C.
        
        # We need to combine the "outer" Strassen structure (t) with the "inner" Strassen structure (s).
        # The rank index for the 4x4 case is t * 7 + s.
        
        for s in range(7):
            col_idx = t * 7 + s
            
            # The "inner" multiplication s computes a part of the product of two 2x2 matrices.
            # Let A_in and B_in be the 2x2 matrices being multiplied.
            # A_in[i] = sum_{block_i} coeff_u22[i,t] * A_block_i
            # B_in[j] = sum_{block_j} coeff_v22[j,t] * B_block_j
            # C_out[k] += coeff_w22[k,t] * (A_in * B_in)_k
            
            # The inner Strassen tensor U22[i, s], V22[j, s], W22[k, s] computes (A_in * B_in)_k.
            # So, for each i, j, k where U22[i, s], V22[j, s], W22[k, s] are non-zero:
            # We need to map A_block_i (which is a 2x2 matrix) to the 4x4 indices.
            # A_block_i is formed by summing 2x2 blocks of A.
            # Wait, A_block_i in the outer Strassen is a 2x2 matrix.
            # In the 4x4 case, A_block_i is one of the four 2x2 quadrants.
            # So A_in[i] (a 2x2 matrix) is a linear combination of the 2x2 quadrants of A.
            
            # Let's trace:
            # U44[row, col_idx] should be the coefficient of A[row] in the term.
            # A[row] is an element of the 4x4 matrix.
            # The element A[row] belongs to a specific 2x2 quadrant Q_A.
            # The coefficient of Q_A in A_in is U22[Q_A_idx, t].
            # The coefficient of the element within Q_A in the inner multiplication is U22[element_idx_in_2x2, s].
            
            # So U44[row, col_idx] = U22[Q_A_idx, t] * U22[element_idx_in_2x2, s].
            
            # Similarly for V and W.
            
            # Map 4x4 index to (Quadrant, Element in 2x2)
            def map_4x4_to_2x2(idx):
                r = idx // 4
                c = idx % 4
                q_r = r // 2
                q_c = c // 2
                q_idx = q_r * 2 + q_c
                e_r = r % 2
                e_c = c % 2
                e_idx = e_r * 2 + e_c
                return q_idx, e_idx
            
            for r in range(16):
                q_idx_A, e_idx_A = map_4x4_to_2x2(r)
                coeff_A = U22[q_idx_A, t] * U22[e_idx_A, s]
                if coeff_A != 0:
                    U44[r][col_idx] = coeff_A
                    
            for r in range(16):
                q_idx_B, e_idx_B = map_4x4_to_2x2(r)
                coeff_B = V22[q_idx_B, t] * V22[e_idx_B, s]
                if coeff_B != 0:
                    V44[r][col_idx] = coeff_B
                    
            for r in range(16):
                q_idx_C, e_idx_C = map_4x4_to_2x2(r)
                coeff_C = W22[q_idx_C, t] * W22[e_idx_C, s]
                if coeff_C != 0:
                    W44[r][col_idx] = coeff_C

    # Now add U44, V44, W44 to the global U, V, W.
    # A22 is A[1:5, 1:5].
    # Map 4x4 index (0..15) to 5x5 index (1..4, 1..4).
    # 4x4 index r corresponds to 5x5 index (1 + r//4) * 5 + (1 + r%4).
    
    for r in range(16):
        row_5x5 = (1 + r // 4) * 5 + (1 + r % 4)
        for c in range(49):
            U[row_5x5][col_offset + c] = U44[r][c]
            V[row_5x5][col_offset + c] = V44[r][c]
            W[row_5x5][col_offset + c] = W44[r][c]
            
    col_offset += 49
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
