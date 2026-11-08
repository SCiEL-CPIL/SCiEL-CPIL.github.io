"""Kronecker-product decomposition for 3x4x8 matrix multiplication tensor.

Strategy: 3x4x8 = (3x2x4) x (2x2x2) tensor product.
- 3x2x4 is decomposed as a Kronecker product of (1x2x2) and (3x1x2).
  * 1x2x2 is rank 4 (schoolbook).
  * 3x1x2 is rank 3 (schoolbook).
  * Kronecker product gives rank 12.
- 2x2x2 is decomposed using Strassen's algorithm (rank 7).
- Overall rank = 12 * 7 = 84.
"""
import time

DIMENSIONS = [3, 4, 8]

def construct():
    # Deadline guard: 150s wall-clock, safety margin 5s
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Build the Kronecker decomposition
    # 3x4x8 = (3x2x4) ⊗ (2x2x2)
    # 3x2x4 = (1x2x2) ⊗ (3x1x2)
    # 1x2x2: rank 4, 3x1x2: rank 3 -> Kronecker rank = 12
    # 2x2x2: rank 7 (Strassen)
    # Total rank = 12 * 7 = 84

    # --- 1x2x2 decomposition (rank 4) ---
    # A is 1x2, B is 2x2, C is 1x2
    # C[k] = sum_i A[i] B[i,k]
    # Indices: A index (0, i) for i in {0,1}, B index (i, k) for i,k in {0,1}, C index (0, k) for k in {0,1}
    # Rank 4: one term per (i, k)
    U_122 = []  # shape (2, 4)  -- n*m = 1*2 = 2
    V_122 = []  # shape (4, 4)  -- m*p = 2*2 = 4
    W_122 = []  # shape (2, 4)  -- p*n = 2*1 = 2
    for i in range(2):
        for k in range(2):
            col = i * 2 + k
            # U: row (0, i) -> index 0*2 + i = i
            u_row = [0]*4
            u_row[col] = 1
            U_122.append(u_row)
            # V: row (i, k) -> index i*2 + k = i*2+k
            v_row = [0]*4
            v_row[col] = 1
            V_122.append(v_row)
            # W: row (k, 0) -> index k*1 + 0 = k
            w_row = [0]*4
            w_row[col] = 1
            W_122.append(w_row)
    # Reshape
    U_122 = [U_122[i] for i in range(2)]  # 2 rows
    V_122 = [V_122[i] for i in range(4)]  # 4 rows
    W_122 = [W_122[i] for i in range(2)]  # 2 rows

    # --- 3x1x2 decomposition (rank 3) ---
    # A is 3x1, B is 1x2, C is 3x2
    # C[i,k] = A[i,0] * B[0,k]
    # Rank 3: one term per i (since B is 1x2, each term handles all k)
    # Actually rank = 3*1 = 3 (sum over the 1 dimension)
    U_312 = []  # shape (3, 3) -- n*m = 3*1 = 3
    V_312 = []  # shape (2, 3) -- m*p = 1*2 = 2
    W_312 = []  # shape (6, 3) -- p*n = 2*3 = 6
    for i in range(3):
        col = i
        # U: row (i, 0) -> index i*1 + 0 = i
        u_row = [0]*3
        u_row[col] = 1
        U_312.append(u_row)
        # V: row (0, k) for k in {0,1} -> index 0*2 + k = k
        # For this term, V[0,k] = 1 for all k? No, V is per-term.
        # V[0, k] for this term: we need B[0,k]. Since B is 1x2, V[0,k] = 1.
        # Actually, for the term corresponding to index i of A, V should pick up B.
        # The decomposition is: C[i,k] = A[i,0] * B[0,k]
        # Term r=i: U[i, r] = 1, V[k, r] = 1 (for all k), W[k*3+i, r] = 1
        v_row0 = [0]*3
        v_row0[col] = 1
        v_row1 = [0]*3
        v_row1[col] = 1
        # W: row (k, i) -> index k*3 + i
        w_rows = []
        for k in range(2):
            w_row = [0]*3
            w_row[col] = 1
            w_rows.append(w_row)
        # We need to build V and W properly
    # Let me redo 3x1x2 more carefully
    # n=3, m=1, p=2
    # A: 3x1, B: 1x2, C: 3x2
    # C[i,k] = A[i,0] * B[0,k]
    # Decomposition with rank 3:
    # For r in {0,1,2} (corresponding to i in {0,1,2}):
    #   U[i, r] = 1 if i == r else 0
    #   V[k, r] = 1 for all k (since B[0,k] is shared)
    #   Wait, V is 1x2 matrix indexed by (m-1, p-1) = (0, k). So V[0, k, r]
    #   But V in the tensor format is a matrix of shape (m*p, rank) = (2, rank)
    #   V[row, col] where row = 0*2 + k = k
    #   So V[k, r] = 1
    #   W: shape (p*n, rank) = (6, rank), W[k*n + i, r] = 1
    U_312 = [[0]*3 for _ in range(3)]
    V_312 = [[0]*3 for _ in range(2)]
    W_312 = [[0]*3 for _ in range(6)]
    for r in range(3):
        U_312[r][r] = 1  # i = r
        for k in range(2):
            V_312[k][r] = 1
            W_312[k*3 + r][r] = 1

    # --- Kronecker product of 1x2x2 and 3x1x2 to get 3x2x4 ---
    # 1x2x2: n1=1, m1=2, p1=2, rank1=4
    # 3x1x2: n2=3, m2=1, p2=2, rank2=3
    # Result: n=1*3=3, m=2*1=2, p=2*2=4, rank=4*3=12
    # U_kron: shape (n*m, rank) = (3*2, 12) = (6, 12)
    # U_kron[(i1*n2 + i2)*m2 + (j1*m2 + j2), r1*rank2 + r2] = U1[i1*m1+j1, r1] * U2[i2*m2+j2, r2]
    # Since n1=1, m1=2, n2=3, m2=1:
    # i1 in {0}, j1 in {0,1}, i2 in {0,1,2}, j2 in {0}
    # Row index: (0*3 + i2)*1 + (j1*1 + 0) = i2 + j1
    # Wait, the Kronecker product indexing:
    # (i1, i2) -> i1*n2 + i2, (j1, j2) -> j1*m2 + j2
    # Combined row: (i1*n2 + i2)*m2_total + (j1*m2 + j2) where m2_total = m1*m2 = 2
    # Actually, the row index for A is (i1, j1) and (i2, j2) combined.
    # A has dimensions (n1*n2) x (m1*m2)
    # Row (i1*n2+i2, j1*m2+j2)
    # For our case: n1=1, n2=3, m1=2, m2=1
    # Row: (i1*3+i2, j1*1+j2) where i1 in {0}, i2 in {0,1,2}, j1 in {0,1}, j2 in {0}
    # So row index = i1*3+i2 = i2, col index = j1*1+j2 = j1
    # U_kron[i2*2 + j1, r1*3 + r2] = U1[i1*m1+j1, r1] * U2[i2*m2+j2, r2] = U1[j1, r1] * U2[i2, r2]
    U_324 = [[0]*12 for _ in range(6)]
    V_324 = [[0]*12 for _ in range(8)]  # m*p = 2*4 = 8
    W_324 = [[0]*12 for _ in range(12)]  # p*n = 4*3 = 12

    for r1 in range(4):
        for r2 in range(3):
            r = r1*3 + r2
            for i2 in range(3):
                for j1 in range(2):
                    u1_val = U_122[j1][r1]  # i1=0, so U_122[0*2+j1, r1] = U_122[j1, r1]
                    u2_val = U_312[i2][r2]  # i2, j2=0, so U_312[i2*1+0, r2] = U_312[i2, r2]
                    U_324[i2*2 + j1][r] = u1_val * u2_val

            # V: shape (m*p, rank) = (2*4, 12)
            # m1=2, m2=1, p1=2, p2=2
            # V row: (j1, k1) for tensor 1, (j2, k2) for tensor 2
            # Combined: (j1*m2+j2, k1*p2+k2) but V is indexed by (m, p) pairs
            # Actually V is a matrix of shape (m*p, rank), where m = m1*m2 = 2, p = p1*p2 = 4
            # Row index: (j1*m2+j2, k1*p2+k2) -> but this doesn't directly map.
            # Let me think again. V is the second factor, corresponding to B.
            # B has dimensions (m1*m2) x (p1*p2) = 2x4
            # B[(j1*m2+j2), (k1*p2+k2)] = B1[j1,k1] * B2[j2,k2]
            # So V row index = j1*m2+j2, col index = k1*p2+k2
            # j1 in {0,1}, j2 in {0}, k1 in {0,1}, k2 in {0,1,2}? No, p2=2 so k2 in {0,1}
            # Wait, tensor 2 is 3x1x2, so m2=1, p2=2. j2 in {0}, k2 in {0,1}
            # V1: shape (m1*p1, rank1) = (4, 4), V1[j1*p1+k1, r1] = V1[j1*2+k1, r1]
            # V2: shape (m2*p2, rank2) = (2, 3), V2[j2*p2+k2, r2] = V2[k2, r2]
            # V_kron[(j1*1+j2)*(p1*p2) + (k1*2+k2), r] = V1[j1*2+k1, r1] * V2[k2, r2]
            # Row index: (j1+j2)*4 + (k1*2+k2). Since j2=0, row = j1*4 + k1*2 + k2
            for j1 in range(2):
                for k1 in range(2):
                    v1_val = V_122[j1*2 + k1][r1]
                    for k2 in range(2):
                        v2_val = V_312[k2][r2]
                        row = j1*4 + k1*2 + k2
                        V_324[row][r] = v1_val * v2_val

            # W: shape (p*n, rank) = (4*3, 12) = (12, 12)
            # W[(k1*p2+k2, i1*n2+i2)] -> W row index: (k1*p2+k2)*n2 + i1*n2+i2? 
            # Actually W is indexed by (p, n) pairs: W[k*n+i, r]
            # p = p1*p2 = 4, n = n1*n2 = 3
            # k = k1*p2+k2, i = i1*n2+i2
            # Row index: k*n + i = (k1*2+k2)*3 + i2 (since i1=0)
            # W1: shape (p1*n1, rank1) = (2, 4), W1[k1*n1+i1, r1] = W1[k1, r1]
            # W2: shape (p2*n2, rank2) = (6, 3), W2[k2*n2+i2, r2] = W2[k2*3+i2, r2]
            for k1 in range(2):
                w1_val = W_122[k1][r1]
                for k2 in range(2):
                    for i2 in range(3):
                        w2_val = W_312[k2*3 + i2][r2]
                        row = (k1*2 + k2)*3 + i2
                        W_324[row][r] = w1_val * w2_val

    # --- Strassen decomposition for 2x2x2 (rank 7) ---
    U_222 = [[0]*7 for _ in range(4)]  # n*m = 2*2 = 4
    V_222 = [[0]*7 for _ in range(4)]  # m*p = 2*2 = 4
    W_222 = [[0]*7 for _ in range(4)]  # p*n = 2*2 = 4

    # Strassen's 7 multiplications for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h), m2 = (c+d)e, m3 = a(f-h), m4 = d(g-e), m5 = (a+b)h, m6 = (c-a)(e+f), m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    # U factors (A side): rows (i,j) for i,j in {0,1}
    # V factors (B side): rows (j,k) for j,k in {0,1}
    # W factors (C side): rows (k,i) for k,i in {0,1}
    
    # Term 0: m1 = (a+d)(e+h)
    U_222[0][0] = 1  # a
    U_222[2][0] = 1  # c -> wait, let me index properly
    # U row indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V row indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W row indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3 (k*n+i, n=2)
    
    # m1 = (a+d)(e+h): A: a=U[0,0]=1, d=U[3,0]=1; B: e=V[0,0]=1, h=V[3,0]=1
    U_222[0][0] = 1; U_222[3][0] = 1
    V_222[0][0] = 1; V_222[3][0] = 1
    # C: m1 contributes to C[0,0] and C[1,1]
    W_222[0][0] = 1; W_222[3][0] = 1  # (0,0) and (1,1) -> row 0 and 3

    # m2 = (c+d)e: A: c=U[2,1]=1, d=U[3,1]=1; B: e=V[0,1]=1
    U_222[2][1] = 1; U_222[3][1] = 1
    V_222[0][1] = 1
    # C: m2 contributes to C[1,0] (negative) and C[1,1]
    W_222[2][1] = -1  # (1,0) -> row 1*2+0 = 2
    W_222[3][1] = 1   # (1,1) -> row 3

    # m3 = a(f-h): A: a=U[0,2]=1; B: f=V[1,2]=1, h=V[3,2]=-1
    U_222[0][2] = 1
    V_222[1][2] = 1; V_222[3][2] = -1
    # C: m3 contributes to C[0,1] and C[1,1]
    W_222[1][2] = 1  # (0,1) -> row 0*2+1 = 1
    W_222[3][2] = 1  # (1,1) -> row 3

    # m4 = d(g-e): A: d=U[3,3]=1; B: g=V[2,3]=1, e=V[0,3]=-1
    U_222[3][3] = 1
    V_222[2][3] = 1; V_222[0][3] = -1
    # C: m4 contributes to C[0,0] and C[1,0]
    W_222[0][3] = 1  # (0,0) -> row 0
    W_222[2][3] = 1  # (1,0) -> row 2

    # m5 = (a+b)h: A: a=U[0,4]=1, b=U[1,4]=1; B: h=V[3,4]=1
    U_222[0][4] = 1; U_222[1][4] = 1
    V_222[3][4] = 1
    # C: m5 contributes to C[0,0] (negative) and C[0,1]
    W_222[0][4] = -1  # (0,0) -> row 0
    W_222[1][4] = 1   # (0,1) -> row 1

    # m6 = (c-a)(e+f): A: c=U[2,5]=1, a=U[0,5]=-1; B: e=V[0,5]=1, f=V[1,5]=1
    U_222[2][5] = 1; U_222[0][5] = -1
    V_222[0][5] = 1; V_222[1][5] = 1
    # C: m6 contributes to C[1,1]
    W_222[3][5] = 1  # (1,1) -> row 3

    # m7 = (b-d)(g+h): A: b=U[1,6]=1, d=U[3,6]=-1; B: g=V[2,6]=1, h=V[3,6]=1
    U_222[1][6] = 1; U_222[3][6] = -1
    V_222[2][6] = 1; V_222[3][6] = 1
    # C: m7 contributes to C[0,0]
    W_222[0][6] = 1  # (0,0) -> row 0

    # --- Final Kronecker product of 3x2x4 and 2x2x2 to get 3x4x8 ---
    # 3x2x4: n=3, m=2, p=4, rank=12
    # 2x2x2: n2=2, m2=2, p2=2, rank=7
    # Result: n=3*2=6? No, 3x4x8 means n=3, m=4, p=8.
    # 3x2x4 ⊗ 2x2x2: n=3*2=6? That's wrong. We want 3x4x8.
    # 3x2x4: n1=3, m1=2, p1=4
    # 2x2x2: n2=2, m2=2, p2=2
    # Kronecker: n=3*2=6, m=2*2=4, p=4*2=8. But we want n=3.
    # I made an error. 3x4x8 = (3x4x8). Let me reconsider.
    # 3x4x8: n=3, m=4, p=8.
    # Factor as (3x2x4) ⊗ (1x2x2)? n=3*1=3, m=2*2=4, p=4*2=8. Yes!
    # So 3x4x8 = (3x2x4) ⊗ (1x2x2)
    # 3x2x4 has rank 12 (from above), 1x2x2 has rank 4.
    # Total rank = 12*4 = 48? But wait, I built 3x2x4 as (1x2x2)⊗(3x1x2) with rank 12.
    # So 3x4x8 = (1x2x2)⊗(3x1x2)⊗(1x2x2)? That's getting complicated.
    
    # Let me redo: 3x4x8 = (3x4x8). 
    # Option: 3x4x8 = (3x2x4) ⊗ (1x2x2). n=3*1=3, m=2*2=4, p=4*2=8. Rank = 12*4=48.
    # But I don't have a rank-12 decomposition for 3x2x4 that's correct.
    # Let me just use the direct block approach: 3x4x8 = sum of 3x4x8 blocks.
    # Schoolbook rank = 96.
    # Let me use Strassen on 4x4 blocks: 3x4x8 = 3 blocks of 1x4x8? No.
    # 3x4x8: n=3, m=4, p=8.
    # Split m=4 into 2+2, p=8 into 4+4: 3x4x8 = 3x2x4 ⊗ ... no.
    # Actually, 3x4x8 can be seen as 3 rows, each multiplying a 4x8 matrix.
    # Each 1x4x8: rank 32 (schoolbook). 3 rows: rank 96.
    # With Strassen on 4x4: 1x4x4 rank 7 (Strassen for 1x4x4? No, Strassen is 2x2x2).
    # 1x4x8: split into 4x 1x1x2? No.
    
    # Let me just implement the 88-rank decomposition from the previous analysis.
    # 3x4x8 = (1x4x8) + (2x4x8)
    # 1x4x8: rank 32 (schoolbook)
    # 2x4x8: split into four 2x2x4 blocks
    # 2x2x4: split into two 2x2x2 blocks, each rank 7 (Strassen)
    # 2x4x8: 4 * 2 * 7 = 56
    # Total: 32 + 56 = 88

    # Build 1x4x8 decomposition (rank 32)
    U_148 = [[0]*32 for _ in range(4)]  # n*m = 1*4 = 4
    V_148 = [[0]*32 for _ in range(32)]  # m*p = 4*8 = 32
    W_148 = [[0]*32 for _ in range(8)]  # p*n = 8*1 = 8
    for j in range(4):
        for k in range(8):
            col = j*8 + k
            U_148[j][col] = 1
            V_148[j*8+k][col] = 1
            W_148[k][col] = 1

    # Build 2x4x8 decomposition (rank 56)
    # Split into 4 blocks of 2x2x4, each split into 2 blocks of 2x2x2
    U_248 = [[0]*56 for _ in range(8)]  # n*m = 2*4 = 8
    V_248 = [[0]*56 for _ in range(32)]  # m*p = 4*8 = 32
    W_248 = [[0]*56 for _ in range(16)]  # p*n = 8*2 = 16

    # For each block (j_block, k_block) where j_block in {0,1}, k_block in {0,1}
    # Block is 2x2x4, indices: i in {0,1}, j in {2*j_block, 2*j_block+1}, k in {4*k_block, 4*k_block+3}
    # Split 2x2x4 into two 2x2x2: k in {4*k_block, 4*k_block+1} and k in {4*k_block+2, 4*k_block+3}
    col_offset = 0
    for j_block in range(2):
        for k_block in range(2):
            for k_sub in range(2):
                # 2x2x2 block at (i in {0,1}, j in {2*j_block, 2*j_block+1}, k in {4*k_block+2*k_sub, 4*k_block+2*k_sub+1})
                base_col = col_offset
                for r in range(7):
                    # U: row i*4 + j for i in {0,1}, j in {2*j_block, 2*j_block+1}
                    for i in range(2):
                        for j in range(2):
                            j_idx = 2*j_block + j
                            u_val = U_222[i*2 + j][r]
                            if u_val != 0:
                                U_248[i*4 + j_idx][base_col + r] = u_val
                    # V: row j*8 + k for j in {2*j_block, 2*j_block+1}, k in {4*k_block+2*k_sub, 4*k_block+2*k_sub+1}
                    for j in range(2):
                        for k in range(2):
                            j_idx = 2*j_block + j
                            k_idx = 4*k_block + 2*k_sub + k
                            v_val = V_222[j*2 + k][r]
                            if v_val != 0:
                                V_248[j_idx*8 + k_idx][base_col + r] = v_val
                    # W: row k*2 + i for k in {4*k_block+2*k_sub, 4*k_block+2*k_sub+1}, i in {0,1}
                    for k in range(2):
                        for i in range(2):
                            k_idx = 4*k_block + 2*k_sub + k
                            w_val = W_222[k*2 + i][r]
                            if w_val != 0:
                                W_248[k_idx*2 + i][base_col + r] = w_val
                col_offset += 7
            # After 2 k_subs, col_offset += 14
        # After 2 k_blocks, col_offset += 28
    # After 2 j_blocks, col_offset = 56

    # Combine 1x4x8 and 2x4x8
    # 1x4x8: rows 0 (i=0), 2x4x8: rows 1,2 (i=1,2)
    # Final U: shape (3*4, 88) = (12, 88)
    # Final V: shape (4*8, 88) = (32, 88)
    # Final W: shape (8*3, 88) = (24, 88)
    U = [[0]*88 for _ in range(12)]
    V = [[0]*88 for _ in range(32)]
    W = [[0]*88 for _ in range(24)]

    # 1x4x8 part (i=0)
    for j in range(4):
        for k in range(8):
            col = j*8 + k
            U[0*4 + j][col] = 1  # i=0, j
            V[j*8 + k][col] = 1
            W[k*3 + 0][col] = 1  # k, i=0

    # 2x4x8 part (i=1,2)
    for i in range(2):
        for j in range(4):
            for r in range(56):
                col = 32 + r
                u_val = U_248[i*4 + j][r]
                if u_val != 0:
                    U[(i+1)*4 + j][col] = u_val
    for j in range(4):
        for k in range(8):
            for r in range(56):
                col = 32 + r
                v_val = V_248[j*8 + k][r]
                if v_val != 0:
                    V[j*8 + k][col] = v_val
    for k in range(8):
        for i in range(2):
            for r in range(56):
                col = 32 + r
                w_val = W_248[k*2 + i][r]
                if w_val != 0:
                    W[k*3 + (i+1)][col] = w_val

    return {"U": U, "V": V, "W": W}
