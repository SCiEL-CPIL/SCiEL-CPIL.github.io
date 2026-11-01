"""Matrix multiplication 2x4 x 4x7 with Strassen block decomposition."""
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Partition: A (2x4) = [A1 | A2] where A1, A2 are 2x2
    # B (4x7) = [B1; B2] where B1, B2 are 2x7
    # C (2x7) = A1*B1 + A2*B2
    #
    # For each 2x2 x 2x7 product:
    #   Split B into 3 blocks of 2x2 and 1 block of 2x1
    #   Use Strassen (rank 7) for each 2x2x2x2
    #   Use schoolbook (rank 4) for 2x2x2x1
    #   Total per 2x2x2x7: 3*7 + 4 = 25
    # Two such products: 2*25 = 50 total rank
    
    def strassen_2x2_2x2():
        """Return (U, V, W) for 2x2 x 2x2 multiplication, rank 7.
        U: (4, 7), V: (4, 7), W: (4, 7)
        U indices: i*2+j for (i,j) in 2x2
        V indices: j*2+k for (j,k) in 2x2  
        W indices: k*2+i for (k,i) in 2x2 (trace convention W[k*n+i,r])
        """
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # M indices: m00=U[0], m10=U[2], m01=U[1], m11=U[3]
        # N indices: n00=V[0], n10=V[2], n01=V[1], n11=V[3]
        # C indices: c00=W[0], c10=W[2], c01=W[1], c11=W[3]
        
        # P1 = (m00+m11)(n00+n11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1  # c00 += P1, c11 += P1
        
        # P2 = (m10+m11)n00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1  # c10 += P2
        
        # P3 = m00(n01-n11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1  # c01 += P3, c11 += P3
        
        # P4 = m11(n10-n00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1  # c00 += P4, c10 += P4
        
        # P5 = (m00+m01)n11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[3][4] = 1  # c00 -= P5, c11 += P5
        
        # P6 = (m10-m00)(n00+n01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1  # c11 += P6
        
        # P7 = (m01-m11)(n10+n11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[1][6] = 1  # c00 += P7, c01 += P7
        
        return U, V, W
    
    def schoolbook_2x2_2x1():
        """Return (U, V, W) for 2x2 x 2x1 multiplication, rank 4.
        U: (4, 4), V: (2, 4), W: (2, 4)
        V indices: j*1+k = j for (j,k) in 2x1
        W indices: k*2+i for (k,i) in 2x1
        """
        rank = 4
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(2)]
        W = [[0]*rank for _ in range(2)]
        
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(1):
                    U[i*2+j][col] = 1
                    V[j][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        
        return U, V, W
    
    def combine(U1, V1, W1, U2, V2, W2):
        """Concatenate two decompositions (block diagonal in rank space)."""
        r1 = len(W1[0]) if W1 else 0
        r2 = len(W2[0]) if W2 else 0
        rank = r1 + r2
        U = [[0]*rank for _ in range(len(U1))]
        V = [[0]*rank for _ in range(len(V1))]
        W = [[0]*rank for _ in range(len(W1))]
        for i in range(len(U1)):
            for c in range(r1):
                U[i][c] = U1[i][c]
        for i in range(len(V1)):
            for c in range(r1):
                V[i][c] = V1[i][c]
        for i in range(len(W1)):
            for c in range(r1):
                W[i][c] = W1[i][c]
        for i in range(len(U2)):
            for c in range(r2):
                U[i][r1+c] = U2[i][c]
        for i in range(len(V2)):
            for c in range(r2):
                V[i][r1+c] = V2[i][c]
        for i in range(len(W2)):
            for c in range(r2):
                W[i][r1+c] = W2[i][c]
        return U, V, W
    
    # Build 2x2 x 2x7 decomposition
    # Split 7 columns into 3 groups of 2 + 1 group of 1
    # U is 4x25, V is 14x25 (2*7), W is 14x25 (7*2)
    
    # For each column block, we need to map local indices to global
    # Local U: 4 rows (2x2 matrix M)
    # Local V: 2 rows per column group (for 2x1) or 4 rows (for 2x2)
    # Local W: 2 rows per column group (for 2x1) or 4 rows (for 2x2)
    
    # Let's build 2x2 x 2x7 piece by piece
    # Column groups: cols 0-1 (2x2), cols 2-3 (2x2), cols 4-5 (2x2), col 6 (2x1)
    
    U_2x2_2x7 = [[0]*25 for _ in range(4)]
    V_2x2_2x7 = [[0]*25 for _ in range(14)]  # 2*7
    W_2x2_2x7 = [[0]*25 for _ in range(14)]  # 7*2
    
    col_offset = 0
    for block_idx in range(3):
        if time.time() > deadline:
            break
        sU, sV, sW = strassen_2x2_2x2()
        # sU: 4x7, sV: 4x7, sW: 4x7
        # Map to global: U rows stay 0-3
        # V local row j*2+k (j=0,1; k=0,1) -> global row (block_idx*2 + k)  [since B is 2x7, row j of B, col k]
        #   Actually B is 2x7, so V index is j*7+k where j in {0,1}, k in {0..6}
        #   For block_idx, the 2 columns are at positions block_idx*2 and block_idx*2+1
        #   So local V row (j,k) -> global V row j*7 + (block_idx*2 + k)
        # W local row k*2+i (k=0,1; i=0,1) -> global row (block_idx*2 + k)*2 + i
        #   C is 2x7, so W index is k*2+i where k in {0..6}, i in {0,1}
        #   For block_idx, columns block_idx*2 and block_idx*2+1
        #   So local W row (k,i) -> global W row (block_idx*2 + k)*2 + i
        
        for i in range(4):
            for c in range(7):
                U_2x2_2x7[i][col_offset + c] = sU[i][c]
        for j in range(2):
            for k in range(2):
                local_row = j * 2 + k
                global_row = j * 7 + (block_idx * 2 + k)
                for c in range(7):
                    V_2x2_2x7[global_row][col_offset + c] = sV[local_row][c]
        for k in range(2):
            for i in range(2):
                local_row = k * 2 + i
                global_row = (block_idx * 2 + k) * 2 + i
                for c in range(7):
                    W_2x2_2x7[global_row][col_offset + c] = sW[local_row][c]
        col_offset += 7
    
    # Handle the last column (col 6) with schoolbook 2x2 x 2x1
    if time.time() <= deadline:
        sU, sV, sW = schoolbook_2x2_2x1()
        # sU: 4x4, sV: 2x4, sW: 2x4
        # V local row j -> global row j*7 + 6
        # W local row k*2+i -> global row 6*2 + i (only k=0)
        for i in range(4):
            for c in range(4):
                U_2x2_2x7[i][col_offset + c] = sU[i][c]
        for j in range(2):
            global_row = j * 7 + 6
            for c in range(4):
                V_2x2_2x7[global_row][col_offset + c] = sV[j][c]
        for i in range(2):
            global_row = 6 * 2 + i
            for c in range(4):
                W_2x2_2x7[global_row][col_offset + c] = sW[i][c]
        col_offset += 4
    
    # Now combine two such decompositions for A1*B1 and A2*B2
    # A1 is rows 0-1 of A (2x4), A2 is... wait, A is 2x4.
    # Actually A is 2x4, so A = [A1 | A2] where A1 is 2x2 (cols 0-1), A2 is 2x2 (cols 2-3)
    # B is 4x7 = [B1; B2] where B1 is 2x7 (rows 0-1), B2 is 2x7 (rows 2-3)
    # C = A1*B1 + A2*B2
    #
    # For A1*B1: A1 is 2x2, B1 is 2x7 -> use U_2x2_2x7 with A1's entries
    # For A2*B2: A2 is 2x2, B2 is 2x7 -> use same structure but different A, B indices
    #
    # In the tensor decomposition:
    # U has 8 rows (2*4 for A: i*4+j, i in {0,1}, j in {0,1,2,3})
    # V has 28 rows (4*7 for B: j*7+k, j in {0,1,2,3}, k in {0..6})
    # W has 14 rows (7*2 for C: k*2+i, k in {0..6}, i in {0,1})
    
    total_rank = 50
    U_final = [[0]*total_rank for _ in range(8)]
    V_final = [[0]*total_rank for _ in range(28)]
    W_final = [[0]*total_rank for _ in range(14)]
    
    # A1*B1: A1 = A[0:2, 0:2], B1 = B[0:2, :]
    # U rows for A1: i*4+j for i in {0,1}, j in {0,1} -> rows 0,1,4,5
    # V rows for B1: j*7+k for j in {0,1}, k in {0..6} -> rows 0-6, 7-13
    # W rows: all 14 rows (k*2+i for k in {0..6}, i in {0,1})
    
    for i in range(2):
        for j in range(2):
            a_row = i * 4 + j
            local_row = i * 2 + j
            for c in range(25):
                U_final[a_row][c] = U_2x2_2x7[local_row][c]
    
    for j in range(2):
        for k in range(7):
            b_row = j * 7 + k
            local_row = j * 2 + k  # Wait, this is wrong
            # B1 is 2x7, so in the 2x2x2x7 decomposition, V index is j*7+k
            # But U_2x2_2x7's V has 14 rows indexed as j*7+k for j in {0,1}, k in {0..6}
            # So local_row = j*7+k is already the global row for B1
            for c in range(25):
                V_final[b_row][c] = V_2x2_2x7[j * 7 + k][c]
    
    for k in range(7):
        for i in range(2):
            c_row = k * 2 + i
            for c in range(25):
                W_final[c_row][c] = W_2x2_2x7[k * 2 + i][c]
    
    # A2*B2: A2 = A[0:2, 2:4], B2 = B[2:4, :]
    # U rows for A2: i*4+j for i in {0,1}, j in {2,3} -> rows 2,3,6,7
    # V rows for B2: j*7+k for j in {2,3}, k in {0..6} -> rows 14-20, 21-27
    
    for i in range(2):
        for j in range(2):
            a_row = i * 4 + (j + 2)
            local_row = i * 2 + j
            for c in range(25):
                U_final[a_row][25 + c] = U_2x2_2x7[local_row][c]
    
    for j in range(2):
        for k in range(7):
            b_row = (j + 2) * 7 + k
            local_row = j * 7 + k
            for c in range(25):
                V_final[b_row][25 + c] = V_2x2_2x7[local_row][c]
    
    for k in range(7):
        for i in range(2):
            c_row = k * 2 + i
            for c in range(25):
                W_final[c_row][25 + c] = W_2x2_2x7[k * 2 + i][c]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
