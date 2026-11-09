<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return (U, V, W) for 2x2 @ 2x2, rank 7.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    # Strassen 7 multiplications:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Index mapping for U (2x2 matrix, row = i*2+j):
    # A11 -> (0,0) -> row 0, A12 -> (0,1) -> row 1
    # A21 -> (1,0) -> row 2, A22 -> (1,1) -> row 3
    #
    # Index mapping for V (2x2 matrix, row = j*2+k):
    # B11 -> (0,0) -> row 0, B12 -> (0,1) -> row 1
    # B21 -> (1,0) -> row 2, B22 -> (1,1) -> row 3
    #
    # Index mapping for W (2x2 matrix, row = k*2+i):
    # C11 -> (0,0) -> row 0, C21 -> (1,0) -> row 2
    # C12 -> (0,1) -> row 1, C22 -> (1,1) -> row 3
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22): A11(+), A22(+) ; B11(+), B22(+)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C11 + M1, C22 + M1
    
    # M2 = (A21+A22)B11: A21(+), A22(+) ; B11(+)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1  # C21 + M2, C22 - M2
    
    # M3 = A11(B12-B22): A11(+) ; B12(+), B22(-)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1  # C12 + M3, C22 + M3
    
    # M4 = A22(B21-B11): A22(+) ; B21(+), B11(-)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1  # C11 + M4, C21 + M4
    
    # M5 = (A11+A12)B22: A11(+), A12(+) ; B22(+)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1  # C11 - M5, C12 + M5
    
    # M6 = (A21-A11)(B11+B12): A21(+), A11(-) ; B11(+), B12(+)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C22 + M6
    
    # M7 = (A12-A22)(B21+B22): A12(+), A22(-) ; B21(+), B22(+)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C11 + M7
    
    return U, V, W


def _schoolbook_2x2():
    """Return (U, V, W) for 2x2 @ 2x2, rank 4 (schoolbook)."""
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(4)]
    c = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j][c] = 1
                V[j*2+k][c] = 1
                W[k*2+i][c] = 1
                c += 1
    return U, V, W


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 @ 4x5.
    Uses Strassen's algorithm on the 2x2 blocks for the first 2 rows/cols of B.
    Strategy:
    - A is 3x4, B is 4x5.
    - Split B's first 4 rows into two groups of 2: B_top (rows 0,1) and B_bot (rows 2,3).
    - For each of the 5 columns of B, we have a 4x1 vector, split into top (2x1) and bot (2x1).
    - A is split into A_left (3x2, cols 0,1) and A_right (3x2, cols 2,3).
    - C = A_left @ B_top + A_right @ B_bot.
    - A_left @ B_top: (3x2) @ (2x5). We can view this as 5 separate (3x2) @ (2x1) multiplications.
    - For each column, we can use a decomposition. But better: treat (3x2) @ (2x2) for two columns at a time, then the 5th separately.
    - Actually, let's just do schoolbook for the 3x2 part and use Strassen where applicable.
    
    Simpler approach: 
    - Use Strassen for the 2x2 block multiplication where possible.
    - A is 3x4. Split A into 3 rows x 2 cols blocks: A0 (rows 0,1, cols 0,1), A1 (row 2, cols 0,1), etc.
    - Actually, let's use the following block structure:
      A = [A11 A12]  where A11 is 3x2, A12 is 3x2
      B = [B11 B21; B21 B22] wait, B is 4x5. Split B into top 2 rows and bottom 2 rows:
      B = [Btop; Bbot] where Btop is 2x5, Bbot is 2x5.
      C = A11 @ Btop + A12 @ Bbot.
    
    For A11 @ Btop: (3x2) @ (2x5). This can be done with rank 3*2*5 = 30 schoolbook, or we can reduce.
    For A12 @ Bbot: (3x2) @ (2x5). Same.
    Total schoolbook: 60.
    
    To reduce: For the (3x2) @ (2x5), we can split the 3 rows into 2+1.
    For the 2x2 @ 2x5 part, use Strassen-like or just 7 multiplications per 2 columns of B.
    2x2 @ 2x2 has rank 7. So 2x2 @ 2x5 = 2*(2x2@2x2) + 1*(2x2@2x1).
    2x2@2x2: rank 7. 2x2@2x1: rank 4 (schoolbook) or we can do better? 
    Actually 2x2 @ 1x2... wait B is 2x5, so Btop is 2x5.
    A11 is 3x2. Split A11 into A11_top (2x2, rows 0,1) and A11_bot (1x2, row 2).
    A11 @ Btop = A11_top @ Btop + A11_bot @ Btop.
    A11_top @ Btop: (2x2) @ (2x5). Split Btop into Btop[:2] (2x2), Btop[2:4] (2x2), Btop[4] (2x1).
    (2x2)@(2x2): rank 7 each, two of them = 14.
    (2x2)@(2x1): rank 4.
    Total for A11_top @ Btop: 18.
    A11_bot @ Btop: (1x2) @ (2x5). Rank 2*5 = 10 schoolbook. Can we do better? 
    (1x2) @ (2x5) = sum of 2x5 scalar mults. Rank 10. 
    So A11 @ Btop total: 18 + 10 = 28.
    Similarly A12 @ Bbot: 28.
    Total: 56. Still not great.
    
    Let me try a different approach. Use Strassen for the whole 4x4 block if possible.
    Actually, the known rank for 3x4x5 is 48. Let me just implement the schoolbook (rank 60) to be safe,
    or try to construct a rank 48 solution if I can.
    
    Given the complexity, let me return the schoolbook decomposition which is guaranteed correct.
    """
    n, m, p = DIMENSIONS
    # Schoolbook: rank = n*m*p = 3*4*5 = 60
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
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import time

def _strassen_2x2():
    """Return (U, V, W) for 2x2 @ 2x2, rank 7.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A11(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A22(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _schoolbook_2x2():
    """Return (U, V, W) for 2x2 @ 2x2, rank 4 (schoolbook)."""
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(4)]
    c = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j][c] = 1
                V[j*2+k][c] = 1
                W[k*2+i][c] = 1
                c += 1
    return U, V, W


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 @ 4x5.
    
    Strategy: Use Strassen's algorithm on 2x2 blocks.
    Split A (3x4) into A_top (2x4, rows 0-1) and A_bot (1x4, row 2).
    Split B (4x5) into B_left (4x2, cols 0-1) and B_right (4x3, cols 2-4).
    
    Actually, let's use a cleaner approach:
    C = A @ B where A is 3x4, B is 4x5.
    
    Split the middle dimension (4) into 2+2:
    A = [A1 | A2] where A1 is 3x2 (cols 0-1), A2 is 3x2 (cols 2-3)
    B = [B1; B2] where B1 is 2x5 (rows 0-1), B2 is 2x5 (rows 2-3)
    
    C = A1 @ B1 + A2 @ B2
    
    For A1 @ B1: (3x2) @ (2x5)
    Split A1 into A1_top (2x2, rows 0-1) and A1_bot (1x2, row 2)
    A1 @ B1 = A1_top @ B1 + A1_bot @ B1
    
    A1_top @ B1: (2x2) @ (2x5)
    Split B1 into B1a (2x2, cols 0-1), B1b (2x2, cols 2-3), B1c (2x1, col 4)
    A1_top @ B1 = A1_top @ B1a + A1_top @ B1b + A1_top @ B1c
    (2x2)@(2x2): rank 7 each, so 14
    (2x2)@(2x1): rank 4
    Total: 18
    
    A1_bot @ B1: (1x2) @ (2x5), rank 10
    
    Total for A1 @ B1: 28
    Similarly for A2 @ B2: 28
    Grand total: 56
    
    This is better than 60 but not optimal. Let me try to get 48.
    
    Alternative: Use Strassen more aggressively.
    For (2x2)@(2x1), can we do better than rank 4?
    (2x2)@(2x1): C1 = A11*B1 + A12*B2, C2 = A21*B1 + A22*B2
    This is a bilinear map from (R^2 x R^2) x R^2 -> R^2.
    The rank is 3 (I believe). Let me verify:
    M1 = A11*B1
    M2 = A22*B2
    M3 = (A11+A22)*(B1+B2) = A11*B1 + A11*B2 + A22*B1 + A22*B2
    C1 = A11*B1 + A12*B2
    C2 = A21*B1 + A22*B2
    From M1, M2, M3: C1 = M1 + A12*B2, still need A12*B2.
    So rank 4 seems right for (2x2)@(2x1).
    
    Let me just go with rank 56 using the above construction.
    """
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # We'll build U, V, W incrementally
    # U has shape (n*m, rank) = (12, rank)
    # V has shape (m*p, rank) = (20, rank)  
    # W has shape (p*n, rank) = (15, rank)
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term. u_vec has len(n*m), v_vec has len(m*p), w_vec has len(p*n)."""
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
    
    # Helper: create zero vectors
    def zeros_u(): return [0]*(n*m)
    def zeros_v(): return [0]*(m*p)
    def zeros_w(): return [0]*(p*n)
    
    # We split: A = [A1 | A2], B = [B1; B2]
    # A1: cols 0-1 of A (indices 0-1 in the middle dim)
    # A2: cols 2-3 of A
    # B1: rows 0-1 of B
    # B2: rows 2-3 of B
    
    # For each pair (A_block, B_block) where A_block is 3x2, B_block is 2x5:
    # We further split A_block into top (2x2) and bot (1x2)
    
    def process_3x2_at_2x5(a_row_offset, b_row_offset):
        """Process A_block (3x2) @ B_block (2x5).
        a_row_offset: which rows of A (0 for top 2 rows, 1 for bottom 1 row... actually we handle the 3 rows)
        Actually, let me parameterize differently.
        
        a_col_start: starting column of A for the 2 columns (0 or 2)
        b_row_start: starting row of B for the 2 rows (0 or 2)
        """
        a_col_start = a_row_offset  # 0 or 2
        b_row_start = b_row_offset  # 0 or 2
        
        # A_block is rows 0-2, cols a_col_start to a_col_start+1 (3x2)
        # B_block is rows b_row_start to b_row_start+1, cols 0-4 (2x5)
        
        # Split A_block: A_top (rows 0-1, 2x2) and A_bot (row 2, 1x2)
        
        # Process A_top @ B_block: (2x2) @ (2x5)
        # Split B_block into B_a (2x2, cols 0-1), B_b (2x2, cols 2-3), B_c (2x1, col 4)
        
        # (2x2)@(2x2) using Strassen for B_a
        sU, sV, sW = _strassen_2x2()
        # Map indices: A_top[i][j] -> U[(0+i)*2+j] in local, but in global:
        # A row i (0 or 1), col a_col_start+j -> global index i*4 + (a_col_start+j)
        # B row b_row_start+j, col k -> global index (b_row_start+j)*5 + k
        # C row k, col i -> global index k*3 + i
        
        for c in range(7):
            u_vec = zeros_u()
            v_vec = zeros_v()
            w_vec = zeros_w()
            for local_i in range(2):
                for local_j in range(2):
                    val = sU[local_i*2+local_j][c]
                    if val != 0:
                        global_i = local_i  # rows 0,1 of A
                        global_j = a_col_start + local_j
                        u_vec[global_i * m + global_j] += val
            for local_j in range(2):
                for local_k in range(2):
                    val = sV[local_j*2+local_k][c]
                    if val != 0:
                        global_j = b_row_start + local_j
                        global_k = local_k  # cols 0,1 of B
                        v_vec[global_j * p + global_k] += val
            for local_k in range(2):
                for local_i in range(2):
                    val = sW[local_k*2+local_i][c]
                    if val != 0:
                        global_k = local_k
                        global_i = local_i
                        w_vec[global_k * n + global_i] += val
            add_term(u_vec, v_vec, w_vec)
        
        # (2x2)@(2x2) using Strassen for B_b (cols 2-3)
        sU, sV, sW = _strassen_2x2()
        for c in range(7):
            u_vec = zeros_u()
            v_vec = zeros_v()
            w_vec = zeros_w()
            for local_i in range(2):
                for local_j in range(2):
                    val = sU[local_i*2+local_j][c]
                    if val != 0:
                        global_i = local_i
                        global_j = a_col_start + local_j
                        u_vec[global_i * m + global_j] += val
            for local_j in range(2):
                for local_k in range(2):
                    val = sV[local_j*2+local_k][c]
                    if val != 0:
                        global_j = b_row_start + local_j
                        global_k = 2 + local_k  # cols 2,3 of B
                        v_vec[global_j * p + global_k] += val
            for local_k in range(2):
                for local_i in range(2):
                    val = sW[local_k*2+local_i][c]
                    if val != 0:
                        global_k = 2 + local_k
                        global_i = local_i
                        w_vec[global_k * n + global_i] += val
            add_term(u_vec, v_vec, w_vec)
        
        # (2x2)@(2x1) for B_c (col 4) - schoolbook, rank 4
        for local_i in range(2):
            for local_j in range(2):
                u_vec = zeros_u()
                v_vec = zeros_v()
                w_vec = zeros_w()
                u_vec[local_i * m + (a_col_start + local_j)] = 1
                v_vec[(b_row_start + local_j) * p + 4] = 1
                w_vec[4 * n + local_i] = 1
                add_term(u_vec, v_vec, w_vec)
        
        # Process A_bot @ B_block: (1x2) @ (2x5), rank 10
        for local_j in range(2):
            for local_k in range(5):
                u_vec = zeros_u()
                v_vec = zeros_v()
                w_vec = zeros_w()
                # A row 2, col a_col_start+local_j
                u_vec[2 * m + (a_col_start + local_j)] = 1
                # B row b_row_start+local_j, col local_k
                v_vec[(b_row_start + local_j) * p + local_k] = 1
                # C row local_k, col 2
                w_vec[local_k * n + 2] = 1
                add_term(u_vec, v_vec, w_vec)
    
    # Process A1 (cols 0-1) @ B1 (rows 0-1)
    process_3x2_at_2x5(0, 0)
    # Process A2 (cols 2-3) @ B2 (rows 2-3)
    process_3x2_at_2x5(2, 2)
    
    rank = len(U_list)
    U = [list(row) for row in U_list]  # Wait, U_list is list of column vectors? No...
    
    # Actually, U_list[i] is the i-th rank-1 term's U vector (length n*m)
    # We need U as (n*m) x rank matrix, so U[row][col] = U_list[col][row]
    U = [[U_list[col][row] for col in range(rank)] for row in range(n*m)]
    V = [[V_list[col][row] for col in range(rank)] for row in range(m*p)]
    W = [[W_list[col][row] for col in range(rank)] for row in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I need to verify this is correct. Let me trace through the logic more carefully.

The tensor decomposition requires: for all i, j, k:
T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]

where T[i,j,k] = 1 if this is a valid matrix multiplication term, i.e., for A[i,j] * B[j,k] -> C[i,k].

Wait, let me re-read the convention. The task says:
"U factor follows the official trace convention W[k*n+i,r]"

So W is indexed as W[k*n+i, r] where k is the column of C (which is the row of B) and i is the row of C (which is the row of A).

U[i*m+j, c] corresponds to A[i,j]
V[j*p+k, c] corresponds to B[j,k]
W[k*n+i, c] corresponds to C[i,k]

The decomposition: C[i,k] = sum_j A[i,j] * B[j,k] = sum_c (sum_j U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c]

Wait, that's not quite right. The tensor decomposition is:
T_{i,j,k} = sum_c U_{i,j,c} * V_{j,k,c} * W_{k,i,c}

where T_{i,j,k} = 1 for the matrix multiplication tensor (i.e., the tensor that represents the bilinear map).

Actually, for matrix multiplication, the tensor T is defined as:
T[i,j,k] = 1 if the (i,j) entry of A times the (j,k) entry of B contributes to C[i,k].

So T[i,j,k] = 1 for all valid i, j, k. And the decomposition must satisfy:
1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for all i, j, k.

Let me verify my Strassen implementation is correct by checking a few entries.

For Strassen 2x2, with the mapping:
U[local_i*2+local_j, c] for A[local_i, local_j]
V[local_j*2+local_k, c] for B[local_j, local_k]
W[local_k*2+local_i, c] for C[local_i, local_k]

Let me verify M1 = (A11+A22)(B11+B22):
U[0,0]=1 (A11), U[3,0]=1 (A22)
V[0,0]=1 (B11), V[3,0]=1 (B22)
W[0,0]=1 (C11), W[3,0]=1 (C22)

Contribution to T[0,0,0] (A11*B11->C11): U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 ✓
Contribution to T[0,1,0] (A11*B21->C11... wait, B[1,0] is B21): U[0,0]*V[2,0]*W[0,0] = 1*0*1 = 0 ✓ (should be 0 since B21 doesn't contribute to C11 in this term)

Wait, I'm confusing myself. Let me re-index.

For 2x2:
A = [[A11, A12], [A21, A22]]
B = [[B11, B12], [B21, B22]]
C = A@B = [[C11, C12], [C21, C22]]

C11 = A11*B11 + A12*B21
C12 = A11*B12 + A12*B22
C21 = A21*B11 + A22*B21
C22 = A21*B12 + A22*B22

The tensor T[i,j,k] = 1 for all i,j,k in {0,1}^3 (since every term contributes).

Wait no! T[i,j,k] = 1 means that A[i,j] * B[j,k] contributes to C[i,k]. So:
T[0,0,0] = 1 (A11*B11 -> C11)
T[0,0,1] = 1 (A11*B12 -> C12)
T[0,1,0] = 1 (A12*B21 -> C11)
T[0,1,1] = 1 (A12*B22 -> C12)
T[1,0,0] = 1 (A21*B11 -> C21)
T[1,0,1] = 1 (A21*B12 -> C22)
T[1,1,0] = 1 (A22*B21 -> C21)
T[1,1,1] = 1 (A22*B22 -> C22)

So T[i,j,k] = 1 for ALL i,j,k in {0,1}^3. The decomposition must satisfy:
sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] = 1 for all i,j,k.

Let me verify Strassen M1:
U[0,0]=1, U[3,0]=1 (A11, A22)
V[0,0]=1, V[3,0]=1 (B11, B22)
W[0,0]=1, W[3,0]=1 (C11, C22)

T[0,0,0]: sum_c includes c=0: U[0,0]*V[0,0]*W[0,0] = 1. Other c's contribute 0 to this. Total = 1 ✓
T[0,0,1]: U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0. Need total = 1. This comes from other terms.
T[0,1,0]: U[0,1]*V[1,0]*W[0,0] = 0*0*1 = 0. 
T[0,1,1]: U[0,1]*V[1,1]*W[1,0] = 0*0*0 = 0.
T[1,0,0]: U[2,0]*V[0,0]*W[0,1] = 0*1*0 = 0.
T[1,0,1]: U[2,0]*V[0,1]*W[1,1] = 0*0*0 = 0.
T[1,1,0]: U[3,0]*V[1,0]*W[0,3] = 1*0*0 = 0.
T[1,1,1]: U[3,0]*V[1,1]*W[1,3] = 1*0*0 = 0.

So M1 only contributes to T[0,0,0] and T[1,1,1] (via W[3,0]=1, T[1,1,1]: U[3,0]*V[1,1]*W[1,3]... wait W is indexed as W[k*2+i, c].

W[0,0]=1 means k=0, i=0, so C[0,0] = C11.
W[3,0]=1 means k=1, i=1, so C[1,1] = C22.

For T[0,0,0] (i=0,j=0,k=0): U[0*2+0,0]*V[0*2+0,0]*W[0*2+0,0] = U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 ✓
For T[1,1,1] (i=1,j=1,k=1): U[1*2+1,0]*V[1*2+1,0]*W[1*2+1,0] = U[3,0]*V[3,0]*W[3,0] = 1*1*1 = 1 ✓

Good. Now I need to verify all 8 entries sum to 1 across all 7 terms. This is tedious but I trust the standard Strassen construction.

Let me just make sure my code is syntactically correct and returns the right shapes. The rank should be 2*(18) + 2*(10) = 56. Wait, let me recount:

For each (3x2)@(2x5) block:
- (2x2)@(2x2) for B_a: 7
- (2x2)@(2x2) for B_b: 7
- (2x2)@(2x1) for B_c: 4
- (1x2)@(2x5): 10
Total: 28

Two such blocks: 56.

So rank = 56, which is better than 60. Let me make sure the code is correct.

Actually, I realize there might be an issue with my code. Let me simplify and make it more robust. Let me also add a deadline guard as required.

Let me rewrite this more carefully:

