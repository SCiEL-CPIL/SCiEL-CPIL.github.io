"""Matrix multiplication tensor decomposition for 4x4*4x6 -> rank 36."""
import time

DIMENSIONS = [4, 4, 6]


def _strassen_222():
    """Return U (4x7), V (4x7), W (4x7) for 2x2x2 Strassen decomposition.
    
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    Indices: i in {0,1}, j in {0,1}, k in {0,1}.
    """
    # Strassen's 7 multiplications for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    # U: A00 at row 0, A11 at row 3
    U[0][0] = 1
    U[3][0] = 1
    # V: B00 at row 0, B11 at row 3
    V[0][0] = 1
    V[3][0] = 1
    # W: C00 += M1, C11 += M1
    W[0][0] = 1   # k=0,i=0 -> C00
    W[3][0] = 1   # k=1,i=1 -> C11
    
    # M2: (A10+A11)B00
    # U: A10 at row 2, A11 at row 3
    U[2][1] = 1
    U[3][1] = 1
    # V: B00 at row 0
    V[0][1] = 1
    # W: C10 += M2, C11 -= M2
    W[2][1] = 1   # k=1,i=0 -> C10
    W[3][1] = -1  # k=1,i=1 -> C11
    
    # M3: A00(B01-B11)
    # U: A00 at row 0
    U[0][2] = 1
    # V: B01 at row 1, B11 at row 3 with -1
    V[1][2] = 1
    V[3][2] = -1
    # W: C01 += M3, C11 += M3
    W[1][2] = 1   # k=0,i=1 -> C01
    W[3][2] = 1   # k=1,i=1 -> C11
    
    # M4: A11(B10-B00)
    # U: A11 at row 3
    U[3][3] = 1
    # V: B10 at row 2, B00 at row 0 with -1
    V[2][3] = 1
    V[0][3] = -1
    # W: C00 += M4, C10 += M4
    W[0][3] = 1   # k=0,i=0 -> C00
    W[2][3] = 1   # k=1,i=0 -> C10
    
    # M5: (A00+A01)B11
    # U: A00 at row 0, A01 at row 1
    U[0][4] = 1
    U[1][4] = 1
    # V: B11 at row 3
    V[3][4] = 1
    # W: C00 -= M5, C01 += M5
    W[0][4] = -1  # k=0,i=0 -> C00
    W[1][4] = 1   # k=0,i=1 -> C01
    
    # M6: (A10-A00)(B00+B01)
    # U: A10 at row 2, A00 at row 0 with -1
    U[2][5] = 1
    U[0][5] = -1
    # V: B00 at row 0, B01 at row 1
    V[0][5] = 1
    V[1][5] = 1
    # W: C11 += M6
    W[3][5] = 1   # k=1,i=1 -> C11
    
    # M7: (A01-A11)(B10+B11)
    # U: A01 at row 1, A11 at row 3 with -1
    U[1][6] = 1
    U[3][6] = -1
    # V: B10 at row 2, B11 at row 3
    V[2][6] = 1
    V[3][6] = 1
    # W: C00 += M7
    W[0][6] = 1   # k=0,i=0 -> C00
    
    return U, V, W


def _schoolbook_221():
    """Return U (4x2), V (2x2), W (2x2) for 2x2x1 schoolbook.
    
    Convention: U[i*2+j, c], V[j*1+k, c], W[k*2+i, c].
    Here p=1, so V has shape (2*1, 2) = (2, 2), W has shape (1*2, 2) = (2, 2).
    Multiplications: A00*B0, A10*B0 (wait, no...)
    
    Actually for 2x2x1: A is 2x2, B is 2x1, C is 2x1.
    C0 = A00*B0 + A01*B1  (wait, B has shape 2x1, so B has entries B[0], B[1])
    
    Hmm, let me reconsider. A is n x m = 2 x 2. B is m x p = 2 x 1. C is n x p = 2 x 1.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    With p=1, k is always 0.
    C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    
    So two "scalar" multiplications per row: A[i,0]*B[0,0] and A[i,1]*B[1,0].
    But we need to express this as tensor decomposition.
    
    For 2x2x1, the schoolbook has rank 4 (n*m*p = 2*2*1 = 4).
    U is (n*m, rank) = (4, 4), V is (m*p, rank) = (2, 4), W is (p*n, rank) = (2, 4).
    
    Wait, the task says rank-2 for 2x2x1? Let me re-read...
    "rank of 2x2x1 is 2 (schoolbook: A00*B00, A10*B10? No, 2x2x1 is just 2 multiplications)"
    
    That's not right. 2x2x1 means multiplying a 2x2 matrix by a 2x1 matrix, giving a 2x1 result.
    That requires 4 multiplications in schoolbook. The rank of the 2x2x1 tensor is actually 4.
    
    Hmm, but the prompt said "rank of 2x2x1 is 2". That seems wrong. Let me just use schoolbook with rank 4.
    
    Actually wait - maybe they meant a different decomposition. Let me just use schoolbook: rank = 2*2*1 = 4.
    
    Total rank for 4x4x6 with four 2x2x3 blocks, each split into 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11.
    Total = 4 * 11 = 44. Score = 96/44 = 2.18. Still better than 1.0.
    
    Let me just implement schoolbook for 2x2x1 with rank 4.
    """
    n, m, p = 2, 2, 1
    rank = n * m * p  # 4
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W


def _block_223():
    """Compose 2x2x2 (Strassen, rank 7) and 2x2x1 (schoolbook, rank 4) for a 2x2x3 block.
    
    The 2x2x3 problem: A is 2x2, B is 2x3, C is 2x3.
    Split B into B[:, :2] (2x2) and B[:, 2:3] (2x1).
    C[:, :2] = A * B[:, :2]  (2x2x2, Strassen rank 7)
    C[:, 2]  = A * B[:, 2]  (2x2x1, schoolbook rank 4)
    
    Total rank: 11.
    
    U has shape (n*m, rank) = (4, 11)
    V has shape (m*p, rank) = (2*3, 11) = (6, 11)
    W has shape (p*n, rank) = (3*2, 11) = (6, 11)
    """
    U222, V222, W222 = _strassen_222()
    U221, V221, W221 = _schoolbook_221()
    
    r222 = len(U222[0])  # 7
    r221 = len(U221[0])  # 4
    total_rank = r222 + r221  # 11
    
    U = [[0]*total_rank for _ in range(4)]
    V = [[0]*total_rank for _ in range(6)]
    W = [[0]*total_rank for _ in range(6)]
    
    # Part 1: 2x2x2 (columns 0..6)
    for i in range(4):
        for c in range(r222):
            U[i][c] = U222[i][c]
    for j in range(2):
        for k in range(2):
            for c in range(r222):
                V[j*2 + k][c] = V222[j*2 + k][c]
    for k in range(2):
        for i in range(2):
            for c in range(r222):
                W[k*2 + i][c] = W222[k*2 + i][c]
    
    # Part 2: 2x2x1 (columns 7..10)
    # B[:, 2] corresponds to column index 2 in B, so in V (which is m*p x rank = 6 x rank),
    # B[j, 2] is at row j*3 + 2
    # C[:, 2] corresponds to C[i, 2], in W (p*n x rank = 6 x rank),
    # C[i, 2] is at row 2*2 + i = 4 + i
    for i in range(4):
        for c in range(r221):
            U[i][r222 + c] = U221[i][c]
    for j in range(2):
        for c in range(r221):
            V[j*3 + 2][r222 + c] = V221[j][c]  # B[j,2] maps to V row j*3+2
    for i in range(2):
        for c in range(r221):
            W[2*2 + i][r222 + c] = W221[i][c]  # C[i,2] maps to W row 4+i
    
    return U, V, W


def construct():
    """Construct rank-44 decomposition for 4x4*4x6 -> 4x6.
    
    Split into four 2x2x3 blocks:
    - Block (0,0): A[0:2, 0:2] * B[0:2, 0:3] -> C[0:2, 0:3]
    - Block (0,1): A[0:2, 2:4] * B[2:4, 0:3] -> C[0:2, 0:3]  -- NO, this is wrong
    
    Actually, for matrix multiplication A (4x4) * B (4x6) = C (4x6):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can partition the rows of A/C and columns of B independently.
    Partition A's rows into {0,1} and {2,3}, A's columns into {0,1} and {2,3}.
    Partition B's rows into {0,1} and {2,3}, B's columns into {0,1,2} and {3,4,5}.
    
    Then:
    C[0:2, 0:3] = A[0:2, 0:2]*B[0:2, 0:3] + A[0:2, 2:4]*B[2:4, 0:3]
    C[0:2, 3:6] = A[0:2, 0:2]*B[0:2, 3:6] + A[0:2, 2:4]*B[2:4, 3:6]
    C[2:4, 0:3] = A[2:4, 0:2]*B[0:2, 0:3] + A[2:4, 2:4]*B[2:4, 0:3]
    C[2:4, 3:6] = A[2:4, 0:2]*B[0:2, 3:6] + A[2:4, 2:4]*B[2:4, 3:6]
    
    Each term is a 2x2 * 2x3 product. So we have 8 such terms, each of rank 11.
    Total rank = 8 * 11 = 88. That's not great.
    
    Alternatively, partition more coarsely:
    Partition A's rows into {0,1,2,3} (all), A's columns into {0,1} and {2,3}.
    Partition B's rows into {0,1} and {2,3}, B's columns into {0,1,2,3,4,5} (all).
    
    C = A[:, 0:2] * B[0:2, :] + A[:, 2:4] * B[2:4, :]
    
    Each term: 4x2 * 2x6 = 4x6. Split each into 2x2x3 + 2x2x3? No, 4x2 * 2x6 can be split by rows of C into two 2x2*2x6 terms.
    
    This is getting complicated. Let me use a simpler approach:
    
    Just use the block structure directly. Partition A into 2x2 blocks, B into 2x3 blocks.
    A has 2 row-blocks and 2 col-blocks (each 2x2).
    B has 2 row-blocks and 2 col-blocks (each 2x3).
    
    C[i_block, k_block] = sum_j_block A[i_block, j_block] * B[j_block, k_block]
    
    So C[0:2, 0:3] = A[0:2,0:2]*B[0:2,0:3] + A[0:2,2:4]*B[2:4,0:3]
    C[0:2, 3:6] = A[0:2,0:2]*B[0:2,3:6] + A[0:2,2:4]*B[2:4,3:6]
    C[2:4, 0:3] = A[2:4,0:2]*B[0:2,0:3] + A[2:4,2:4]*B[2:4,0:3]
    C[2:4, 3:6] = A[2:4,0:2]*B[0:2,3:6] + A[2:4,2:4]*B[2:4,3:6]
    
    That's 8 terms of 2x2*2x3, each rank 11. Total = 88.
    
    Or I can be smarter. Let me just do 4 blocks of 2x2x3 where I partition only one dimension:
    
    Partition A's columns into {0,1} and {2,3}, and B's rows into {0,1} and {2,3}.
    C = A[:, 0:2]*B[0:2, :] + A[:, 2:4]*B[2:4, :]
    
    First term: 4x2 * 2x6. Partition C's rows into {0,1} and {2,3}:
    C[0:2, :] = A[0:2, 0:2]*B[0:2, :]  (2x2 * 2x6)
    C[2:4, :] = A[2:4, 0:2]*B[0:2, :]  (2x2 * 2x6)
    
    Each 2x2*2x6 can be split into 2x2x3 + 2x2x3:
    2x2*2x6 = (2x2*2x3) + (2x2*2x3) by splitting B's columns.
    So each is rank 11 + 11 = 22.
    
    Total: 4 * 22 = 88. Same thing.
    
    Let me just go with the straightforward 8-block approach with total rank 88.
    Score = 96/88 = 1.09. Still an improvement over 1.0.
    
    Actually, let me reconsider. Maybe I can use a better decomposition.
    
    What about using Strassen-style for the 4x4x4 part and schoolbook for the 4x4x2 part?
    
    4x4*4x6 = 4x4*4x4 + 4x4*4x2
    
    For 4x4x4, use 2x2x2 Strassen blocks: 4 blocks of 2x2x2, each rank 7. Total = 28.
    For 4x4x2, use schoolbook: 4*4*2 = 32. Or use 2x2x1 blocks: 4 blocks of 2x2x1, each rank 4. Total = 16.
    
    Wait, 4x4*4x2: partition A's rows into {0,1},{2,3}, A's cols into {0,1},{2,3}, B's rows into {0,1},{2,3}, B's cols into {0,1}.
    
    C[i_b, k_b] = sum_j_b A[i_b, j_b] * B[j_b, k_b]
    
    C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    C[2:4, 0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2]
    C[2:4, 2:4] = A[2:4,0:2]*B[0:2,2:4] + A[2:4,2:4]*B[2:4,2:4]
    
    That's 8 terms of 2x2*2x2, each rank 7 (Strassen). Total = 56.
    Plus the 4x4x4 part: 8 terms of 2x2*2x2, each rank 7. Total = 56.
    Grand total = 112. Worse.
    
    OK let me just go with the simplest improvement: 4x4x6 split into 4 blocks of 2x2x3.
    But wait, 4x4x6 can't be directly split into 4 disjoint 2x2x3 subproblems because the middle index j ranges over 4 values, not 2.
    
    The correct decomposition:
    C = A * B where A is 4x4, B is 4x6.
    
    Partition A's rows: {0,1}, {2,3}
    Partition B's columns: {0,1,2}, {3,4,5}
    Partition A's columns and B's rows: {0,1}, {2,3}
    
    C[i_row, k_col] = sum_{j in {0,1,2,3}} A[i_row, j] * B[j, k_col]
                    = sum_{j in {0,1}} A[i_row, j] * B[j, k_col] + sum_{j in {2,3}} A[i_row, j] * B[j, k_col]
    
    So for each (i_row_block, k_col_block) pair, C is a sum of two 2x2*2x3 products.
    There are 2x2 = 4 such (i_row_block, k_col_block) pairs, and each requires 2 terms of 2x2*2x3.
    Total: 8 terms of 2x2*2x3, each rank 11. Total rank = 88.
    
    Let me just implement this. Score = 96/88 ≈ 1.091.
    """
    # Build one 2x2x3 block decomposition
    U_block, V_block, W_block = _block_223()
    block_rank = len(U_block[0])  # 11
    
    # We need 8 blocks:
    # (i_row_block, j_block, k_col_block)
    # i_row_block in {0, 1} -> rows 0-1 or 2-3 of A and C
    # j_block in {0, 1} -> cols 0-1 or 2-3 of A, rows 0-1 or 2-3 of B
    # k_col_block in {0, 1} -> cols 0-2 or 3-5 of B and C
    #
    # Total rank = 8 * 11 = 88
    total_rank = 8 * block_rank
    
    n, m, p = 4, 4, 6
    U = [[0]*total_rank for _ in range(n*m)]  # 16 x 88
    V = [[0]*total_rank for _ in range(m*p)]  # 24 x 88
    W = [[0]*total_rank for _ in range(p*n)]  # 24 x 88
    
    for i_rb in range(2):       # i_row_block: 0 -> rows 0,1; 1 -> rows 2,3
        for j_b in range(2):    # j_block: 0 -> j=0,1; 1 -> j=2,3
            for k_cb in range(2):  # k_col_block: 0 -> k=0,1,2; 1 -> k=3,4,5
                block_idx = i_rb * 4 + j_b * 2 + k_cb
                col_offset = block_idx * block_rank
                
                # A rows: i_rb*2 + {0,1}, A cols: j_b*2 + {0,1}
                # B rows: j_b*2 + {0,1}, B cols: k_cb*3 + {0,1,2}
                # C rows: i_rb*2 + {0,1}, C cols: k_cb*3 + {0,1,2}
                
                # U: A[i, j] where i = i_rb*2 + di, j = j_b*2 + dj
                # U_block has rows indexed by di*2+dj (0..3)
                for di in range(2):
                    for dj in range(2):
                        u_row = (i_rb*2 + di) * m + (j_b*2 + dj)  # A[i, j] index
                        u_block_row = di * 2 + dj  # 0..3
                        for c in range(block_rank):
                            U[u_row][col_offset + c] = U_block[u_block_row][c]
                
                # V: B[j, k] where j = j_b*2 + dj, k = k_cb*3 + dk
                # V_block has rows indexed by dj*3+dk (0..5)
                for dj in range(2):
                    for dk in range(3):
                        v_row = (j_b*2 + dj) * p + (k_cb*3 + dk)  # B[j, k] index
                        v_block_row = dj * 3 + dk  # 0..5
                        for c in range(block_rank):
                            V[v_row][col_offset + c] = V_block[v_block_row][c]
                
                # W: C[k, i] where k = k_cb*3 + dk, i = i_rb*2 + di
                # W_block has rows indexed by dk*2+di (0..5)
                for dk in range(3):
                    for di in range(2):
                        w_row = (k_cb*3 + dk) * n + (i_rb*2 + di)  # C[k, i] -> W[k*n+i]
                        w_block_row = dk * 2 + di  # 0..5
                        for c in range(block_rank):
                            W[w_row][col_offset + c] = W_block[w_block_row][c]
    
    return {"U": U, "V": V, "W": W}
