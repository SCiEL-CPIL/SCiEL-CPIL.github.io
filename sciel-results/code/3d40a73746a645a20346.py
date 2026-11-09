"""
Rank-minimal tensor decomposition for 4x4 * 4x6 matrix multiplication.

Strategy:
  - The multiplication tensor has rank 96 (schoolbook).
  - We use a block decomposition: split A (4x4) into 2x2 blocks of size 2x2,
    B (4x6) into 2x3 blocks, C (4x6) into 2x3 blocks.
  - For each pair of blocks (A_block, B_block) -> C_block, we need a decomposition
    of 2x2 * 2x3 -> 2x3.
  - The minimal rank of 2x2 * 2x3 is 6 (since 2*2*3=12, and rank of matrix mult
    tensor for m x n * n x p is at most m*n*p, but for 2x2x3 it's known to be 6?
    Actually, let me think: the tensor T_{i,j,k} for (A_{i,j}, B_{j,k}) -> C_{i,k}
    with i in {0,1}, j in {0,1}, k in {0,1,2}. The rank of this tensor...
    
    For 2x2x2, rank is 7 (Strassen). For 2x2x3, we can use a similar approach.
    Actually, the rank of the 2x2x3 tensor: we need to compute C_{i,k} = sum_j A_{i,j} B_{j,k}.
    This is a bilinear map from R^4 x R^6 -> R^6.
    
    Let me think of it as: for each i, C_{i,0:3} = A_{i,0} * B_{0,0:3} + A_{i,1} * B_{1,0:3}.
    So C = A * B where A is 2x2 and B is 2x3.
    
    The tensor has indices (i, j, k) with i in [0,2), j in [0,2), k in [0,3).
    Rank of this tensor: it's the rank of the matrix multiplication tensor for 2x2 and 2x3.
    Known result: rank of 2x2x2 is 7. For 2x2x3, I believe it's 6 or 7.
    
    Actually, let me just use a schoolbook decomposition for 2x2x3 which has rank 12,
    but that gives total rank 96 again.
    
    Let me try a different approach. Use the known rank-7 decomposition for 2x2x2
    and handle the 2x2x3 case by splitting the 2x3 into 2x2 + 2x1.
    
    For 2x2 * 2x1: C is 2x1. C_{i} = sum_j A_{i,j} B_{j,0}.
    This is a bilinear map from R^4 x R^2 -> R^2.
    The tensor has indices (i, j, k=0) with i in [0,2), j in [0,2).
    Rank of 2x2x1 tensor: this is just the rank of the "vector times matrix" type.
    C_i = A_{i,0} B_{0,0} + A_{i,1} B_{1,0}.
    This can be done with rank 2: two terms.
    Actually, it's rank 2: U has 4 rows, V has 2 rows, W has 2 rows.
    Term 1: A_{0,0} * B_{0,0} -> contributes to C_0
    Term 2: A_{1,0} * B_{0,0} -> contributes to C_1
    Term 3: A_{0,1} * B_{1,0} -> contributes to C_0
    Term 4: A_{1,1} * B_{1,0} -> contributes to C_1
    That's rank 4 schoolbook. Can we do better?
    
    C_0 = A_{0,0} B_{0,0} + A_{0,1} B_{1,0}
    C_1 = A_{1,0} B_{0,0} + A_{1,1} B_{1,0}
    
    This is like C = A * b where A is 2x2 and b is 2x1.
    The tensor rank of 2x2x1... I think it's 2? No, it's 4 because each output depends
    on different products.
    
    Actually, let me reconsider. The tensor for 2x2 * 2x1:
    T_{i,j,k} where i is row of C (0,1), j is col of A / row of B (0,1), k is col of C (only 0).
    T_{i,j,0} = 1 for all i,j.
    This is a 2x2x1 tensor. Its rank: we need to express the bilinear form.
    C_i = sum_j A_{i,j} B_{j,0}
    This is a set of 2 bilinear forms, each of rank 2 (as a matrix).
    The tensor rank is the minimum r such that we can write T as sum of r simple tensors.
    For 2x2x1, I believe the rank is 2. Let's see:
    Simple tensor: u_i * v_j * w_k.
    We need sum_r u_{i,r} v_{j,r} w_{0,r} = 1 for all i,j.
    w_{0,r} is just a scalar for each r.
    So we need sum_r (w_{0,r}) u_{i,r} v_{j,r} = 1 for all i,j.
    This means the sum of rank-1 matrices (w_{0,r} u_r v_r^T) equals the all-ones 2x2 matrix.
    The all-ones 2x2 matrix has rank 1, so we can do it with r=1:
    w_{0,1} = 1, u_{i,1} = 1, v_{j,1} = 1.
    So rank 1! C_i = A_{i,0} B_{0,0} + A_{i,1} B_{1,0}... wait, that's not right.
    
    Let me re-examine. The tensor T for multiplying A (2x2) by B (2x1) to get C (2x1):
    C_{i,0} = sum_{j=0}^{1} A_{i,j} B_{j,0}
    The tensor has components T_{i,j,k} where i indexes rows of C, j indexes the shared dimension, k indexes columns of C.
    T_{i,j,k} = delta_{i,k} ... no.
    
    Actually, in the standard tensor notation for matrix multiplication C = AB:
    T_{i,j,k} = 1 if the product A_{i,j} * B_{j,k} contributes to C_{i,k}.
    So T_{i,j,k} = 1 for all valid i,j,k (it's the all-ones tensor in the appropriate shape).
    
    For 2x2 * 2x1: i in {0,1}, j in {0,1}, k in {0}. T_{i,j,0} = 1 for all i,j.
    This is a 2x2x1 tensor. Rank = ?
    
    We want to write T_{i,j,0} = sum_r U_{i,r} V_{j,r} W_{0,r}.
    Since k=0 only, W_{0,r} is a scalar. Let w_r = W_{0,r}.
    Then sum_r w_r U_{i,r} V_{j,r} = 1 for all i,j.
    This is sum_r (w_r U_r V_r^T) = J (all-ones 2x2 matrix).
    J has rank 1, so we can write J = w_1 U_1 V_1^T with w_1=1, U_1=(1,1), V_1=(1,1).
    So rank 1!
    
    Similarly, for 2x1 * 1x2: A is 2x1, B is 1x2, C is 2x2.
    C_{i,k} = A_{i,0} B_{0,k}.
    T_{i,j,k} with j=0 only. T_{i,0,k} = 1 for all i,k.
    Sum_r U_{i,r} V_{0,r} W_{k,r} = 1 for all i,k.
    This is sum_r (U_r W_r^T) * V_{0,r} = J.
    J is 2x2 all-ones, rank 1. So rank 1.
    
    For 1x2 * 2x1: A is 1x2, B is 2x1, C is 1x1.
    C_{0,0} = A_{0,0} B_{0,0} + A_{0,1} B_{1,0}.
    T_{i,j,k} with i=0, k=0 only. T_{0,j,0} = 1 for j=0,1.
    Sum_r U_{0,r} V_{j,r} W_{0,r} = 1 for j=0,1.
    Let u_r = U_{0,r}, w_r = W_{0,r}.
    sum_r u_r w_r V_{j,r} = 1 for j=0,1.
    This is a 2-component vector equation. We can do it with r=1:
    u_1 w_1 V_{0,1} = 1, u_1 w_1 V_{1,1} = 1.
    So V_{0,1}=V_{1,1}=1, u_1 w_1 = 1. Rank 1.
    
    OK so now for 2x2 * 2x3:
    We can split B (2x3) into B_left (2x2) and B_right (2x1).
    C (2x3) = A * B_left (2x2) + A * B_right (2x1).
    A * B_left: 2x2 * 2x2, rank 7 (Strassen).
    A * B_right: 2x2 * 2x1, rank 1 as shown above.
    Total: 7 + 1 = 8.
    
    But wait, we need to be careful. The decomposition must be for the full tensor.
    Let me just use rank 8 for 2x2x3.
    
    Actually, I realize I should just hardcode a known decomposition.
    
    Let me use a simpler approach: use the schoolbook for the whole thing but try to find
    a lower rank decomposition using known results.
    
    For 4x4 * 4x6, the rank is at most 96. Can we do better?
    
    Let me try: split into 2x2 blocks.
    A = [[A00, A01], [A10, A11]] where each is 2x2.
    B = [[B00, B01, B02], [B10, B11, B12]] where each is 2x2 or 2x1... 
    Actually B is 4x6, so blocks are 2x3.
    B = [[B00, B01], [B10, B11]] where each is 2x3.
    C = [[C00, C01], [C10, C11]] where each is 2x3.
    
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    Each Ci,j is 2x3.
    
    For each product A_block * B_block (2x2 * 2x3 -> 2x3), we need a decomposition.
    As computed, 2x2 * 2x3 can be done with rank 8 (7 for 2x2x2 part + 1 for 2x2x1 part).
    
    There are 8 such products (4 A-blocks * 2 B-blocks per row... wait).
    A has 4 blocks: A00, A01, A10, A11.
    B has 4 blocks: B00, B01, B10, B11.
    C has 4 blocks: C00, C01, C10, C11.
    
    C00 = A00*B00 + A01*B10: two 2x2x3 products.
    C01 = A00*B01 + A01*B11: two 2x2x3 products.
    C10 = A10*B00 + A11*B10: two 2x2x3 products.
    C11 = A10*B01 + A11*B11: two 2x2x3 products.
    
    Total 8 products of 2x2x3. If each has rank 8, total rank 64.
    
    Can we do better with Strassen-like recursion?
    Strassen for 2x2x2 uses 7 multiplications. For 2x2x3, maybe we can use 7 as well?
    
    Actually, let me just implement the rank-8 decomposition for 2x2x3 and use 8 of them.
    Total rank 64.
    
    Let me define the 2x2x3 decomposition.
    A is 2x2: indices (i,j) with i,j in {0,1}.
    B is 2x3: indices (j,k) with j in {0,1}, k in {0,1,2}.
    C is 2x3: indices (i,k) with i in {0,1}, k in {0,1,2}.
    
    C_{i,k} = sum_j A_{i,j} B_{j,k}.
    
    Split B into B_left (2x2, cols 0,1) and B_right (2x1, col 2).
    C_left (2x2) = A * B_left: use Strassen rank 7.
    C_right (2x1) = A * B_right: rank 1.
    
    Strassen for 2x2x2:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 - M4 + M6
    C01 = M5 + M7
    C10 = M2 + M4
    C11 = M1 - M3 + M5
    
    For 2x2x2, U has 4 rows (A indices), V has 4 rows (B indices), W has 4 rows (C indices).
    
    For 2x2x3, U has 4 rows, V has 6 rows, W has 6 rows.
    
    Let me construct it.
    """
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We'll build the decomposition using block structure.
    # Split A (4x4) into 2x2 blocks of size 2x2.
    # Split B (4x6) into 2x3 blocks.
    # Split C (4x6) into 2x3 blocks.
    
    # For each 2x2 * 2x3 product, we use rank 8 decomposition.
    # There are 8 such products total.
    # Total rank = 8 * 8 = 64.
    
    rank = 64
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 64
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 64
    W = [[0.0] * rank for _ in range(p * n)]  # 24 x 64
    
    # Define the 2x2x3 decomposition (rank 8).
    # Local indices: A is 2x2 (rows 0-1, cols 0-1), B is 2x3 (rows 0-1, cols 0-2), C is 2x3 (rows 0-1, cols 0-2).
    # U_local: 4 x 8, V_local: 6 x 8, W_local: 6 x 8.
    
    # Strassen for the 2x2x2 part (cols 0,1 of B -> cols 0,1 of C):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # For 2x2x2:
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # M1: A00+A11 -> U: [1,0,0,1], B00+B11 -> V: [1,0,0,1], C00+C11 -> W: [1,0,0,1]
    # M2: A10+A11 -> U: [0,0,1,1], B00 -> V: [1,0,0,0], C10 -> W: [0,0,1,0]
    # M3: A00 -> U: [1,0,0,0], B10-B11 -> V: [0,0,1,-1], C11 -> W: [0,0,0,1]
    # M4: A11 -> U: [0,0,0,1], B01-B00 -> V: [-1,1,0,0], C00 -> W: [1,0,0,0]
    # M5: A00+A01 -> U: [1,1,0,0], B11 -> V: [0,0,0,1], C01 -> W: [0,1,0,0]
    # M6: A10-A00 -> U: [-1,0,1,0], B00+B10 -> V: [1,0,1,0], C00 -> W: [1,0,0,0]
    # M7: A01-A11 -> U: [0,1,0,-1], B01+B11 -> V: [0,1,0,1], C01 -> W: [0,1,0,0]
    
    # For the 2x2x1 part (col 2 of B -> col 2 of C):
    # C_{i,2} = A_{i,0} B_{0,2} + A_{i,1} B_{1,2}
    # Rank 1: U: [1,1,1,1], V: [0,0,0,0,1,1] (B_{0,2} and B_{1,2}), W: [0,0,0,0,1,1] (C_{0,2} and C_{1,2})
    # Wait, W for C_{i,2}: C indices are (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5.
    # So W: [0,0,1,0,0,1]
    
    # Let me build the local 2x2x3 decomposition.
    U_local = [[0.0] * 8 for _ in range(4)]
    V_local = [[0.0] * 8 for _ in range(6)]
    W_local = [[0.0] * 8 for _ in range(6)]
    
    # M1
    U_local[0][0] += 1; U_local[3][0] += 1
    V_local[0][0] += 1; V_local[3][0] += 1
    W_local[0][0] += 1; W_local[3][0] += 1
    # M2
    U_local[2][1] += 1; U_local[3][1] += 1
    V_local[0][1] += 1
    W_local[2][1] += 1
    # M3
    U_local[0][2] += 1
    V_local[2][2] += 1; V_local[3][2] -= 1
    W_local[3][2] += 1
    # M4
    U_local[3][3] += 1
    V_local[1][3] -= 1; V_local[0][3] += 1
    W_local[0][3] += 1
    # M5
    U_local[0][4] += 1; U_local[1][4] += 1
    V_local[3][4] += 1
    W_local[1][4] += 1
    # M6
    U_local[2][5] += 1; U_local[0][5] -= 1
    V_local[0][5] += 1; V_local[2][5] += 1
    W_local[0][5] += 1
    # M7
    U_local[1][6] += 1; U_local[3][6] -= 1
    V_local[1][6] += 1; V_local[3][6] += 1
    W_local[1][6] += 1
    # M8 (2x2x1 part)
    U_local[0][7] += 1; U_local[1][7] += 1; U_local[2][7] += 1; U_local[3][7] += 1
    V_local[4][7] += 1; V_local[5][7] += 1
    W_local[2][7] += 1; W_local[5][7] += 1
    
    # Now combine 8 blocks.
    # Block (bi, bj) where bi is A block row (0,1), bj is B block col (0,1).
    # A block (bi, bj_a) where bj_a is A block col (0,1).
    # B block (bi_b, bj) where bi_b is B block row (0,1).
    # C block (bi, bj) where bi is C block row, bj is C block col.
    
    # C_{bi, bj} = A_{bi, 0} * B_{0, bj} + A_{bi, 1} * B_{1, bj}
    
    # So for each (bi, bj):
    #   Product 1: A_{bi, 0} * B_{0, bj} -> contributes to C_{bi, bj}
    #   Product 2: A_{bi, 1} * B_{1, bj} -> contributes to C_{bi, bj}
    
    # A block (bi, ja) has rows 2*bi to 2*bi+1, cols 2*ja to 2*ja+1.
    # B block (jb, bj) has rows 2*jb to 2*jb+1, cols 3*bj to 3*bj+2.
    # C block (bi, bj) has rows 2*bi to 2*bi+1, cols 3*bj to 3*bj+2.
    
    # W index: k*n + i where k is col of C, i is row of C.
    # C row i_global = 2*bi + i_local, C col k_global = 3*bj + k_local.
    # W index = k_global * n + i_global = (3*bj + k_local) * 4 + (2*bi + i_local).
    
    # U index: i*m + j where i is row of A, j is col of A.
    # A row i_global = 2*bi + i_local, A col j_global = 2*ja + j_local.
    # U index = i_global * m + j_global = (2*bi + i_local) * 4 + (2*ja + j_local).
    
    # V index: j*p + k where j is row of B, k is col of B.
    # B row j_global = 2*jb + j_local, B col k_global = 3*bj + k_local.
    # V index = j_global * p + k_global = (2*jb + j_local) * 6 + (3*bj + k_local).
    
    col_offset = 0
    for bi in range(2):
        for bj in range(2):
            for ja in range(2):  # A block col
                for jb in range(2):  # B block row
                    if ja != jb:
                        continue  # A_{bi, ja} * B_{jb, bj} only when ja == jb? No!
                    # Actually, C_{bi, bj} = sum_{jb} A_{bi, jb} * B_{jb, bj}
                    # So for fixed (bi, bj), we sum over jb=0,1.
                    # A_{bi, jb} * B_{jb, bj}
                    
                    # A block: rows 2*bi to 2*bi+1, cols 2*jb to 2*jb+1.
                    # B block: rows 2*jb to 2*jb+1, cols 3*bj to 3*bj+2.
                    # C block: rows 2*bi to 2*bi+1, cols 3*bj to 3*bj+2.
                    
                    for r in range(8):
                        for il in range(2):
                            for jl in range(2):
                                u_idx = (2*bi + il) * 4 + (2*jb + jl)
                                U[u_idx][col_offset + r] += U_local[il * 2 + jl][r]
                        for jl in range(2):
                            for kl in range(3):
                                v_idx = (2*jb + jl) * 6 + (3*bj + kl)
                                V[v_idx][col_offset + r] += V_local[jl * 3 + kl][r]
                        for il in range(2):
                            for kl in range(3):
                                w_idx = (3*bj + kl) * 4 + (2*bi + il)
                                W[w_idx][col_offset + r] += W_local[il * 3 + kl][r]
                    col_offset += 8
    
    # Convert to lists of integers/half-integers
    U_out = [[float(x) for x in row] for row in U]
    V_out = [[float(x) for x in row] for row in V]
    W_out = [[float(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
