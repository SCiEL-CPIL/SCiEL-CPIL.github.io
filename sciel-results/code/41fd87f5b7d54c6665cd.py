"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook multiplication."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_222():
    """
    Return (U, V, W) for 2x2x2 Strassen multiplication.
    Shapes: U (4, 7), V (4, 7), W (4, 7).
    Indices:
    U rows: (i, j) -> i*2+j
    V rows: (j, k) -> j*2+k
    W rows: (k, i) -> k*2+i
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 + A11)(B00 + B11)
    # U: A00 (0,0), A11 (1,1)
    U[0][0] = 1; U[3][0] = 1
    # V: B00 (0,0), B11 (1,1)
    V[0][0] = 1; V[3][0] = 1
    # W: C00 (0,0), C11 (1,1)
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10 + A11)B00
    # U: A10 (1,0), A11 (1,1)
    U[2][1] = 1; U[3][1] = 1
    # V: B00 (0,0)
    V[0][1] = 1
    # W: C10 (0,1)
    W[2][1] = 1
    
    # M3 = A00(B01 - B11)
    # U: A00 (0,0)
    U[0][2] = 1
    # V: B01 (0,1), B11 (1,1)
    V[1][2] = 1; V[3][2] = -1
    # W: C01 (1,0)
    W[1][2] = 1
    
    # M4 = A11(B10 - B00)
    # U: A11 (1,1)
    U[3][3] = 1
    # V: B10 (1,0), B00 (0,0)
    V[2][3] = 1; V[0][3] = -1
    # W: C00 (0,0), C10 (0,1)
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00 + A01)B11
    # U: A00 (0,0), A01 (0,1)
    U[0][4] = 1; U[1][4] = 1
    # V: B11 (1,1)
    V[3][4] = 1
    # W: C00 (0,0), C01 (1,0)
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    # U: A10 (1,0), A00 (0,0)
    U[2][5] = 1; U[0][5] = -1
    # V: B00 (0,0), B01 (0,1)
    V[0][5] = 1; V[1][5] = 1
    # W: C11 (1,1)
    W[3][5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    # U: A01 (0,1), A11 (1,1)
    U[1][6] = 1; U[3][6] = -1
    # V: B10 (1,0), B11 (1,1)
    V[2][6] = 1; V[3][6] = 1
    # W: C00 (0,0)
    W[0][6] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition."""
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
        
    # Efficient check using numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                x = U_3d[i, j, :] * V_3d[j, k, :]
                val = np.dot(x, W_3d[k, i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _combine_block_factors(global_U, global_V, global_W,
                           local_U, local_V, local_W,
                           i_start, j_start, k_start,
                           n_block, m_block, p_block,
                           col_offset):
    """
    Place local factors into global factors.
    
    local_U: (n_block*m_block, rank_local)
    local_V: (m_block*p_block, rank_local)
    local_W: (p_block*n_block, rank_local)
    
    Global indices:
    U row: i*m + j
    V row: j*p + k
    W row: k*n + i
    
    Local indices (within block):
    U row: i_l*m_block + j_l
    V row: j_l*p_block + k_l
    W row: k_l*n_block + i_l
    
    Global i = i_start + i_l
    Global j = j_start + j_l
    Global k = k_start + k_l
    """
    rank_local = len(local_U[0])
    
    # Precompute row mappings
    # For U: global_row = (i_start + i_l)*m + (j_start + j_l)
    # i_l = u_row // m_block, j_l = u_row % m_block
    u_rows = []
    for u_row in range(n_block * m_block):
        i_l = u_row // m_block
        j_l = u_row % m_block
        i_g = i_start + i_l
        j_g = j_start + j_l
        u_rows.append(i_g * m + j_g)
        
    # For V: global_row = (j_start + j_l)*p + (k_start + k_l)
    # j_l = v_row // p_block, k_l = v_row % p_block
    v_rows = []
    for v_row in range(m_block * p_block):
        j_l = v_row // p_block
        k_l = v_row % p_block
        j_g = j_start + j_l
        k_g = k_start + k_l
        v_rows.append(j_g * p + k_g)
        
    # For W: global_row = (k_start + k_l)*n + (i_start + i_l)
    # k_l = w_row // n_block, i_l = w_row % n_block
    w_rows = []
    for w_row in range(p_block * n_block):
        k_l = w_row // n_block
        i_l = w_row % n_block
        k_g = k_start + k_l
        i_g = i_start + i_l
        w_rows.append(k_g * n + i_g)
        
    # Copy data
    for c in range(rank_local):
        for local_idx, global_idx in enumerate(u_rows):
            global_U[global_idx][col_offset + c] = local_U[local_idx][c]
        for local_idx, global_idx in enumerate(v_rows):
            global_V[global_idx][col_offset + c] = local_V[local_idx][c]
        for local_idx, global_idx in enumerate(w_rows):
            global_W[global_idx][col_offset + c] = local_W[local_idx][c]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    deadline = time.time() + 140  # Safety margin
    
    # Strategy: Block decomposition
    # Partition A (3x5) into rows: {0}, {1,2}
    # Partition B (5x5) into cols: {0,1}, {2,3,4}
    # Wait, the prompt suggests:
    # "Represent the 5x5 matrix multiplication as a 2x2 block multiplication of sizes (2+3) by (2+3)"
    # This implies partitioning the index sets of the multiplication.
    # A is 3x5 (indices i in 0..2, j in 0..4)
    # B is 5x5 (indices j in 0..4, k in 0..4)
    # C is 3x5 (indices i in 0..2, k in 0..4)
    
    # Let's partition the "middle" index j into J1={0,1} and J2={2,3,4}.
    # And the "output" index k into K1={0,1} and K2={2,3,4}.
    # And the "input" index i into I1={0} and I2={1,2}.
    
    # This creates 2x2x2 blocks based on (I, J, K) partitions?
    # No, the tensor is T[i, j, k].
    # We can decompose the sum over j and k.
    # C[i, k] = sum_j A[i, j] B[j, k].
    
    # Let's use the block structure on the (j, k) plane of B and (i, j) plane of A.
    # Actually, the standard block matrix multiplication decomposition is:
    # If we partition A into blocks A_{ip, jq}, B into B_{jq, kp}, C into C_{ip, kp}.
    # Here n=3, m=5, p=5.
    # Let's partition I (rows of A/C) into I1={0}, I2={1,2}. Sizes 1, 2.
    # Let's partition J (cols of A/rows of B) into J1={0,1}, J2={2,3,4}. Sizes 2, 3.
    # Let's partition K (cols of B/C) into K1={0,1}, K2={2,3,4}. Sizes 2, 3.
    
    # This gives blocks:
    # A_{11} (1x2), A_{12} (1x3)
    # A_{21} (2x2), A_{22} (2x3)
    # B_{11} (2x2), B_{12} (2x3)
    # B_{21} (3x2), B_{22} (3x3)
    # C_{11} (1x2), C_{12} (1x3)
    # C_{21} (2x2), C_{22} (2x3)
    
    # The tensor T[i, j, k] is non-zero only if (i, j, k) are compatible.
    # We can decompose the problem into 4 sub-problems based on the J and K partitions?
    # No, the multiplication is sum over J.
    # C[i, k] = sum_{j in J1} A[i, j] B[j, k] + sum_{j in J2} A[i, j] B[j, k].
    
    # This doesn't immediately give 4 independent tensor decompositions because the sums are coupled in C.
    # However, we can view the tensor as a sum of tensors corresponding to the blocks.
    # T = T_{J1, K1} + T_{J1, K2} + T_{J2, K1} + T_{J2, K2} ?
    # No, T[i, j, k] is the basis element.
    # The decomposition is sum_r U[i, j, r] V[j, k, r] W[k, i, r] = delta_{i, i'} delta_{j, j'} delta_{k, k'}?
    # No, the target tensor for matrix multiplication is T[i, j, k] = 1 for all i, j, k?
    # Wait, the tensor for matrix multiplication C = AB is:
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T_{ijk} such that C_{ik} = sum_{j} T_{ijk} A_{ij} B_{jk} is T_{ijk} = 1 for all i, j, k.
    # Yes, the task says "exactly decompose the matrix multiplication tensor".
    # The tensor is T_{ijk} = 1.
    
    # So we need sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1 for all i, j, k.
    
    # We can partition the set of (i, j, k) tuples into disjoint sets and solve for each set?
    # No, the decomposition is global.
    # But we can use the block structure to build a decomposition.
    # Let's partition the indices I, J, K.
    # I = I1 U I2, J = J1 U J2, K = K1 U K2.
    # We can construct the decomposition by summing decompositions for each block (I_a, J_b, K_c).
    # For a fixed block (I_a, J_b, K_c), the sub-tensor T_{ijk} is 1 if i in I_a, j in J_b, k in K_c, and 0 otherwise.
    # This sub-tensor is the tensor for multiplying a matrix of size |I_a|x|J_b| by |J_b|x|K_c|.
    # Wait, if i in I_a, j in J_b, k in K_c, then the term contributes to C_{ik}.
    # The sum over all blocks gives the full tensor.
    # Since the blocks are disjoint in (i, j, k) space, we can solve each block independently and concatenate the ranks.
    
    # Blocks:
    # 1. I1 (1), J1 (2), K1 (2) -> 1x2x2 multiplication. Rank = 1*2*2 = 4 (schoolbook) or less?
    # 2. I1 (1), J1 (2), K2 (3) -> 1x2x3 multiplication. Rank = 1*2*3 = 6.
    # 3. I1 (1), J2 (3), K1 (2) -> 1x3x2 multiplication. Rank = 1*3*2 = 6.
    # 4. I1 (1), J2 (3), K2 (3) -> 1x3x3 multiplication. Rank = 1*3*3 = 9.
    # 5. I2 (2), J1 (2), K1 (2) -> 2x2x2 multiplication. Rank = 7 (Strassen).
    # 6. I2 (2), J1 (2), K2 (3) -> 2x2x3 multiplication. Rank = 2*2*3 = 12.
    # 7. I2 (2), J2 (3), K1 (2) -> 2x3x2 multiplication. Rank = 2*3*2 = 12.
    # 8. I2 (2), J2 (3), K2 (3) -> 2x3x3 multiplication. Rank = 2*3*3 = 18.
    
    # Total rank = 4 + 6 + 6 + 9 + 7 + 12 + 12 + 18 = 74.
    # This is only 1 better than 75.
    
    # Can we do better?
    # The prompt suggests: "applying the rank-7 Strassen algorithm to the 2x2x2 sub-blocks and the schoolbook algorithm to the 3x3x3 sub-blocks"
    # This implies a different partition.
    # Maybe partition J into {0,1} and {2,3,4} and K into {0,1} and {2,3,4} is not the only way.
    # What if we partition I into {0} and {1,2}, J into {0,1,2} and {3,4}, K into {0,1,2} and {3,4}?
    # Sizes: I: 1, 2. J: 3, 2. K: 3, 2.
    # Blocks:
    # 1. 1x3x3 -> 9
    # 2. 1x3x2 -> 6
    # 3. 1x2x3 -> 6
    # 4. 1x2x2 -> 4
    # 5. 2x3x3 -> 18
    # 6. 2x3x2 -> 12
    # 7. 2x2x3 -> 12
    # 8. 2x2x2 -> 7
    # Total = 9+6+6+4+18+12+12+7 = 74. Still 74.
    
    # To get 48, we need much better compression.
    # The prompt says: "total rank of 3 * 7 + 3 * 9 = 48".
    # This implies 3 Strassen (7) and 3 Schoolbook 3x3x3 (9)?
    # 3*7 + 3*9 = 21 + 27 = 48.
    # This suggests a partition where we have three 2x2x2 blocks and three 3x3x3 blocks?
    # Or maybe the partition is different.
    
    # Let's look at the dimensions again. 3x5x5.
    # Maybe partition J into 3 blocks of sizes 2, 2, 1? No.
    # Maybe the "3" in "3 * 7" refers to the 3 rows of A?
    # If we treat each row of A separately?
    # Row i of C is sum_j A[i, j] Row_j of B.
    # This is a sum of 5 vector-matrix multiplications.
    # This doesn't seem to lead to 48.
    
    # Let's try to find a decomposition with rank < 75.
    # The simple block sum gave 74.
    # Can we improve the 1x2x2, 1x2x3, etc blocks?
    # 1x2x2: Rank 4. Can it be 3?
    # T_{ijk} = 1 for i=0, j in {0,1}, k in {0,1}.
    # This is equivalent to multiplying a 1x2 matrix by a 2x2 matrix.
    # C_{0k} = A_{00}B_{0k} + A_{01}B_{1k}.
    # This is 4 multiplications.
    # Can we do it in 3?
    # Strassen for 2x2 is rank 7.
    # For 1x2x2, it's a special case.
    # Actually, 1x2x2 is just 2 vector multiplications of length 2? No.
    # It's 4 scalar multiplications.
    # Is there a rank 3 decomposition?
    # Lower bound for 1x2x2 is 4?
    # The tensor is the sum of 4 basis tensors.
    # If we can combine them, maybe.
    # But generally, 1xn x nx1 is rank n.
    # 1x2x2 is not 1x2x1.
    
    # Let's stick to the 74 solution for now, as it is an improvement.
    # We can implement the 8 blocks.
    
    # Block definitions:
    # I1 = [0], I2 = [1, 2]
    # J1 = [0, 1], J2 = [2, 3, 4]
    # K1 = [0, 1], K2 = [2, 3, 4]
    
    # Blocks (I_part, J_part, K_part):
    # 1: (I1, J1, K1) -> 1x2x2
    # 2: (I1, J1, K2) -> 1x2x3
    # 3: (I1, J2, K1) -> 1x3x2
    # 4: (I1, J2, K2) -> 1x3x3
    # 5: (I2, J1, K1) -> 2x2x2
    # 6: (I2, J1, K2) -> 2x2x3
    # 7: (I2, J2, K1) -> 2x3x2
    # 8: (I2, J2, K2) -> 2x3x3
    
    # We will use Strassen for 2x2x2 (Block 5).
    # Schoolbook for others.
    
    # Total Rank = 4 + 6 + 6 + 9 + 7 + 12 + 12 + 18 = 74.
    
    # Let's implement this.
    
    # Pre-compute local factors for each block type
    # We need a function to generate factors for n x m x p
    def _get_factors(n_loc, m_loc, p_loc):
        if n_loc == 2 and m_loc == 2 and p_loc == 2:
            return _strassen_222()
        else:
            return _schoolbook_factors(n_loc, m_loc, p_loc)
            
    # Define partitions
    I_parts = [[0], [1, 2]]
    J_parts = [[0, 1], [2, 3, 4]]
    K_parts = [[0, 1], [2, 3, 4]]
    
    # Initialize global factors
    # We don't know the total rank yet, so let's compute it.
    total_rank = 0
    blocks = []
    for i_idx, i_part in enumerate(I_parts):
        for j_idx, j_part in enumerate(J_parts):
            for k_idx, k_part in enumerate(K_parts):
                n_loc = len(i_part)
                m_loc = len(j_part)
                p_loc = len(k_part)
                U_loc, V_loc, W_loc = _get_factors(n_loc, m_loc, p_loc)
                rank_loc = len(U_loc[0])
                total_rank += rank_loc
                blocks.append({
                    'i_part': i_part, 'j_part': j_part, 'k_part': k_part,
                    'U': U_loc, 'V': V_loc, 'W': W_loc,
                    'rank': rank_loc,
                    'offset': total_rank - rank_loc
                })
                
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for b in blocks:
        i_part = b['i_part']
        j_part = b['j_part']
        k_part = b['k_part']
        i_start = i_part[0]
        j_start = j_part[0]
        k_start = k_part[0]
        n_loc = len(i_part)
        m_loc = len(j_part)
        p_loc = len(k_part)
        
        _combine_block_factors(U, V, W,
                               b['U'], b['V'], b['W'],
                               i_start, j_start, k_start,
                               n_loc, m_loc, p_loc,
                               b['offset'])
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook_factors(n, m, p)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
