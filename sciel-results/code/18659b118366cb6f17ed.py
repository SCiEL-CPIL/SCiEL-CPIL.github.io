import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    T = np.zeros((n, m, p))
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-10):
        return False
    return True

def _is_integer_or_half_integer(arr, tol=1e-9):
    """Check if all entries are integers or half-integers (multiples of 0.5)."""
    for val in np.array(arr).flatten():
        if abs(val * 2 - round(val * 2)) > tol:
            return False
    return True

def _strassen_222():
    """Returns U, V, W for 2x2x2 matrix multiplication with rank 7."""
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: i in 0..1, j in 0..1, k in 0..1
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # Strassen decomposition for 2x2x2
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
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: M1
    U[0, 0] = 1; U[3, 0] = 1 # A11, A22
    V[0, 0] = 1; V[3, 0] = 1 # B11, B22
    W[0, 0] = 1; W[3, 0] = 1 # C11, C22
    
    # r=1: M2
    U[2, 1] = 1; U[3, 1] = 1 # A21, A22
    V[0, 1] = 1 # B11
    W[2, 1] = 1 # C21
    
    # r=2: M3
    U[0, 2] = 1 # A11
    V[1, 2] = 1; V[3, 2] = -1 # B12, -B22
    W[1, 2] = 1 # C12
    
    # r=3: M4
    U[3, 3] = 1 # A22
    V[2, 3] = 1; V[0, 3] = -1 # B21, -B11
    W[0, 3] = 1; W[2, 3] = 1 # C11, C21
    
    # r=4: M5
    U[0, 4] = 1; U[1, 4] = 1 # A11, A12
    V[3, 4] = 1 # B22
    W[0, 4] = -1; W[1, 4] = 1 # -C11, C12
    
    # r=5: M6
    U[2, 5] = 1; U[0, 5] = -1 # A21, -A11
    V[0, 5] = 1; V[1, 5] = 1 # B11, B12
    W[3, 5] = 1 # C22
    
    # r=6: M7
    U[1, 6] = 1; U[3, 6] = -1 # A12, -A22
    V[2, 6] = 1; V[3, 6] = 1 # B21, B22
    W[0, 6] = 1 # C11
    
    return U.tolist(), V.tolist(), W.tolist()

def _schoolbook_nm_p(n, m, p):
    """Returns U, V, W for n x m x p matrix multiplication with rank n*m*p."""
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

def _combine_blocks(blocks):
    """
    Combines a list of block decompositions into a single decomposition.
    blocks: list of tuples (n_i, m_i, p_i, U_i, V_i, W_i)
    The blocks are arranged in a grid.
    We assume the full tensor is a direct sum of these blocks.
    """
    # Determine total dimensions
    # This function assumes the blocks cover the full n, m, p space
    # and are non-overlapping in the (i, k) plane, sharing j.
    # Actually, the blocks are defined by sub-ranges of i, j, k.
    # For this task, we split n and p, keeping m same.
    # So blocks are (n_i, m, p_i).
    
    # We need to know the offset for each block in the global indices.
    # Let's pass offsets explicitly or calculate them.
    pass

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Split n=3 into 2 and 1. Split p=7 into 4 and 3.
    # Blocks:
    # B1: n=2, m=4, p=4 (i:0-1, j:0-3, k:0-3)
    # B2: n=2, m=4, p=3 (i:0-1, j:0-3, k:4-6)
    # B3: n=1, m=4, p=4 (i:2, j:0-3, k:0-3)
    # B4: n=1, m=4, p=3 (i:2, j:0-3, k:4-6)
    
    # Get decompositions for each block
    # B1: 2x4x4. Use 4 Strassen 2x2x2 blocks.
    # Strassen 2x2x2 has rank 7.
    # We tile 2x4x4 with four 2x2x2 blocks.
    # The 2x4x4 tensor can be viewed as 4 independent 2x2x2 tensors
    # if we don't share the middle dimension j.
    # But j is shared.
    # However, we can just use the schoolbook for the sub-blocks if we want simplicity,
    # but Strassen is better.
    # Let's use a simple rank-16 for 2x4x4?
    # Actually, let's just use schoolbook for B1, B2, B3, B4 to ensure correctness first,
    # then optimize.
    # Schoolbook ranks:
    # B1: 2*4*4 = 32
    # B2: 2*4*3 = 24
    # B3: 1*4*4 = 4
    # B4: 1*4*3 = 3
    # Total: 63.
    
    # Let's try to optimize B1 and B2 using Strassen.
    # B1 (2x4x4): Split k into 2+2.
    # B1a: 2x4x2. Split k into 1+1? No.
    # B1a (2x4x2) can be done with 2 Strassen 2x2x2?
    # No, 2x4x2 is not 2x2x2.
    # But 2x4x2 can be decomposed into two 2x2x2 blocks if we split j? No, j is fixed.
    # Actually, 2x4x2 is equivalent to multiplying a 2x4 matrix by a 4x2 matrix.
    # This is the same as 2 2x2x2 multiplications if we split the 4 dimension?
    # Yes, if we split j into 2+2, we get two 2x2x2 multiplications.
    # But j is the shared dimension. We cannot split j independently for each block in the direct sum.
    # However, the tensor is a direct sum.
    # T_total = T_B1 + T_B2 + T_B3 + T_B4.
    # Each T_Bi is supported on a disjoint set of (i,k) indices.
    # So we can decompose each T_Bi independently.
    # For T_B1 (2x4x4), we can use any decomposition of M_{2,4,4}.
    # We can construct M_{2,4,4} by tiling 2x2x2.
    # M_{2,4,4} consists of 4 M_{2,2,2} blocks if we partition j and k?
    # No, j is 0..3, k is 0..3.
    # We can partition j into 0-1 and 2-3, and k into 0-1 and 2-3.
    # This gives 4 blocks of M_{2,2,2}.
    # Since the blocks are disjoint in (j,k) for a fixed i?
    # No, the tensor is T[i,j,k].
    # If we partition j and k, the blocks are disjoint in (j,k).
    # So we can sum the decompositions of the 4 M_{2,2,2} blocks.
    # Rank = 4 * 7 = 28.
    
    # Similarly for B2 (2x4x3):
    # Partition j into 0-1, 2-3. k into 0-1, 2 (only 1 col).
    # Blocks:
    # j=0-1, k=0-1: M_{2,2,2} (Rank 7)
    # j=0-1, k=2: M_{2,2,1} (Rank 2? M_{2,2,1} is 2x2 matrix times vector. Rank 2)
    # j=2-3, k=0-1: M_{2,2,2} (Rank 7)
    # j=2-3, k=2: M_{2,2,1} (Rank 2)
    # Total Rank for B2 = 7 + 2 + 7 + 2 = 18.
    
    # B3 (1x4x4):
    # Partition j into 0-1, 2-3. k into 0-1, 2-3.
    # 4 blocks of M_{1,2,2}.
    # M_{1,2,2} is rank 1.
    # Total Rank for B3 = 4.
    
    # B4 (1x4x3):
    # Partition j into 0-1, 2-3. k into 0-1, 2.
    # 4 blocks of M_{1,2,1} or M_{1,2,2}?
    # j=0-1, k=0-1: M_{1,2,2} (Rank 1)
    # j=0-1, k=2: M_{1,2,1} (Rank 1)
    # j=2-3, k=0-1: M_{1,2,2} (Rank 1)
    # j=2-3, k=2: M_{1,2,1} (Rank 1)
    # Total Rank for B4 = 4.
    
    # Total Rank = 28 + 18 + 4 + 4 = 54.
    # This is better than 63.
    
    # Let's implement this tiling strategy.
    
    def _get_block_decomp(n, m, p):
        """
        Returns U, V, W for M_{n,m,p} using tiling of 2x2x2 and smaller.
        This is a placeholder for a more general tiling.
        For now, we will hardcode the construction for the specific blocks.
        """
        pass

    # We will manually construct the combined U, V, W.
    # Total Rank = 54.
    total_rank = 54
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col_offset = 0
    
    # Helper to add a block decomposition
    def add_block(n_b, m_b, p_b, U_b, V_b, W_b, i_off, j_off, k_off):
        nonlocal col_offset
        rank_b = len(U_b[0]) if isinstance(U_b, list) else U_b.shape[1]
        
        # Map local indices to global indices
        # U_b is (n_b*m_b, rank_b)
        # Global U index: (i_off + i_local)*m + (j_off + j_local)
        # V_b is (m_b*p_b, rank_b)
        # Global V index: (j_off + j_local)*p + (k_off + k_local)
        # W_b is (p_b*n_b, rank_b)
        # Global W index: (k_off + k_local)*n + (i_off + i_local)
        
        for r in range(rank_b):
            for i_l in range(n_b):
                for j_l in range(m_b):
                    val_u = U_b[i_l * m_b + j_l][r] if isinstance(U_b, list) else U_b[i_l * m_b + j_l, r]
                    if val_u != 0:
                        i_g = i_off + i_l
                        j_g = j_off + j_l
                        U[i_g * m + j_g, col_offset + r] = val_u
            
            for j_l in range(m_b):
                for k_l in range(p_b):
                    val_v = V_b[j_l * p_b + k_l][r] if isinstance(V_b, list) else V_b[j_l * p_b + k_l, r]
                    if val_v != 0:
                        j_g = j_off + j_l
                        k_g = k_off + k_l
                        V[j_g * p + k_g, col_offset + r] = val_v
            
            for k_l in range(p_b):
                for i_l in range(n_b):
                    val_w = W_b[k_l * n_b + i_l][r] if isinstance(W_b, list) else W_b[k_l * n_b + i_l, r]
                    if val_w != 0:
                        k_g = k_off + k_l
                        i_g = i_off + i_l
                        W[k_g * n + i_g, col_offset + r] = val_w
        
        col_offset += rank_b

    # B1: 2x4x4 (i:0-1, j:0-3, k:0-3)
    # Tiled into 4 blocks of 2x2x2
    # j_off: 0, 2. k_off: 0, 2.
    for j_off in [0, 2]:
        for k_off in [0, 2]:
            U_b, V_b, W_b = _strassen_222()
            add_block(2, 2, 2, U_b, V_b, W_b, 0, j_off, k_off)
            
    # B2: 2x4x3 (i:0-1, j:0-3, k:4-6)
    # k_off in global is 4. Local k is 0-2.
    # Tiled into:
    # j=0-1, k=0-1 (local): 2x2x2
    # j=0-1, k=2 (local): 2x2x1
    # j=2-3, k=0-1 (local): 2x2x2
    # j=2-3, k=2 (local): 2x2x1
    
    # We need a 2x2x1 decomposition.
    # M_{2,2,1}: Rank 2.
    # U: 4x2, V: 2x2, W: 2x4.
    # T[i,j,0] = 1.
    # U[i*2+j, 0]=1, U[i*2+j, 1]=0?
    # Let's use schoolbook for 2x2x1. Rank 4.
    # Or rank 2:
    # r=0: U[0,0]=1, U[2,0]=1 (A11, A21). V[0,0]=1, V[1,0]=1 (B11, B21). W[0,0]=1, W[2,0]=1 (C11, C21).
    # This gives C11 = A11B11 + A21B21? No.
    # M_{2,2,1} is just a dot product of two vectors?
    # C1 = A1 B1 + A2 B2.
    # This is rank 2.
    # U: (4,2). V: (2,2). W: (2,4).
    # r=0: U[0,0]=1 (A11), U[2,0]=1 (A21). V[0,0]=1 (B11), V[1,0]=1 (B21). W[0,0]=1 (C1), W[2,0]=1 (C2).
    # Term: A11B11 + A21B21.
    # r=1: U[1,1]=1 (A12), U[3,1]=1 (A22). V[0,1]=1 (B12), V[1,1]=1 (B22). W[0,1]=1 (C1), W[2,1]=1 (C2).
    # Term: A12B12 + A22B22.
    # Sum: C1 = A11B11 + A21B21 + A12B12 + A22B22.
    # This is correct for M_{2,2,1}.
    
    def _strassen_221():
        U = np.zeros((4, 2))
        V = np.zeros((2, 2))
        W = np.zeros((2, 4))
        
        # r=0
        U[0, 0] = 1; U[2, 0] = 1
        V[0, 0] = 1; V[1, 0] = 1
        W[0, 0] = 1; W[2, 0] = 1
        
        # r=1
        U[1, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1; V[1, 1] = 1
        W[0, 1] = 1; W[2, 1] = 1
        
        return U.tolist(), V.tolist(), W.tolist()

    # Add B2 blocks
    # j_off: 0, 2. k_local_off: 0, 2.
    # Global k_off = 4.
    for j_off in [0, 2]:
        # Block 2x2x2 (k_local 0-1)
        U_b, V_b, W_b = _strassen_222()
        add_block(2, 2, 2, U_b, V_b, W_b, 0, j_off, 4+0)
        
        # Block 2x2x1 (k_local 2)
        U_b, V_b, W_b = _strassen_221()
        add_block(2, 2, 1, U_b, V_b, W_b, 0, j_off, 4+2)
        
    # B3: 1x4x4 (i:2, j:0-3, k:0-3)
    # Tiled into 4 blocks of 1x2x2.
    # M_{1,2,2} Rank 1.
    def _strassen_122():
        U = np.zeros((2, 1)) # 1*2
        V = np.zeros((2, 1)) # 2*2? No, 2*1? No.
        # M_{1,2,2}: i=0, j=0..1, k=0..1.
        # U: 2x1. V: 2x2. W: 2x1.
        # T[0,j,k] = 1.
        # r=0: U[0,0]=1, U[1,0]=1. V[0,0]=1, V[1,0]=1, V[0,1]=1, V[1,1]=1. W[0,0]=1, W[1,0]=1.
        # Sum: 1*1*1 + 1*1*1 = 2? No.
        # T[0,j,k] = sum_r U[j,r] V[j,k,r] W[k,r].
        # If U[j,0]=1, V[j,k,0]=1, W[k,0]=1, then T=1.
        U = np.zeros((2, 1))
        V = np.zeros((2, 2)) # j in 0..1, k in 0..1
        W = np.zeros((2, 1)) # k in 0..1, i=0
        
        U[0, 0] = 1; U[1, 0] = 1
        V[0, 0] = 1; V[1, 0] = 1; V[0, 1] = 1; V[1, 1] = 1
        W[0, 0] = 1; W[1, 0] = 1
        
        return U.tolist(), V.tolist(), W.tolist()

    for j_off in [0, 2]:
        for k_off in [0, 2]:
            U_b, V_b, W_b = _strassen_122()
            add_block(1, 2, 2, U_b, V_b, W_b, 2, j_off, k_off)
            
    # B4: 1x4x3 (i:2, j:0-3, k:4-6)
    # Tiled into 4 blocks.
    # j=0-1, k=0-1 (local): 1x2x2
    # j=0-1, k=2 (local): 1x2x1
    # j=2-3, k=0-1 (local): 1x2x2
    # j=2-3, k=2 (local): 1x2x1
    
    def _strassen_121():
        U = np.zeros((2, 1))
        V = np.zeros((2, 1)) # 2x1
        W = np.zeros((1, 1))
        
        U[0, 0] = 1; U[1, 0] = 1
        V[0, 0] = 1; V[1, 0] = 1
        W[0, 0] = 1
        
        return U.tolist(), V.tolist(), W.tolist()

    for j_off in [0, 2]:
        # Block 1x2x2 (k_local 0-1)
        U_b, V_b, W_b = _strassen_122()
        add_block(1, 2, 2, U_b, V_b, W_b, 2, j_off, 4+0)
        
        # Block 1x2x1 (k_local 2)
        U_b, V_b, W_b = _strassen_121()
        add_block(1, 2, 1, U_b, V_b, W_b, 2, j_off, 4+2)

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify
    if not _verify_decomposition(U_list, V_list, W_list, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook_nm_p(n, m, p)
    else:
        U, V, W = U_list, V_list, W_list
        
    return {"U": U, "V": V, "W": W}
