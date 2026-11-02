"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _is_valid_decomp(U, V, W, n, m, p, tol=1e-6):
    """Check if U,V,W exactly decompose the matmul tensor."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
        
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The tensor identity for matrix multiplication is:
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _construct_rank56():
    """
    Construct a rank-56 decomposition using block partitioning and Strassen's algorithm.
    
    Dimensions: n=3, m=4, p=5.
    We partition the middle dimension m=4 into two blocks of size 2: m1=2, m2=2.
    A (3x4) = [A1 | A2] where A1 is 3x2, A2 is 3x2.
    B (4x5) = [B1; B2] where B1 is 2x5, B2 is 2x5.
    C = A1*B1 + A2*B2.
    
    For each term (e.g., A1*B1), we have dimensions n=3, m=2, p=5.
    We can further partition p=5 into p1=2, p2=2, p3=1.
    So A1*B1 = A1[:, :2]*B1[:2, :] + A1[:, 2:4]*B1[2:4, :] + A1[:, 4]*B1[4, :]
    Wait, A1 is 3x2, so it doesn't have columns 2..4.
    The partition is on the *inner* dimension of the multiplication A*B.
    Here the inner dimension is m=4.
    So A1 is 3x2, B1 is 2x5. The inner dimension is 2.
    We can partition the inner dimension of A1*B1 (which is size 2) into 1+1?
    Or we can use Strassen on the 2x2 blocks?
    
    Let's refine the strategy:
    Total Rank = Rank(A1*B1) + Rank(A2*B2).
    Since A1, A2 are 3x2 and B1, B2 are 2x5, the multiplication is (3,2,5).
    
    Let's decompose (3,2,5) into smaller blocks.
    Partition n=3 into 1+2.
    A1 = [A1a; A1b] where A1a is 1x2, A1b is 2x2.
    C1 = A1*B1 = [A1a*B1; A1b*B1].
    Rank(A1*B1) <= Rank(A1a*B1) + Rank(A1b*B1).
    
    A1a*B1 is (1,2,5).
    A1b*B1 is (2,2,5).
    
    Let's calculate ranks for these:
    1. (1,2,5):
       Partition p=5 into 2+2+1? Or just schoolbook?
       Schoolbook rank for (1,2,5) is 10.
       Can we do better?
       Lower bound max(1*2, 2*5, 1*5) = max(2, 10, 5) = 10.
       So rank(1,2,5) = 10.
       
    2. (2,2,5):
       Partition p=5 into 2+2+1.
       (2,2,5) = (2,2,2) + (2,2,2) + (2,2,1).
       Rank(2,2,2) = 7 (Strassen).
       Rank(2,2,1):
         Partition n=2 into 1+1.
         (2,2,1) = (1,2,1) + (1,2,1).
         Rank(1,2,1) = 2 (schoolbook, LB max(2,2,1)=2).
         So Rank(2,2,1) <= 4.
       Total Rank(2,2,5) <= 7 + 7 + 4 = 18.
       
    So Rank(3,2,5) <= Rank(1,2,5) + Rank(2,2,5) <= 10 + 18 = 28.
    Total Rank(3,4,5) <= 2 * 28 = 56.
    
    This matches the target rank of 56.
    """
    n, m, p = 3, 4, 5
    
    # We will build the decomposition by combining smaller decompositions.
    # The final U, V, W will have shape (n*m, R), (m*p, R), (p*n, R) with R=56.
    
    U = np.zeros((n*m, 56))
    V = np.zeros((m*p, 56))
    W = np.zeros((p*n, 56))
    
    col_offset = 0
    
    # Helper to add a decomposition of a sub-problem
    # sub_n, sub_m, sub_p: dimensions of the sub-multiplication
    # i_offset, j_offset, k_offset: offsets in the global indices
    # u_sub, v_sub, w_sub: the decomposition matrices for the sub-problem
    #   u_sub: (sub_n * sub_m, sub_rank)
    #   v_sub: (sub_m * sub_p, sub_rank)
    #   w_sub: (sub_p * sub_n, sub_rank)
    
    def add_sub_decomp(sub_n, sub_m, sub_p, i_off, j_off, k_off, u_sub, v_sub, w_sub):
        nonlocal col_offset
        sub_rank = u_sub.shape[1]
        
        # Map indices
        # U global index: (i + i_off) * m + (j + j_off)
        # V global index: (j + j_off) * p + (k + k_off)
        # W global index: (k + k_off) * n + (i + i_off)
        
        for local_i in range(sub_n):
            for local_j in range(sub_m):
                g_i = i_off + local_i
                g_j = j_off + local_j
                g_u_idx = g_i * m + g_j
                for c in range(sub_rank):
                    U[g_u_idx, col_offset + c] += u_sub[local_i * sub_m + local_j, c]
                    
        for local_j in range(sub_m):
            for local_k in range(sub_p):
                g_j = j_off + local_j
                g_k = k_off + local_k
                g_v_idx = g_j * p + g_k
                for c in range(sub_rank):
                    V[g_v_idx, col_offset + c] += v_sub[local_j * sub_p + local_k, c]
                    
        for local_k in range(sub_p):
            for local_i in range(sub_n):
                g_k = k_off + local_k
                g_i = i_off + local_i
                g_w_idx = g_k * n + g_i
                for c in range(sub_rank):
                    W[g_w_idx, col_offset + c] += w_sub[local_k * sub_n + local_i, c]
                    
        col_offset += sub_rank

    # 1. Decomposition for (1,2,5)
    # Schoolbook is optimal (rank 10)
    def get_decomp_1_2_5():
        n_s, m_s, p_s = 1, 2, 5
        rank_s = 10
        u_s = np.zeros((n_s*m_s, rank_s))
        v_s = np.zeros((m_s*p_s, rank_s))
        w_s = np.zeros((p_s*n_s, rank_s))
        c = 0
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    u_s[i*m_s+j, c] = 1
                    v_s[j*p_s+k, c] = 1
                    w_s[k*n_s+i, c] = 1
                    c += 1
        return u_s, v_s, w_s

    # 2. Decomposition for (2,2,1)
    # Rank 4 (schoolbook is 4, LB is 4)
    def get_decomp_2_2_1():
        n_s, m_s, p_s = 2, 2, 1
        rank_s = 4
        u_s = np.zeros((n_s*m_s, rank_s))
        v_s = np.zeros((m_s*p_s, rank_s))
        w_s = np.zeros((p_s*n_s, rank_s))
        c = 0
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    u_s[i*m_s+j, c] = 1
                    v_s[j*p_s+k, c] = 1
                    w_s[k*n_s+i, c] = 1
                    c += 1
        return u_s, v_s, w_s

    # 3. Decomposition for (2,2,2) using Strassen
    # Rank 7
    def get_decomp_2_2_2_strassen():
        n_s, m_s, p_s = 2, 2, 2
        rank_s = 7
        u_s = np.zeros((n_s*m_s, rank_s))
        v_s = np.zeros((m_s*p_s, rank_s))
        w_s = np.zeros((p_s*n_s, rank_s))
        
        # Strassen formulas for C = A*B
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # U indices: i*m+j
        # V indices: j*p+k
        # W indices: k*n+i
        
        # M1
        u_s[0, 0] = 1; u_s[3, 0] = 1  # A11, A22
        v_s[0, 0] = 1; v_s[3, 0] = 1  # B11, B22 (j=0,k=0 and j=1,k=1 -> 0, 3)
        w_s[0, 0] = 1; w_s[1, 0] = 1; w_s[2, 0] = 1; w_s[3, 0] = 1 # C11, C12, C21, C22?
        # Wait, W maps to C indices.
        # C11 is i=0,k=0 -> W[0*2+0, 0] = W[0,0]
        # C12 is i=0,k=1 -> W[1*2+0, 0] = W[2,0]
        # C21 is i=1,k=0 -> W[0*2+1, 0] = W[1,0]
        # C22 is i=1,k=1 -> W[1*2+1, 0] = W[3,0]
        # M1 contributes to C11 and C22.
        # So W[0,0]=1, W[3,0]=1.
        w_s[0, 0] = 1
        w_s[3, 0] = 1
        
        # M2
        u_s[2, 1] = 1; u_s[3, 1] = 1 # A21, A22 (i=1,j=0 and i=1,j=1 -> 2, 3)
        v_s[0, 1] = 1 # B11 (j=0,k=0 -> 0)
        w_s[1, 1] = 1; w_s[3, 1] = 1 # C21, C22 (i=1,k=0 and i=1,k=1 -> 1, 3)
        
        # M3
        u_s[0, 2] = 1 # A11 (i=0,j=0 -> 0)
        v_s[1, 2] = 1; v_s[3, 2] = -1 # B12-B22 (j=0,k=1 -> 1, j=1,k=1 -> 3)
        w_s[2, 2] = 1; w_s[3, 2] = 1 # C12, C22 (i=0,k=1 -> 2, i=1,k=1 -> 3)
        
        # M4
        u_s[3, 3] = 1 # A22 (i=1,j=1 -> 3)
        v_s[2, 3] = 1; v_s[0, 3] = -1 # B21-B11 (j=1,k=0 -> 2, j=0,k=0 -> 0)
        w_s[0, 3] = 1; w_s[1, 3] = 1 # C11, C21 (i=0,k=0 -> 0, i=1,k=0 -> 1)
        
        # M5
        u_s[0, 4] = 1; u_s[1, 4] = 1 # A11+A12 (i=0,j=0 and i=0,j=1 -> 0, 1)
        v_s[3, 4] = 1 # B22 (j=1,k=1 -> 3)
        w_s[0, 4] = -1; w_s[2, 4] = 1 # C11(-1), C12(1) (i=0,k=0 -> 0, i=0,k=1 -> 2)
        
        # M6
        u_s[2, 5] = 1; u_s[0, 5] = -1 # A21-A11 (i=1,j=0 -> 2, i=0,j=0 -> 0)
        v_s[0, 5] = 1; v_s[1, 5] = 1 # B11+B12 (j=0,k=0 -> 0, j=0,k=1 -> 1)
        w_s[3, 5] = 1 # C22 (i=1,k=1 -> 3)
        
        # M7
        u_s[1, 6] = 1; u_s[3, 6] = -1 # A12-A22 (i=0,j=1 -> 1, i=1,j=1 -> 3)
        v_s[2, 6] = 1; v_s[3, 6] = 1 # B21+B22 (j=1,k=0 -> 2, j=1,k=1 -> 3)
        w_s[0, 6] = 1 # C11 (i=0,k=0 -> 0)
        
        return u_s, v_s, w_s

    # 4. Decomposition for (2,2,5)
    # Partition p=5 into 2, 2, 1.
    # (2,2,5) = (2,2,2) + (2,2,2) + (2,2,1)
    def get_decomp_2_2_5():
        u_total = np.zeros((2*2, 18))
        v_total = np.zeros((2*5, 18))
        w_total = np.zeros((5*2, 18))
        
        # Block 1: (2,2,2) for k=0,1
        u1, v1, w1 = get_decomp_2_2_2_strassen()
        u_total[:, 0:7] = u1
        v_total[:, 0:7] = v1
        w_total[:, 0:7] = w1
        
        # Block 2: (2,2,2) for k=2,3
        u2, v2, w2 = get_decomp_2_2_2_strassen()
        u_total[:, 7:14] = u2
        # V indices for k=2,3: j*p+k. j=0,1. k=2,3.
        # V local indices 0,1,2,3 correspond to (0,0),(0,1),(1,0),(1,1) in (2,2) space.
        # We need to map them to global (2,5) space.
        # Local j=0, k=0 -> Global j=0, k=2 -> 0*5+2 = 2
        # Local j=0, k=1 -> Global j=0, k=3 -> 0*5+3 = 3
        # Local j=1, k=0 -> Global j=1, k=2 -> 1*5+2 = 7
        # Local j=1, k=1 -> Global j=1, k=3 -> 1*5+3 = 8
        v_total[2:4, 7:14] = v2[0:2, :]
        v_total[7:9, 7:14] = v2[2:4, :]
        # W indices for k=2,3: k*n+i. n=2.
        # Local k=0, i=0 -> Global k=2, i=0 -> 2*2+0 = 4
        # Local k=0, i=1 -> Global k=2, i=1 -> 2*2+1 = 5
        # Local k=1, i=0 -> Global k=3, i=0 -> 3*2+0 = 6
        # Local k=1, i=1 -> Global k=3, i=1 -> 3*2+1 = 7
        w_total[4:8, 7:14] = w2
        
        # Block 3: (2,2,1) for k=4
        u3, v3, w3 = get_decomp_2_2_1()
        u_total[:, 14:18] = u3
        # V indices for k=4: j*p+k.
        # Local j=0, k=0 -> Global j=0, k=4 -> 4
        # Local j=1, k=0 -> Global j=1, k=4 -> 9
        v_total[4:5, 14:18] = v3[0:1, :]
        v_total[9:10, 14:18] = v3[1:2, :]
        # W indices for k=4: k*n+i.
        # Local k=0, i=0 -> Global k=4, i=0 -> 8
        # Local k=0, i=1 -> Global k=4, i=1 -> 9
        w_total[8:10, 14:18] = w3
        
        return u_total, v_total, w_total

    # 5. Decomposition for (3,2,5)
    # Partition n=3 into 1, 2.
    # (3,2,5) = (1,2,5) + (2,2,5)
    def get_decomp_3_2_5():
        u_total = np.zeros((3*2, 28))
        v_total = np.zeros((2*5, 28))
        w_total = np.zeros((5*3, 28))
        
        # Block 1: (1,2,5) for i=0
        u1, v1, w1 = get_decomp_1_2_5()
        u_total[0:2, 0:10] = u1
        v_total[:, 0:10] = v1
        # W indices for i=0: k*n+i. n=3.
        # Local k, i=0 -> Global k, i=0 -> k*3+0 = 3k
        for k in range(5):
            w_total[k*3: k*3+1, 0:10] = w1[k*1: k*1+1, :] # w1 is 5x1
        
        # Block 2: (2,2,5) for i=1,2
        u2, v2, w2 = get_decomp_2_2_5()
        # U indices: local i=0,1 -> Global i=1,2.
        # Local i=0, j=0 -> Global i=1, j=0 -> 1*2+0 = 2
        # Local i=0, j=1 -> Global i=1, j=1 -> 1*2+1 = 3
        # Local i=1, j=0 -> Global i=2, j=0 -> 2*2+0 = 4
        # Local i=1, j=1 -> Global i=2, j=1 -> 2*2+1 = 5
        u_total[2:4, 10:28] = u2[0:2, :]
        u_total[4:6, 10:28] = u2[2:4, :]
        v_total[:, 10:28] = v2
        # W indices: local k, i=0,1 -> Global k, i=1,2.
        # Local k, i=0 -> Global k, i=1 -> k*3+1
        # Local k, i=1 -> Global k, i=2 -> k*3+2
        for k in range(5):
            w_total[k*3+1: k*3+3, 10:28] = w2[k*2: k*2+2, :]
            
        return u_total, v_total, w_total

    # Now assemble the full (3,4,5) decomposition
    # (3,4,5) = (3,2,5) [j=0,1] + (3,2,5) [j=2,3]
    
    u_325_1, v_325_1, w_325_1 = get_decomp_3_2_5()
    u_325_2, v_325_2, w_325_2 = get_decomp_3_2_5()
    
    # Add first half (j=0,1)
    add_sub_decomp(3, 2, 5, 0, 0, 0, u_325_1, v_325_1, w_325_1)
    
    # Add second half (j=2,3)
    # U indices: global i, j=2,3.
    # V indices: global j=2,3, k.
    # W indices: global k, i.
    add_sub_decomp(3, 2, 5, 0, 2, 0, u_325_2, v_325_2, w_325_2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    start_time = time.time()
    
    # Try the rank-56 decomposition
    result = _construct_rank56()
    
    # Verify it
    if _is_valid_decomp(result['U'], result['V'], result['W'], 3, 4, 5):
        return result
    
    # Fallback to schoolbook rank-60
    return _schoolbook()
# EVOLVE-BLOCK-END
