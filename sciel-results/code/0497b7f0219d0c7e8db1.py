"""Program for matrix multiplication 3x3x4 tensor decomposition."""
DIMENSIONS = [3, 3, 4]

def construct():
    """
    Construct a rank-minimal (or reduced rank) tensor decomposition for
    multiplying a 3x3 matrix by a 3x4 matrix.
    
    We use a block-based approach with Strassen for 2x2x2 blocks.
    Rank 26.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    rank = 26
    U = [[0] * rank for _ in range(n * m)]  # 9 x 26
    V = [[0] * rank for _ in range(m * p)]  # 12 x 26
    W = [[0] * rank for _ in range(p * n)]  # 12 x 26
    
    def add_strassen(offset_u, offset_v, offset_w, col_start):
        """
        Add Strassen decomposition for 2x2x2 block.
        offset_u: index offset for U (i*m + j)
        offset_v: index offset for V (j*p + k)
        offset_w: index offset for W (k*n + i)
        """
        # Local indices 0,1 map to global offsets
        # U indices: (0,0)->offset_u+0, (0,1)->offset_u+1, (1,0)->offset_u+2, (1,1)->offset_u+3
        # V indices: (0,0)->offset_v+0, (0,1)->offset_v+1, (1,0)->offset_v+2, (1,1)->offset_v+3
        # W indices: (0,0)->offset_w+0, (0,1)->offset_w+1, (1,0)->offset_w+2, (1,1)->offset_w+3
        
        # M1
        U[offset_u + 0][col_start] += 1
        U[offset_u + 3][col_start] += 1
        V[offset_v + 0][col_start] += 1
        V[offset_v + 3][col_start] += 1
        W[offset_w + 0][col_start] += 1
        W[offset_w + 3][col_start] += 1
        
        # M2
        U[offset_u + 2][col_start + 1] += 1
        U[offset_u + 3][col_start + 1] += 1
        V[offset_v + 0][col_start + 1] += 1
        W[offset_w + 2][col_start + 1] += 1
        W[offset_w + 3][col_start + 1] += -1
        
        # M3
        U[offset_u + 0][col_start + 2] += 1
        V[offset_v + 2][col_start + 2] += 1
        V[offset_v + 3][col_start + 2] += -1
        W[offset_w + 1][col_start + 2] += 1
        W[offset_w + 3][col_start + 2] += 1
        
        # M4
        U[offset_u + 3][col_start + 3] += 1
        V[offset_v + 1][col_start + 3] += 1
        V[offset_v + 0][col_start + 3] += -1
        W[offset_w + 0][col_start + 3] += 1
        W[offset_w + 2][col_start + 3] += 1
        
        # M5
        U[offset_u + 0][col_start + 4] += 1
        U[offset_u + 1][col_start + 4] += 1
        V[offset_v + 3][col_start + 4] += 1
        W[offset_w + 0][col_start + 4] += -1
        W[offset_w + 1][col_start + 4] += 1
        
        # M6
        U[offset_u + 2][col_start + 5] += 1
        U[offset_u + 0][col_start + 5] += -1
        V[offset_v + 0][col_start + 5] += 1
        V[offset_v + 1][col_start + 5] += 1
        W[offset_w + 3][col_start + 5] += 1
        
        # M7
        U[offset_u + 1][col_start + 6] += 1
        U[offset_u + 3][col_start + 6] += -1
        V[offset_v + 2][col_start + 6] += 1
        V[offset_v + 3][col_start + 6] += 1
        W[offset_w + 0][col_start + 6] += 1

    def add_simple_2x1x2(i_start, j, k_start, col_start):
        """
        Add decomposition for A[i_start:i_start+2, j] * B[j, k_start:k_start+2]
        Result C[i_start:i_start+2, k_start:k_start+2]
        Rank 2.
        """
        # Term 1: i=i_start, k=k_start
        u_idx = i_start * m + j
        v_idx = j * p + k_start
        w_idx = k_start * n + i_start
        U[u_idx][col_start] += 1
        V[v_idx][col_start] += 1
        W[w_idx][col_start] += 1
        
        # Term 2: i=i_start+1, k=k_start+1? No.
        # The product is sum over j. Here j is fixed.
        # C[i,k] = A[i,j] B[j,k].
        # We need to cover all (i,k) in the 2x2 block.
        # The tensor is sum_{i,k} E_{i,j} E_{j,k} E_{k,i}.
        # This is not rank 1. It is rank 2 as derived before.
        # My previous derivation:
        # u1 has 1 at (i_start, j). v1 has 1 at (j, k_start) and (j, k_start+1).
        # w1 has 1 at (k_start, i_start) and (k_start+1, i_start).
        # This covers (i_start, k_start) and (i_start, k_start+1).
        # u2 has 1 at (i_start+1, j). v2 has 1 at (j, k_start) and (j, k_start+1).
        # w2 has 1 at (k_start, i_start+1) and (k_start+1, i_start+1).
        # This covers (i_start+1, k_start) and (i_start+1, k_start+1).
        
        # Let's implement this.
        # Term 1 (for i_start)
        U[u_idx][col_start] += 1 # Already added above? No, I added specific indices.
        # Let's restart this function logic.
        pass

    # Re-implementing add_simple_2x1x2 correctly
    def add_simple_A12_B21(i_start, j, k_start, col_start):
        """
        A[i_start:i_start+2, j] * B[j, k_start:k_start+2] -> C[i_start:i_start+2, k_start:k_start+2]
        """
        # Term 1: i = i_start
        u_idx1 = i_start * m + j
        v_idx1a = j * p + k_start
        v_idx1b = j * p + (k_start + 1)
        w_idx1a = k_start * n + i_start
        w_idx1b = (k_start + 1) * n + i_start
        
        U[u_idx1][col_start] += 1
        V[v_idx1a][col_start] += 1
        V[v_idx1b][col_start] += 1
        W[w_idx1a][col_start] += 1
        W[w_idx1b][col_start] += 1
        
        # Term 2: i = i_start + 1
        u_idx2 = (i_start + 1) * m + j
        v_idx2a = j * p + k_start
        v_idx2b = j * p + (k_start + 1)
        w_idx2a = k_start * n + (i_start + 1)
        w_idx2b = (k_start + 1) * n + (i_start + 1)
        
        U[u_idx2][col_start + 1] += 1
        V[v_idx2a][col_start + 1] += 1
        V[v_idx2b][col_start + 1] += 1
        W[w_idx2a][col_start + 1] += 1
        W[w_idx2b][col_start + 1] += 1

    def add_simple_A21_B11(i, j_start, k_start, col_start):
        """
        A[i, j_start:j_start+2] * B[j_start:j_start+2, k_start:k_start+2] -> C[i, k_start:k_start+2]
        """
        # Term 1: j = j_start
        u_idx1 = i * m + j_start
        v_idx1a = j_start * p + k_start
        v_idx1b = j_start * p + (k_start + 1)
        w_idx1a = k_start * n + i
        w_idx1b = (k_start + 1) * n + i
        
        U[u_idx1][col_start] += 1
        V[v_idx1a][col_start] += 1
        V[v_idx1b][col_start] += 1
        W[w_idx1a][col_start] += 1
        W[w_idx1b][col_start] += 1
        
        # Term 2: j = j_start + 1
        u_idx2 = i * m + (j_start + 1)
        v_idx2a = (j_start + 1) * p + k_start
        v_idx2b = (j_start + 1) * p + (k_start + 1)
        w_idx2a = k_start * n + i
        w_idx2b = (k_start + 1) * n + i
        
        U[u_idx2][col_start + 1] += 1
        V[v_idx2a][col_start + 1] += 1
        V[v_idx2b][col_start + 1] += 1
        W[w_idx2a][col_start + 1] += 1
        W[w_idx2b][col_start + 1] += 1

    def add_simple_A22_B21(i, j, k_start, col_start):
        """
        A[i, j] * B[j, k_start:k_start+2] -> C[i, k_start:k_start+2]
        """
        # Term 1: k = k_start
        u_idx = i * m + j
        v_idx1 = j * p + k_start
        w_idx1 = k_start * n + i
        
        U[u_idx][col_start] += 1
        V[v_idx1][col_start] += 1
        W[w_idx1][col_start] += 1
        
        # Term 2: k = k_start + 1
        v_idx2 = j * p + (k_start + 1)
        w_idx2 = (k_start + 1) * n + i
        
        U[u_idx][col_start + 1] += 1
        V[v_idx2][col_start + 1] += 1
        W[w_idx2][col_start + 1] += 1

    # Block a: k in {0, 1}
    # A11 B11a: i in {0,1}, j in {0,1}, k in {0,1}
    # U offset: 0*3+0=0. V offset: 0*4+0=0. W offset: 0*3+0=0.
    add_strassen(0, 0, 0, 0)
    
    # A12 B21a: i in {0,1}, j=2, k in {0,1}
    # U offset: 0*3+2=2. V offset: 2*4+0=8. W offset: 0*3+0=0.
    add_simple_A12_B21(0, 2, 0, 7)
    
    # A21 B11a: i=2, j in {0,1}, k in {0,1}
    # U offset: 2*3+0=6. V offset: 0*4+0=0. W offset: 0*3+2=2.
    add_simple_A21_B11(2, 0, 0, 9)
    
    # A22 B21a: i=2, j=2, k in {0,1}
    # U offset: 2*3+2=8. V offset: 2*4+0=8. W offset: 0*3+2=2.
    add_simple_A22_B21(2, 2, 0, 11)
    
    # Block b: k in {2, 3}
    # A11 B11b: i in {0,1}, j in {0,1}, k in {2,3}
    # U offset: 0. V offset: 0*4+2=2. W offset: 2*3+0=6.
    add_strassen(0, 2, 6, 13)
    
    # A12 B21b: i in {0,1}, j=2, k in {2,3}
    # U offset: 2. V offset: 2*4+2=10. W offset: 2*3+0=6.
    add_simple_A12_B21(0, 2, 2, 20)
    
    # A21 B11b: i=2, j in {0,1}, k in {2,3}
    # U offset: 6. V offset: 0*4+2=2. W offset: 2*3+2=8.
    add_simple_A21_B11(2, 0, 2, 22)
    
    # A22 B21b: i=2, j=2, k in {2,3}
    # U offset: 8. V offset: 2*4+2=10. W offset: 2*3+2=8.
    add_simple_A22_B21(2, 2, 2, 24)

    return {"U": U, "V": V, "W": W}
