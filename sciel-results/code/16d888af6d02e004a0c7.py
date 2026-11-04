import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _validate_decomposition(U, V, W, n, m, p, tol=1e-7):
    """
    Check that the factor matrices define the matrix multiplication tensor.
    Constraint: for all i in [0,n), a in [0,m), b in [0,m), k in [0,p):
        sum_c W[k*n+i, c] * U[i*m+a, c] * V[b*p+k, c] = (1 if a==b else 0)
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    # Reshape for easier indexing
    U_reshaped = U_arr.reshape(n, m, rank)  # U[i, a, c]
    V_reshaped = V_arr.reshape(m, p, rank)  # V[b, k, c]
    W_reshaped = W_arr.reshape(p, n, rank)  # W[k, i, c]
    
    # Efficient vectorized check
    # For each (i, a, b, k), we need sum_c W[k,i,c] * U[i,a,c] * V[b,k,c]
    # We can compute this as a batched operation
    
    # Create arrays for all combinations
    # U_reshaped: (n, m, rank)
    # V_reshaped: (m, p, rank)
    # W_reshaped: (p, n, rank)
    
    # For each i, a: U[i, a, :] has shape (rank,)
    # For each b, k: V[b, k, :] has shape (rank,)
    # For each k, i: W[k, i, :] has shape (rank,)
    
    # We need: sum_c W[k,i,c] * U[i,a,c] * V[b,k,c]
    
    # Let's do it with broadcasting
    # U[i,a,:] * V[b,k,:] * W[k,i,:]
    
    # U_reshaped: (n, m, rank) -> expand to (n, m, 1, p, rank)
    # V_reshaped: (m, p, rank) -> expand to (1, m, p, 1, rank)
    # W_reshaped: (p, n, rank) -> expand to (n, 1, 1, p, rank)
    
    # Product: (n, m, p, rank) after summing over c
    # Wait, let me think again.
    
    # U[i, a, c] * V[b, k, c] * W[k, i, c]
    # Sum over c.
    
    # Let's use einsum
    # U: (n, m, c)
    # V: (m, p, c)
    # W: (p, n, c)
    
    # Result[i, a, b, k] = sum_c U[i,a,c] * V[b,k,c] * W[k,i,c]
    
    # This is a bit tricky with einsum because the indices are mixed.
    # Let's try:
    # temp1[i, a, b, k, c] = U[i,a,c] * V[b,k,c]
    # result[i, a, b, k] = sum_c temp1[i,a,b,k,c] * W[k,i,c]
    
    # Using einsum:
    # U: iac
    # V: bkc
    # W: kic
    
    # We want: iabc? No, i, a, b, k.
    # sum_c U[i,a,c] * V[b,k,c] * W[k,i,c]
    
    # einsum('iac,bkc,kic->iabk', U, V, W)
    
    try:
        result = np.einsum('iac,bkc,kic->iabk', U_reshaped, V_reshaped, W_reshaped)
    except:
        # Fallback to loops
        for i in range(n):
            for a in range(m):
                for b in range(m):
                    for k in range(p):
                        s = np.dot(W_reshaped[k, i, :], U_reshaped[i, a, :] * V_reshaped[b, k, :])
                        expected = 1.0 if a == b else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True
    
    # Check against expected
    # Expected: 1 if a==b else 0
    # result shape: (n, m, m, p)
    # We need result[i, a, b, k] == (1 if a==b else 0) for all i, k
    
    # Create expected array
    expected = np.zeros((n, m, m, p))
    for a in range(m):
        for b in range(m):
            if a == b:
                expected[:, a, b, :] = 1.0
    
    if np.max(np.abs(result - expected)) > tol:
        return False
    
    return True

def _strassen_2x2x2():
    """
    Returns the Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where:
    U is indexed by (i, a) -> i*2+a for i,a in {0,1}
    V is indexed by (b, k) -> b*2+k for b,k in {0,1}
    W is indexed by (k, i) -> k*2+i for k,i in {0,1}
    
    Strassen's 7 products:
    M0 = (A00+A11)(B00+B11)
    M1 = (A10+A11)B00
    M2 = A00(B01-B11)
    M3 = A11(B10-B00)
    M4 = (A00+A01)B11
    M5 = (A10-A00)(B00+B01)
    M6 = (A01-A11)(B10+B11)
    
    C00 = M0 - M3 + M5
    C01 = M2 + M4
    C10 = M1 + M3
    C11 = M0 - M2 + M6
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M0: (A00+A11)(B00+B11) -> contributes to C00 (+1), C11 (+1)
    # A00 is U[0*2+0]=U[0], A11 is U[1*2+1]=U[3]
    # B00 is V[0*2+0]=V[0], B11 is V[1*2+1]=V[3]
    # C00 is W[0*2+0]=W[0], C11 is W[1*2+1]=W[3]
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11
    
    # M1: (A10+A11)B00 -> contributes to C10 (+1)
    # A10 is U[1*2+0]=U[2], A11 is U[3], B00 is V[0]
    # C10 is W[0*2+1]=W[1]
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    W[1][1] = 1  # C10
    
    # M2: A00(B01-B11) -> contributes to C01 (+1), C11 (-1)
    # A00 is U[0], B01 is V[0*2+1]=V[1], B11 is V[3]
    # C01 is W[1*2+0]=W[2], C11 is W[3]
    U[0][2] = 1  # A00
    V[1][2] = 1  # B01
    V[3][2] = -1 # B11
    W[2][2] = 1  # C01
    W[3][2] = -1 # C11
    
    # M3: A11(B10-B00) -> contributes to C10 (+1), C00 (-1)
    # A11 is U[3], B10 is V[1*2+0]=V[2], B00 is V[0]
    # C10 is W[1], C00 is W[0]
    U[3][3] = 1  # A11
    V[2][3] = 1  # B10
    V[0][3] = -1 # B00
    W[1][3] = 1  # C10
    W[0][3] = -1 # C00
    
    # M4: (A00+A01)B11 -> contributes to C01 (+1)
    # A00 is U[0], A01 is U[0*2+1]=U[1], B11 is V[3]
    # C01 is W[2]
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    W[2][4] = 1  # C01
    
    # M5: (A10-A00)(B00+B01) -> contributes to C00 (+1)
    # A10 is U[2], A00 is U[0], B00 is V[0], B01 is V[1]
    # C00 is W[0]
    U[2][5] = 1  # A10
    U[0][5] = -1 # A00
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    W[0][5] = 1  # C00
    
    # M6: (A01-A11)(B10+B11) -> contributes to C11 (+1)
    # A01 is U[1], A11 is U[3], B10 is V[2], B11 is V[3]
    # C11 is W[3]
    U[1][6] = 1  # A01
    U[3][6] = -1 # A11
    V[2][6] = 1  # B10
    V[3][6] = 1  # B11
    W[3][6] = 1  # C11
    
    return U, V, W

def _schoolbook_block(n_block, m_block, p_block):
    """
    Returns the schoolbook decomposition for an n_block x m_block x p_block block.
    Returns U (n_block*m_block x rank), V (m_block*p_block x rank), W (p_block*n_block x rank)
    where rank = n_block * m_block * p_block.
    """
    rank = n_block * m_block * p_block
    U = [[0]*rank for _ in range(n_block * m_block)]
    V = [[0]*rank for _ in range(m_block * p_block)]
    W = [[0]*rank for _ in range(p_block * n_block)]
    
    c = 0
    for i in range(n_block):
        for j in range(m_block):
            for k in range(p_block):
                U[i * m_block + j][c] = 1
                V[j * p_block + k][c] = 1
                W[k * n_block + i][c] = 1
                c += 1
    
    return U, V, W

def _try_block_decomposition(n, m, p):
    """
    Constructs a decomposition using block Strassen for 2x2x2 sub-blocks.
    
    Strategy: Partition the dimensions into blocks, and for each 2x2x2 block of
    (I_block, J_block, K_block) use Strassen (rank 7) instead of schoolbook (rank 8).
    For remaining 1-wide slices, use schoolbook (rank 1 each).
    
    We partition:
    - I dimension (size n=3): [2, 1]  -> I0 has size 2, I1 has size 1
    - J dimension (size m=5): [2, 2, 1] -> J0 size 2, J1 size 2, J2 size 1
    - K dimension (size p=7): [2, 2, 2, 1] -> K0 size 2, K1 size 2, K2 size 2, K3 size 1
    
    For each combination of blocks (I_x, J_y, K_z):
    - If all have size 2: use Strassen, contributes rank 7
    - Otherwise: use schoolbook, contributes rank (I_x_size * J_y_size * K_z_size)
    """
    
    # Define block partitions
    # I blocks: list of (start, size)
    I_blocks = [(0, 2), (2, 1)]  # I0: indices 0,1 (size 2); I1: index 2 (size 1)
    # J blocks:
    J_blocks = [(0, 2), (2, 2), (4, 1)]  # J0: 0,1; J1: 2,3; J2: 4
    # K blocks:
    K_blocks = [(0, 2), (2, 2), (4, 2), (6, 1)]  # K0: 0,1; K1: 2,3; K2: 4,5; K3: 6
    
    # Get Strassen factors for 2x2x2
    S_U, S_V, S_W = _strassen_2x2x2()
    
    # Initialize full matrices with zeros
    total_rank = 0
    # First, compute total rank
    for i_start, i_size in I_blocks:
        for j_start, j_size in J_blocks:
            for k_start, k_size in K_blocks:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    total_rank += 7
                else:
                    total_rank += i_size * j_size * k_size
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    for i_start, i_size in I_blocks:
        for j_start, j_size in J_blocks:
            for k_start, k_size in K_blocks:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Use Strassen
                    block_U, block_V, block_W = S_U, S_V, S_W
                    block_rank = 7
                else:
                    # Use schoolbook
                    block_U, block_V, block_W = _schoolbook_block(i_size, j_size, k_size)
                    block_rank = i_size * j_size * k_size
                
                # Fill in the global matrices
                for local_i in range(i_size):
                    for local_a in range(j_size):
                        global_i = i_start + local_i
                        global_a = j_start + local_a
                        for c in range(block_rank):
                            U[global_i * m + global_a][col_offset + c] = block_U[local_i * j_size + local_a][c]
                
                for local_b in range(j_size):
                    for local_k in range(k_size):
                        global_b = j_start + local_b
                        global_k = k_start + local_k
                        for c in range(block_rank):
                            V[global_b * p + global_k][col_offset + c] = block_V[local_b * k_size + local_k][c]
                
                for local_k in range(k_size):
                    for local_i in range(i_size):
                        global_k = k_start + local_k
                        global_i = i_start + local_i
                        for c in range(block_rank):
                            W[global_k * n + global_i][col_offset + c] = block_W[local_k * i_size + local_i][c]
                
                col_offset += block_rank
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_decomposition(n, m, p):
    """
    Constructs the schoolbook decomposition for matrix multiplication.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Constructs the tensor decomposition for matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W'.
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # First, try the block Strassen decomposition
    try:
        result = _try_block_decomposition(n, m, p)
        
        # Validate the result
        if _validate_decomposition(result['U'], result['V'], result['W'], n, m, p):
            return result
        else:
            # If validation fails, fall back to schoolbook
            return _schoolbook_decomposition(n, m, p)
    except Exception as e:
        # If anything goes wrong, fall back to schoolbook
        return _schoolbook_decomposition(n, m, p)
