"""Matrix multiplication 3x5x7 via block decomposition with Strassen for 2x2x2 blocks."""
import numpy as np

DIMENSIONS = [3, 5, 7]

def _schoolbook(n, m, p):
    """Return (U, V, W) for schoolbook multiplication of n x m by m x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Return (U, V, W) for Strassen's 2x2x2 multiplication."""
    # U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    # Indices: A[i,j] -> i*2+j, B[i,j] -> i*2+j, C[i,j] -> j*2+i (trace convention: W[k*n+i] where n=2, so W[k*2+i])
    # Actually for 2x2x2: n=2, m=2, p=2
    # U index: i*m+j = i*2+j
    # V index: j*p+k = j*2+k
    # W index: k*n+i = k*2+i
    
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # M1 = (A11+A22)(B11+B22)
    # A11 = U[0], A22 = U[3]
    # B11 = V[0], B22 = V[3]
    # C11 = W[0], C12 = W[1], C21 = W[2], C22 = W[3]
    # Wait, W index is k*n+i = k*2+i
    # C[i,k] -> W[k*2+i]
    # C[0,0] -> W[0], C[0,1] -> W[1], C[1,0] -> W[2], C[1,1] -> W[3]
    
    # M1 contributes to C11, C22
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]  # C11 += M1, C22 += M1
    
    # M2 = (A21+A22)B11
    # A21 = U[2], A22 = U[3]
    # B11 = V[0]
    # C21 += M2, C22 -= M2
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 0, 1, -1]  # C21 += M2, C22 -= M2
    
    # M3 = A11(B21-B22)
    # A11 = U[0]
    # B21 = V[2], B22 = V[3]
    # C12 += M3, C22 += M3
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 0, 1, -1]
    W[:, 2] = [0, 1, 0, 1]  # C12 += M3, C22 += M3
    
    # M4 = A22(B12-B11)
    # A22 = U[3]
    # B12 = V[1], B11 = V[0]
    # C11 += M4, C21 += M4
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [-1, 1, 0, 0]
    W[:, 3] = [1, 0, 1, 0]  # C11 += M4, C21 += M4
    
    # M5 = (A11+A12)B22
    # A11 = U[0], A12 = U[1]
    # B22 = V[3]
    # C11 -= M5, C12 += M5
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 1, 0, 0]  # C11 -= M5, C12 += M5
    
    # M6 = (A21-A11)(B11+B12)
    # A21 = U[2], A11 = U[0]
    # B11 = V[0], B12 = V[1]
    # C22 += M6
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1]  # C22 += M6
    
    # M7 = (A12-A22)(B21+B22)
    # A12 = U[1], A22 = U[3]
    # B21 = V[2], B22 = V[3]
    # C11 += M7
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0]  # C11 += M7
    
    return U, V, W

def _block_multiply(n1, n2, m1, m2, p1, p2):
    """
    Multiply (n1+n2) x (m1+m2) by (m1+m2) x (p1+p2).
    Block structure:
    A = [A11 A12]    B = [B11 B12]
        [A21 A22]        [B21 B22]
    where A11: n1xm1, A12: n1xm2, A21: n2xm1, A22: n2xm2
          B11: m1xp1, B12: m1xp2, B21: m2xp1, B22: m2xp2
    
    C = [C11 C12]
        [C21 C22]
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    """
    all_U = []
    all_V = []
    all_W = []
    
    def add_block(n_i, m_j, p_k, i_offset, j_offset, k_offset, U_block, V_block, W_block):
        """Add a block decomposition, offsetting indices appropriately."""
        # U_block has shape (n_i * m_j, rank)
        # V_block has shape (m_j * p_k, rank)
        # W_block has shape (p_k * n_i, rank)
        
        # We need to embed these into the full matrices
        # Full U has shape ((n1+n2) * (m1+m2), rank)
        # Full V has shape ((m1+m2) * (p1+p2), rank)
        # Full W has shape ((p1+p2) * (n1+n2), rank)
        
        rank = U_block.shape[1]
        U_full = np.zeros(((n1 + n2) * (m1 + m2), rank))
        V_full = np.zeros(((m1 + m2) * (p1 + p2), rank))
        W_full = np.zeros(((p1 + p2) * (n1 + n2), rank))
        
        # Map U indices: local (i, j) -> global (i_offset + i, j_offset + j)
        # local index = i * m_j + j
        # global index = (i_offset + i) * (m1 + m2) + (j_offset + j)
        for col in range(rank):
            for i_local in range(n_i):
                for j_local in range(m_j):
                    local_idx = i_local * m_j + j_local
                    global_i = i_offset + i_local
                    global_j = j_offset + j_local
                    global_idx = global_i * (m1 + m2) + global_j
                    U_full[global_idx, col] = U_block[local_idx, col]
        
        # Map V indices: local (j, k) -> global (j_offset + j, k_offset + k)
        # local index = j * p_k + k
        # global index = (j_offset + j) * (p1 + p2) + (k_offset + k)
        for col in range(rank):
            for j_local in range(m_j):
                for k_local in range(p_k):
                    local_idx = j_local * p_k + k_local
                    global_j = j_offset + j_local
                    global_k = k_offset + k_local
                    global_idx = global_j * (p1 + p2) + global_k
                    V_full[global_idx, col] = V_block[local_idx, col]
        
        # Map W indices: local (k, i) -> global (k_offset + k, i_offset + i)
        # local index = k * n_i + i
        # global index = (k_offset + k) * (n1 + n2) + (i_offset + i)
        for col in range(rank):
            for k_local in range(p_k):
                for i_local in range(n_i):
                    local_idx = k_local * n_i + i_local
                    global_k = k_offset + k_local
                    global_i = i_offset + i_local
                    global_idx = global_k * (n1 + n2) + global_i
                    W_full[global_idx, col] = W_block[local_idx, col]
        
        all_U.append(U_full)
        all_V.append(V_full)
        all_W.append(W_full)
    
    # C11 = A11*B11 + A12*B21
    if n1 > 0 and m1 > 0 and p1 > 0:
        U11, V11, W11 = _schoolbook(n1, m1, p1)
        add_block(n1, m1, p1, 0, 0, 0, U11, V11, W11)
    if n1 > 0 and m2 > 0 and p1 > 0:
        U12, V12, W12 = _schoolbook(n1, m2, p1)
        add_block(n1, m2, p1, 0, m1, 0, U12, V12, W12)
    
    # C12 = A11*B12 + A12*B22
    if n1 > 0 and m1 > 0 and p2 > 0:
        U11, V11, W11 = _schoolbook(n1, m1, p2)
        add_block(n1, m1, p2, 0, 0, p1, U11, V11, W11)
    if n1 > 0 and m2 > 0 and p2 > 0:
        U12, V12, W12 = _schoolbook(n1, m2, p2)
        add_block(n1, m2, p2, 0, m1, p1, U12, V12, W12)
    
    # C21 = A21*B11 + A22*B21
    if n2 > 0 and m1 > 0 and p1 > 0:
        U21, V21, W21 = _schoolbook(n2, m1, p1)
        add_block(n2, m1, p1, n1, 0, 0, U21, V21, W21)
    if n2 > 0 and m2 > 0 and p1 > 0:
        U22, V22, W22 = _schoolbook(n2, m2, p1)
        add_block(n2, m2, p1, n1, m1, 0, U22, V22, W22)
    
    # C22 = A21*B12 + A22*B22
    if n2 > 0 and m1 > 0 and p2 > 0:
        U21, V21, W21 = _schoolbook(n2, m1, p2)
        add_block(n2, m1, p2, n1, 0, p1, U21, V21, W21)
    if n2 > 0 and m2 > 0 and p2 > 0:
        U22, V22, W22 = _schoolbook(n2, m2, p2)
        add_block(n2, m2, p2, n1, m1, p1, U22, V22, W22)
    
    if not all_U:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0))
    
    U_total = np.hstack(all_U)
    V_total = np.hstack(all_V)
    W_total = np.hstack(all_W)
    
    return U_total, V_total, W_total

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Split n=3 into [2,1], m=5 into [2,3], p=7 into [4,3]
    n1, n2 = 2, 1
    m1, m2 = 2, 3
    p1, p2 = 4, 3
    
    # For the 2x2x2 blocks, we can use Strassen
    # A11 is 2x2, B11 is 2x4 -> split B11 into [2x2 | 2x2]
    # A12 is 2x3, B12 is 2x3
    # A21 is 1x2, B21 is 3x4
    # A22 is 1x3, B22 is 3x3
    
    # Let's do a more fine-grained approach:
    # Split m=5 into [2,2,1] and p=7 into [2,2,2,1]
    # n=3 stays as is or split into [2,1]
    
    # Actually, let's just use the block decomposition with n=[2,1], m=[2,3], p=[4,3]
    # and within each sub-block, if it's 2x2x2, use Strassen
    
    all_U = []
    all_V = []
    all_W = []
    
    def add_subblock(n_i, m_j, p_k, i_off, j_off, k_off):
        """Add a sub-block decomposition."""
        if n_i == 2 and m_j == 2 and p_k == 2:
            U_b, V_b, W_b = _strassen_2x2x2()
        else:
            U_b, V_b, W_b = _schoolbook(n_i, m_j, p_k)
        
        rank = U_b.shape[1]
        U_full = np.zeros((n * m, rank))
        V_full = np.zeros((m * p, rank))
        W_full = np.zeros((p * n, rank))
        
        for col in range(rank):
            for i_loc in range(n_i):
                for j_loc in range(m_j):
                    li = i_loc * m_j + j_loc
                    gi = (i_off + i_loc) * m + (j_off + j_loc)
                    U_full[gi, col] = U_b[li, col]
            
            for j_loc in range(m_j):
                for k_loc in range(p_k):
                    lj = j_loc * p_k + k_loc
                    gj = (j_off + j_loc) * p + (k_off + k_loc)
                    V_full[gj, col] = V_b[lj, col]
            
            for k_loc in range(p_k):
                for i_loc in range(n_i):
                    lk = k_loc * n_i + i_loc
                    gk = (k_off + k_loc) * n + (i_off + i_loc)
                    W_full[gk, col] = W_b[lk, col]
        
        all_U.append(U_full)
        all_V.append(V_full)
        all_W.append(W_full)
    
    # n split: [2, 1]
    # m split: [2, 2, 1]
    # p split: [2, 2, 2, 1]
    
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2, 1]
    
    i_offsets = []
    off = 0
    for ns in n_splits:
        i_offsets.append(off)
        off += ns
    
    j_offsets = []
    off = 0
    for ms in m_splits:
        j_offsets.append(off)
        off += ms
    
    k_offsets = []
    off = 0
    for ps in p_splits:
        k_offsets.append(off)
        off += ps
    
    for i_idx, n_i in enumerate(n_splits):
        for j_idx, m_j in enumerate(m_splits):
            for k_idx, p_k in enumerate(p_splits):
                add_subblock(n_i, m_j, p_k, i_offsets[i_idx], j_offsets[j_idx], k_offsets[k_idx])
    
    U_total = np.hstack(all_U)
    V_total = np.hstack(all_V)
    W_total = np.hstack(all_W)
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
