"""Matrix multiplication 2,2,10 - Strassen block decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 10]

# Strassen's 7-term decomposition for 2x2 matrix multiplication
# For A (2x2) * B (2x2) = C (2x2):
# M1 = (A00 + A11) * (B00 + B11)
# M2 = (A10 + A11) * B00
# M3 = A00 * (B01 - B11)
# M4 = A11 * (B10 - B00)
# M5 = (A00 + A01) * B11
# M6 = (A10 - A00) * (B00 + B01)
# M7 = (A01 - A11) * (B10 + B11)
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6

def _strassen_2x2():
    """
    Return U, V, W for 2x2*2x2 matrix multiplication with rank 7.
    
    U has shape (n*m, rank) = (4, 7) where index i*m+j
    V has shape (m*p, rank) = (4, 7) where index j*p+k
    W has shape (p*n, rank) = (4, 7) where index k*n+i (trace convention)
    
    A is indexed by (i,j) where i=0..1 (row of A), j=0..1 (col of A = row of B)
    B is indexed by (j,k) where j=0..1 (row of B), k=0..1 (col of B)
    C is indexed by (i,k) where i=0..1 (row of C), k=0..1 (col of C)
    
    W index: k*n + i where n=2
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # A indices: i*m + j
    # B indices: j*p + k
    # C indices: k*n + i
    
    # M1 = (A00 + A11) * (B00 + B11)
    # A00 = (0,0) -> U idx 0; A11 = (1,1) -> U idx 3
    # B00 = (0,0) -> V idx 0; B11 = (1,1) -> V idx 3
    # C00 = (0,0) -> W idx 0*2+0 = 0; C11 = (1,1) -> W idx 1*2+1 = 3
    col = 0
    U[0, col] = 1  # A00
    U[3, col] = 1  # A11
    V[0, col] = 1  # B00
    V[3, col] = 1  # B11
    W[0, col] = 1  # C00
    W[3, col] = 1  # C11
    col += 1
    
    # M2 = (A10 + A11) * B00
    # A10 = (1,0) -> U idx 2; A11 = (1,1) -> U idx 3
    # B00 = (0,0) -> V idx 0
    # C10 = (1,0) -> W idx 0*2+1 = 1
    col = 1
    U[2, col] = 1  # A10
    U[3, col] = 1  # A11
    V[0, col] = 1  # B00
    W[1, col] = 1  # C10
    col += 1
    
    # M3 = A00 * (B01 - B11)
    # A00 = (0,0) -> U idx 0
    # B01 = (0,1) -> V idx 1; B11 = (1,1) -> V idx 3
    # C01 = (0,1) -> W idx 1*2+0 = 2
    col = 2
    U[0, col] = 1  # A00
    V[1, col] = 1  # B01
    V[3, col] = -1 # -B11
    W[2, col] = 1  # C01
    col += 1
    
    # M4 = A11 * (B10 - B00)
    # A11 = (1,1) -> U idx 3
    # B10 = (1,0) -> V idx 2; B00 = (0,0) -> V idx 0
    # C00 = (0,0) -> W idx 0; C10 = (1,0) -> W idx 1
    col = 3
    U[3, col] = 1  # A11
    V[2, col] = 1  # B10
    V[0, col] = -1 # -B00
    W[0, col] = 1  # C00
    W[1, col] = 1  # C10
    col += 1
    
    # M5 = (A00 + A01) * B11
    # A00 = (0,0) -> U idx 0; A01 = (0,1) -> U idx 1
    # B11 = (1,1) -> V idx 3
    # C00 = (0,0) -> W idx 0; C01 = (0,1) -> W idx 2
    col = 4
    U[0, col] = 1  # A00
    U[1, col] = 1  # A01
    V[3, col] = 1  # B11
    W[0, col] = -1 # -C00
    W[2, col] = 1  # C01
    col += 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    # A10 = (1,0) -> U idx 2; A00 = (0,0) -> U idx 0
    # B00 = (0,0) -> V idx 0; B01 = (0,1) -> V idx 1
    # C11 = (1,1) -> W idx 3
    col = 5
    U[2, col] = 1  # A10
    U[0, col] = -1 # -A00
    V[0, col] = 1  # B00
    V[1, col] = 1  # B01
    W[3, col] = 1  # C11
    col += 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    # A01 = (0,1) -> U idx 1; A11 = (1,1) -> U idx 3
    # B10 = (1,0) -> V idx 2; B11 = (1,1) -> V idx 3
    # C00 = (0,0) -> W idx 0
    col = 6
    U[1, col] = 1  # A01
    U[3, col] = -1 # -A11
    V[2, col] = 1  # B10
    V[3, col] = 1  # B11
    W[0, col] = 1  # C00
    
    return U, V, W


def construct():
    """
    Construct rank-35 decomposition for 2x2 * 2x10 matrix multiplication.
    
    The 2x10 B matrix can be viewed as 5 independent 2x2 blocks:
    B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8], B[:, 8:10]
    
    Each block uses Strassen's rank-7 decomposition, giving total rank 5*7 = 35.
    
    U: shape (n*m, rank) = (4, 35)
    V: shape (m*p, rank) = (20, 35)
    W: shape (p*n, rank) = (20, 35)
    """
    n, m, p = DIMENSIONS  # 2, 2, 10
    num_blocks = p // 2  # 5 blocks of 2x2
    block_rank = 7
    total_rank = num_blocks * block_rank  # 35
    
    # Get the base Strassen decomposition for 2x2
    U_base, V_base, W_base = _strassen_2x2()
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    for b in range(num_blocks):
        # Block b handles B columns [2*b, 2*b+2)
        col_offset = b * block_rank
        
        # U is the same for all blocks (A is the same 2x2 matrix)
        U[:, col_offset:col_offset + block_rank] = U_base
        
        # V: for block b, B indices j*p+k where k is in [2*b, 2*b+2)
        # Within the block, local k' = k - 2*b, so local V index = j*2 + k'
        # Global V index = j*p + k = j*10 + 2*b + k'
        for j in range(m):
            for k_prime in range(2):
                local_v_idx = j * 2 + k_prime  # j*p_local + k' where p_local=2
                global_v_idx = j * p + 2 * b + k_prime
                V[global_v_idx, col_offset:col_offset + block_rank] = V_base[local_v_idx, :]
        
        # W: for block b, C indices k*n+i where k is in [2*b, 2*b+2)
        # Within the block, local k' = k - 2*b, so local W index = k'*n + i
        # Global W index = k*n + i = (2*b + k')*n + i
        for k_prime in range(2):
            for i in range(n):
                local_w_idx = k_prime * n + i  # k'*n + i where n=2
                global_w_idx = (2 * b + k_prime) * n + i
                W[global_w_idx, col_offset:col_offset + block_rank] = W_base[local_w_idx, :]
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
