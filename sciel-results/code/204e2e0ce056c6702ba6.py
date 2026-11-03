"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, tol=1e-8):
    n, m, p = DIMENSIONS
    # Random test matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    # Compute C from factors
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    R = U.shape[1]
    C_comp = np.zeros((n, p))
    
    # Efficient computation:
    # C_ik = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    # Reshape U to (n, m, R)
    U_reshaped = U.reshape(n, m, R)
    S_A = np.einsum('ijr,ij->ir', U_reshaped, A)
    
    # Reshape V to (m, p, R)
    V_reshaped = V.reshape(m, p, R)
    S_B = np.einsum('jkr,jk->kr', V_reshaped, B)
    
    # Reshape W to (p, n, R)
    W_reshaped = W.reshape(p, n, R)
    
    C_comp = np.einsum('kir,ir,kr->ik', W_reshaped, S_A, S_B)
    
    return np.allclose(C_true, C_comp, atol=tol)

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen.
    U: (4, 7) for A (2x2)
    V: (4, 7) for B (2x2)
    W: (4, 7) for C (2x2)
    Indices:
    U row: i*2+j
    V row: j*2+k
    W row: k*2+i
    """
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1]    # A11
    ], dtype=float)
    
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 0, 1, 0, 1, 0],    # B01
        [0, 0, 1, 0, 1, 0, 1],    # B10
        [1, 0, -1, 0, 1, 0, 1]    # B11
    ], dtype=float)
    
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0]    # C11
    ], dtype=float)
    
    return U, V, W

def _schoolbook_2x2x1():
    """
    Returns U, V, W for 2x2x1 schoolbook.
    U: (4, 4) for A (2x2)
    V: (4, 4) for B (2x2) -- Note: only B[j,0] is used effectively, but shape must match 2x2 block of B
    W: (4, 4) for C (2x2) -- Note: only C[i,0] is used effectively
    """
    # C_i0 = A_i0 B_00 + A_i1 B_10
    # Terms:
    # 0: A00 B00 -> C00
    # 1: A01 B10 -> C00
    # 2: A10 B00 -> C10
    # 3: A11 B10 -> C10
    
    U = np.array([
        [1, 0, 0, 0], # A00
        [0, 1, 0, 0], # A01
        [0, 0, 1, 0], # A10
        [0, 0, 0, 1]  # A11
    ], dtype=float)
    
    V = np.array([
        [1, 0, 1, 0], # B00 (used in term 0 and 2)
        [0, 0, 0, 0], # B01 (not used)
        [0, 1, 0, 1], # B10 (used in term 1 and 3)
        [0, 0, 0, 0]  # B11 (not used)
    ], dtype=float)
    
    W = np.array([
        [1, 1, 0, 0], # C00
        [0, 0, 0, 0], # C01
        [0, 0, 1, 1], # C10
        [0, 0, 0, 0]  # C11
    ], dtype=float)
    
    return U, V, W

def _schoolbook_1x4x7():
    """
    Returns U, V, W for 1x4x7 schoolbook.
    n=1, m=4, p=7.
    U: (4, 28)
    V: (28, 28)
    W: (7, 28)
    """
    n, m, p = 1, 4, 7
    rank = m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=7
    
    # We will build the full U, V, W matrices.
    # Total rank is sum of ranks of blocks.
    
    # Blocks:
    # Part 1: i in {0,1} (n=2)
    #   Part 1a: j in {0,1}, k in {0,1} (2x2x2 Strassen)
    #   Part 1b: j in {0,1}, k in {2,3} (2x2x2 Strassen)
    #   Part 1c: j in {0,1}, k in {4,5} (2x2x2 Strassen)
    #   Part 1d: j in {0,1}, k in {6}   (2x2x1 Schoolbook)
    #   Part 1e: j in {2,3}, k in {0,1} (2x2x2 Strassen)
    #   Part 1f: j in {2,3}, k in {2,3} (2x2x2 Strassen)
    #   Part 1g: j in {2,3}, k in {4,5} (2x2x2 Strassen)
    #   Part 1h: j in {2,3}, k in {6}   (2x2x1 Schoolbook)
    # Part 2: i in {2} (n=1)
    #   Part 2a: j in {0,1,2,3}, k in {0,1,2,3,4,5,6} (1x4x7 Schoolbook)
    
    # Ranks:
    # 1a: 7
    # 1b: 7
    # 1c: 7
    # 1d: 4
    # 1e: 7
    # 1f: 7
    # 1g: 7
    # 1h: 4
    # 2a: 28
    # Total: 7*6 + 4*2 + 28 = 42 + 8 + 28 = 78.
    
    total_rank = 78
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    current_rank = 0
    
    def add_block(U_block, V_block, W_block, i_off, j_off, k_off, block_n, block_m, block_p):
        nonlocal current_rank
        block_rank = U_block.shape[1]
        
        # Copy U
        # U_block is (block_n * block_m, block_rank)
        # We need to place it in U_full at rows i_off*block_m + ...
        # Actually, U_full rows are indexed by i*m + j.
        # Block rows are local i in [0, block_n), local j in [0, block_m).
        # Global i = i_off + local_i. Global j = j_off + local_j.
        # Global row index = (i_off + local_i)*m + (j_off + local_j).
        
        for local_i in range(block_n):
            for local_j in range(block_m):
                local_row = local_i * block_m + local_j
                global_row = (i_off + local_i) * m + (j_off + local_j)
                U_full[global_row, current_rank:current_rank+block_rank] = U_block[local_row, :]
                
        # Copy V
        # V_block is (block_m * block_p, block_rank)
        # Local row = local_j * block_p + local_k
        # Global row = (j_off + local_j)*p + (k_off + local_k)
        for local_j in range(block_m):
            for local_k in range(block_p):
                local_row = local_j * block_p + local_k
                global_row = (j_off + local_j) * p + (k_off + local_k)
                V_full[global_row, current_rank:current_rank+block_rank] = V_block[local_row, :]
                
        # Copy W
        # W_block is (block_p * block_n, block_rank)
        # Local row = local_k * block_n + local_i
        # Global row = (k_off + local_k)*n + (i_off + local_i)
        for local_k in range(block_p):
            for local_i in range(block_n):
                local_row = local_k * block_n + local_i
                global_row = (k_off + local_k) * n + (i_off + local_i)
                W_full[global_row, current_rank:current_rank+block_rank] = W_block[local_row, :]
                
        current_rank += block_rank

    # Part 1: i in {0,1}
    # 1a: j in {0,1}, k in {0,1}
    U_b, V_b, W_b = _strassen_2x2x2()
    add_block(U_b, V_b, W_b, 0, 0, 0, 2, 2, 2)
    
    # 1b: j in {0,1}, k in {2,3}
    U_b, V_b, W_b = _strassen_2x2x2()
    add_block(U_b, V_b, W_b, 0, 0, 2, 2, 2, 2)
    
    # 1c: j in {0,1}, k in {4,5}
    U_b, V_b, W_b = _strassen_2x2x2()
    add_block(U_b, V_b, W_b, 0, 0, 4, 2, 2, 2)
    
    # 1d: j in {0,1}, k in {6}
    U_b, V_b, W_b = _schoolbook_2x2x1()
    add_block(U_b, V_b, W_b, 0, 0, 6, 2, 2, 1)
    
    # 1e: j in {2,3}, k in {0,1}
    U_b, V_b, W_b = _strassen_2x2x2()
    add_block(U_b, V_b, W_b, 0, 2, 0, 2, 2, 2)
    
    # 1f: j in {2,3}, k in {2,3}
    U_b, V_b, W_b = _strassen_2x2x2()
    add_block(U_b, V_b, W_b, 0, 2, 2, 2, 2, 2)
    
    # 1g: j in {2,3}, k in {4,5}
    U_b, V_b, W_b = _strassen_2x2x2()
    add_block(U_b, V_b, W_b, 0, 2, 4, 2, 2, 2)
    
    # 1h: j in {2,3}, k in {6}
    U_b, V_b, W_b = _schoolbook_2x2x1()
    add_block(U_b, V_b, W_b, 0, 2, 6, 2, 2, 1)
    
    # Part 2: i in {2}
    # 2a: j in {0,1,2,3}, k in {0,1,2,3,4,5,6}
    U_b, V_b, W_b = _schoolbook_1x4x7()
    add_block(U_b, V_b, W_b, 2, 0, 0, 1, 4, 7)
    
    # Verify
    if not _verify(U_full, V_full, W_full):
        # Fallback to schoolbook if verification fails
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    return {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }
# EVOLVE-BLOCK-END
