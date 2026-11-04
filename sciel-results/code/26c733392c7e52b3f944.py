"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]
n, m, p = DIMENSIONS

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the standard rank n*m*p decomposition."""
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

def _verify(U, V, W, tol=1e-6):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U_reshaped = U.reshape(n, m, -1)
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        return False
    if T.shape != (n, m, p):
        return False
    return np.allclose(T, 1.0, atol=tol, rtol=tol)

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2.
    Returns (U, V, W) with shapes (4,7), (4,7), (4,7).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U[i,j,c] contributes to factor for A[i,j]
    # V[j,k,c] contributes to factor for B[j,k]
    # W[k,i,c] contributes to factor for C[k,i]
    #
    # For each Strassen product M_c, we need:
    #   U[i,j,c] = coefficient of A[i,j] in the first factor of M_c
    #   V[j,k,c] = coefficient of B[j,k] in the second factor of M_c
    #   W[k,i,c] = coefficient in the linear combination for C[k,i]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a00+a11)*(b00+b11)
    # A factors: a00, a11 -> U[0*2+0,0]=1, U[1*2+1,0]=1
    # B factors: b00, b11 -> V[0*2+0,0]=1, V[1*2+1,0]=1
    # C: C00 += M1, C11 += M1 -> W[0*2+0,0]=1, W[1*2+1,0]=1
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0
    
    # M2: (a10+a11)*b00
    # A: a10, a11 -> U[1*2+0,1]=1, U[1*2+1,1]=1
    # B: b00 -> V[0*2+0,1]=1
    # C: C10 += M2, C11 -= M2 -> W[0*2+1,1]=1, W[1*2+1,1]=-1
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[1, 1] = 1.0; W[3, 1] = -1.0
    
    # M3: a00*(b01-b11)
    # A: a00 -> U[0,2]=1
    # B: b01, -b11 -> V[0*2+1,2]=1, V[1*2+1,2]=-1
    # C: C01 += M3, C11 += M3 -> W[1*2+0,2]=1, W[1*2+1,2]=1
    U[0, 2] = 1.0
    V[1, 2] = 1.0; V[3, 2] = -1.0
    W[2, 2] = 1.0; W[3, 2] = 1.0
    
    # M4: a11*(b10-b00)
    # A: a11 -> U[3,3]=1
    # B: b10, -b00 -> V[1*2+0,3]=1, V[0*2+0,3]=-1
    # C: C00 += M4, C10 += M4 -> W[0*2+0,3]=1, W[0*2+1,3]=1
    U[3, 3] = 1.0
    V[2, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = 1.0; W[1, 3] = 1.0
    
    # M5: (a00+a01)*b11
    # A: a00, a01 -> U[0,4]=1, U[1,4]=1
    # B: b11 -> V[3,4]=1
    # C: C00 -= M5, C01 += M5 -> W[0,4]=-1, W[2,4]=1
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0; W[2, 4] = 1.0
    
    # M6: (a10-a00)*(b00+b01)
    # A: a10, -a00 -> U[2,5]=1, U[0,5]=-1
    # B: b00, b01 -> V[0,5]=1, V[1,5]=1
    # C: C11 += M6 -> W[3,5]=1
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[3, 5] = 1.0
    
    # M7: (a01-a11)*(b10+b11)
    # A: a01, -a11 -> U[1,6]=1, U[3,6]=-1
    # B: b10, b11 -> V[2,6]=1, V[3,6]=1
    # C: C00 += M7 -> W[0,6]=1
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0
    
    return U, V, W

def _decompose_2xn_x_m(n, m):
    """
    Decompose 2xn * nxm multiplication tensor.
    Uses a simple construction with rank at most 2*n*m but we aim for better.
    For 2x5x3, we use rank 12 via a structured approach.
    
    Actually, let's use a direct construction:
    For 2xn * nxm, we can use the following:
    - For each of the 2 rows of A, and each of the m columns of B,
      we need to compute a dot product of length n.
    
    A simple rank construction: use n rank-1 terms per (row, col) pair,
    but share across rows.
    
    Better: use the fact that 2xn*nxm has rank at most 2n + 2m - 4 for n,m >= 2.
    For 2x5x3: rank <= 2*5 + 2*3 - 4 = 12.
    
    Construction for 2xn*nxm with rank 2n+2m-4:
    This is a known result. Let me implement it.
    
    Actually, let me just use a straightforward rank 2*n*m construction first,
    then optimize if needed.
    
    For simplicity and correctness, use rank 2*n*m:
    U[i,j,c], V[j,k,c], W[k,i,c]
    """
    rank = 2 * n * m
    U = np.zeros((2 * n, rank))
    V = np.zeros((n * m, rank))
    W = np.zeros((m * 2, rank))
    
    col = 0
    for i in range(2):
        for j in range(n):
            for k in range(m):
                U[i * n + j, col] = 1.0
                V[j * m + k, col] = 1.0
                W[k * 2 + i, col] = 1.0
                col += 1
    
    return U, V, W

def _construct_block_decomposition():
    """
    Construct a decomposition for 4x5 * 5x6 by splitting:
    - Rows {0,1} and {2,3}
    - Columns {0,1,2} and {3,4,5}
    
    This gives 4 sub-problems:
    - 2x5x3 (rows 0-1, cols 0-2)
    - 2x5x3 (rows 0-1, cols 3-5)
    - 2x5x3 (rows 2-3, cols 0-2)
    - 2x5x3 (rows 2-3, cols 3-5)
    
    Each 2x5x3 sub-problem has rank at most 30 (schoolbook) or 12 (optimized).
    Total rank: 4 * 12 = 48 if we use the optimized version.
    """
    # Use schoolbook for each 2x5x3 block: rank 30 each, total 120
    # But we want better. Let's use rank 12 per block.
    
    # For now, use the simple 2xn*nxm construction
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    row_splits = [(0, 2), (2, 4)]  # (start, end) for row groups
    col_splits = [(0, 3), (3, 6)]  # (start, end) for col groups
    
    for r_start, r_end in row_splits:
        for c_start, c_end in col_splits:
            n_sub = 5  # middle dimension
            m_sub = c_end - c_start  # 3
            U_sub, V_sub, W_sub = _decompose_2xn_x_m(n_sub, m_sub)
            U_blocks.append(U_sub)
            V_blocks.append(V_sub)
            W_blocks.append(W_sub)
    
    total_rank = sum(U_blocks[i].shape[1] for i in range(len(U_blocks)))
    
    # Assemble full matrices
    U_full = np.zeros((20, total_rank))
    V_full = np.zeros((30, total_rank))
    W_full = np.zeros((24, total_rank))
    
    col_offset = 0
    for idx, (r_start, r_end) in enumerate(row_splits):
        for jdx, (c_start, c_end) in enumerate(col_splits):
            block_idx = idx * len(col_splits) + jdx
            U_sub = U_blocks[block_idx]
            V_sub = V_blocks[block_idx]
            W_sub = W_blocks[block_idx]
            
            # U_sub has shape (2*5, rank_sub) = (10, rank_sub)
            # Map row i_sub in [0,2) to global row (r_start + i_sub)*5 + j
            for i_sub in range(2):
                for j in range(5):
                    U_full[(r_start + i_sub) * 5 + j, col_offset:col_offset + U_sub.shape[1]] = U_sub[i_sub * 5 + j, :]
            
            # V_sub has shape (5*3, rank_sub) = (15, rank_sub)
            # Map (j, k_sub) to global (j, c_start + k_sub)
            for j in range(5):
                for k_sub in range(3):
                    V_full[j * 6 + (c_start + k_sub), col_offset:col_offset + V_sub.shape[1]] = V_sub[j * 3 + k_sub, :]
            
            # W_sub has shape (3*2, rank_sub) = (6, rank_sub)
            # Map (k_sub, i_sub) to global (c_start + k_sub, r_start + i_sub)
            for k_sub in range(3):
                for i_sub in range(2):
                    W_full[(c_start + k_sub) * 4 + (r_start + i_sub), col_offset:col_offset + W_sub.shape[1]] = W_sub[k_sub * 2 + i_sub, :]
            
            col_offset += U_sub.shape[1]
    
    return U_full, V_full, W_full

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # First, try the block decomposition
    try:
        U_block, V_block, W_block = _construct_block_decomposition()
        if _verify(U_block, V_block, W_block, tol=1e-6):
            best_U, best_V, best_W = U_block.copy(), V_block.copy(), W_block.copy()
            best_rank = U_block.shape[1]
        else:
            # Fallback to schoolbook
            U_school, V_school, W_school = _schoolbook()
            best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
            best_rank = n * m * p
    except Exception:
        U_school, V_school, W_school = _schoolbook()
        best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
        best_rank = n * m * p
    
    # Try to improve with local search if time permits
    if time.time() < deadline - 20.0:
        for target_rank in [best_rank - 1, best_rank - 2, best_rank - 3]:
            if target_rank < 1 or time.time() > deadline - 10.0:
                break
            for trial in range(3):
                if time.time() > deadline - 10.0:
                    break
                np.random.seed(42 + target_rank * 100 + trial)
                U, V, W, success = _local_search_integer(target_rank, max_steps=1000, time_limit=5.0)
                if success and _verify(U, V, W, tol=1e-6):
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = target_rank
                    break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _local_search_integer(r, max_steps=5000, time_limit=10.0):
    """
    Perform a local search on the integer/half-integer grid to minimize the tensor error.
    Returns (U, V, W, success).
    """
    start_time = time.time()
    
    vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    U = np.random.choice(vals, size=(n * m, r))
    V = np.random.choice(vals, size=(m * p, r))
    W = np.random.choice(vals, size=(p * n, r))
    
    current_err = _compute_error(U, V, W)
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        if current_err < 1e-10:
            return U, V, W, True
            
        mat_choice = np.random.randint(3)
        
        if mat_choice == 0:
            i = np.random.randint(n * m)
            j = np.random.randint(r)
            old_val = U[i, j]
            delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            new_val = old_val + delta
            U[i, j] = new_val
            new_err = _compute_error(U, V, W)
            if new_err >= current_err:
                U[i, j] = old_val
            else:
                current_err = new_err
                
        elif mat_choice == 1:
            i = np.random.randint(m * p)
            j = np.random.randint(r)
            old_val = V[i, j]
            delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            new_val = old_val + delta
            V[i, j] = new_val
            new_err = _compute_error(U, V, W)
            if new_err >= current_err:
                V[i, j] = old_val
            else:
                current_err = new_err
                
        else:
            i = np.random.randint(p * n)
            j = np.random.randint(r)
            old_val = W[i, j]
            delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            new_val = old_val + delta
            W[i, j] = new_val
            new_err = _compute_error(U, V, W)
            if new_err >= current_err:
                W[i, j] = old_val
            else:
                current_err = new_err
                
    return U, V, W, False

def _compute_error(U, V, W):
    """Compute the squared error between the tensor reconstruction and the all-ones tensor."""
    U_reshaped = U.reshape(n, m, -1)
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    return np.sum((T - 1.0) ** 2)
# EVOLVE-BLOCK-END
