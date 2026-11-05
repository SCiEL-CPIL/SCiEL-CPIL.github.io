"""Matrix multiplication 3x4x8 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s kill

# EVOLVE-BLOCK-START

def _schoolbook_decomposition(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build true tensor
    T_true = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Reconstruct
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Check exactness
    if not np.allclose(T_recon, T_true, atol=tol):
        return False
    return True

def _strassen_2x2():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a00+a11)*(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (a10+a11)*b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3: a00*(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4: a11*(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[3, 3] = 1
    
    # M5: (a00+a01)*b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1
    
    # M6: (a10-a00)*(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[1, 5] = 1; W[3, 5] = 1
    
    # M7: (a01-a11)*(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[2, 6] = 1
    
    return U, V, W

def _block_strassen_decomposition(n, m, p, deadline):
    """
    Block-diagonal Strassen decomposition:
    - Split m=4 into two groups of 2: j in {0,1} and j in {2,3}
    - Split p=8 into four groups of 2: k in {0,1}, {2,3}, {4,5}, {6,7}
    
    For each (j_group, k_group) pair, we have a 3x2x2 sub-problem.
    For the 3x2x2 sub-problem, we embed the 3x2 input into a 4x2 block
    (padding the third row) and use Strassen-like decomposition.
    
    Actually, let's think more carefully.
    
    The full problem: A(3x4) * B(4x8) = C(3x8)
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Split j into j1 (0,1) and j2 (2,3):
    C[i,k] = sum_{j in j1} A[i,j]*B[j,k] + sum_{j in j2} A[i,j]*B[j,k]
    
    For each j_group (2 groups) and each k_group (4 groups of 2 k's):
    We need to compute C_sub[i, k_group] = sum_{j in j_group} A[i,j] * B[j, k_group]
    where C_sub is 3x2, A_sub is 3x2, B_sub is 2x2.
    
    This is a 3x2x2 matrix multiplication.
    
    For 3x2x2: A(3x2) * B(2x2) = C(3x2)
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    We can decompose this using 6 terms (schoolbook for 3x2x2):
    For each i in {0,1,2} and k in {0,1}:
      Term: A[i,0]*B[0,k] and A[i,1]*B[1,k]
    That's 3*2*2 = 12 terms for schoolbook.
    
    But we can do better. Notice that for fixed i:
    C[i,:] = A[i,0] * B[0,:] + A[i,1] * B[1,:]
    This is a linear combination of rows of B.
    
    Actually, the rank of the 3x2x2 tensor is at most 6 (since it's a sum of 6 rank-1 tensors
    if we view it differently), but let's just use the schoolbook 12 terms for simplicity
    and see if we can reduce.
    
    Wait, let me reconsider. The total rank using this approach:
    - 2 j_groups * 4 k_groups * (rank of 3x2x2)
    
    If 3x2x2 has rank 12 (schoolbook): 2*4*12 = 96 (same as before, no improvement)
    
    We need to find a better decomposition for 3x2x2.
    
    For 3x2x2: A(3x2) * B(2x2) = C(3x2)
    C[i,k] = sum_j A[i,j] B[j,k]
    
    This can be written as:
    C = A * B where A is 3x2 and B is 2x2.
    
    The tensor T_{i,j,k} = 1 for the matrix multiplication.
    
    One approach: Write C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    We can use 6 terms:
    For each i, we have C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
    
    This is essentially the outer product structure. The rank of this tensor
    is at most 6 (since we can write it as sum over i of rank-2 terms, but actually
    each term A[i,j]*B[j,k] is rank 1 in the tensor sense).
    
    Let me just use schoolbook for 3x2x2 which gives 12 terms, but that doesn't help.
    
    Alternative: Use the fact that 3x2x2 can be embedded in 2x2x2 by padding.
    Pad A from 3x2 to 4x2 (add a zero row), then use Strassen for 2x2x2 (rank 7).
    But we only need 3 of the 4 output rows. The rank is still 7, but we waste one row.
    
    Actually, for 3x2x2, we can use a decomposition of rank 6:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k]
    
    This is 3*2 = 6 independent products if we group by k... no, that's not right.
    
    Let me just count: the tensor has 3*2*2 = 12 entries, each of which is 1.
    The schoolbook decomposition has 12 terms. Can we do better?
    
    The lower bound for 3x2x2 is max(3*2, 2*2, 2*3) = max(6, 4, 6) = 6.
    So rank is between 6 and 12.
    
    I believe the rank of 3x2x2 matrix multiplication is 6. Let me construct it.
    
    For 3x2x2:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
    C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
    
    We can write this as 6 terms:
    For each i in {0,1,2}:
      Term i,0: A[i,0] * B[0,0] contributes to C[i,0]
      Term i,1: A[i,0] * B[0,1] contributes to C[i,1]
      ... this is getting complicated.
    
    Let me just use the schoolbook 12-term decomposition for 3x2x2 for now.
    Total rank: 2 * 4 * 12 = 96. No improvement.
    
    Let me try a different approach: use Strassen for the 2x2x2 blocks.
    
    Split n=3 into n=2 and n=1:
    A = [A_top (2x4); A_bottom (1x4)]
    C_top = A_top * B (2x8)
    C_bottom = A_bottom * B (1x8)
    
    For C_top (2x8): Split B into four 2x2 blocks (by columns):
    B = [B1(4x2), B2(4x2), B3(4x2), B4(4x2)]
    C_top = [A_top*B1, A_top*B2, A_top*B3, A_top*B4]
    Each A_top*B_i is 2x2 * 2x2 (if we also split rows of B).
    
    Actually, A_top is 2x4, B_i is 4x2. So A_top * B_i is 2x2.
    For each 2x2 * 2x2 = 2x2, use Strassen (rank 7).
    4 blocks * 7 = 28 terms for C_top.
    
    For C_bottom (1x8): A_bottom is 1x4, B_i is 4x2.
    A_bottom * B_i is 1x2.
    C_bottom[i,k] = sum_j A_bottom[j] * B[j,k]
    This is just a dot product. For each B_i (4x2), we need 4*2 = 8 terms (schoolbook).
    4 blocks * 8 = 32 terms for C_bottom.
    
    Total: 28 + 32 = 60 terms.
    
    Wait, let me recount. For C_bottom:
    A_bottom is 1x4: [a0, a1, a2, a3]
    B_i is 4x2: [[b00, b01], [b10, b11], [b20, b21], [b30, b31]]
    C_bottom = [a0*b00 + a1*b10 + a2*b20 + a3*b30, a0*b01 + a1*b11 + a2*b21 + a3*b31]
    
    This is 4 multiplications for each of the 2 output entries, so 8 terms per block.
    4 blocks * 8 = 32 terms.
    
    Total: 28 + 32 = 60.
    
    But wait, I need to be more careful about the indexing. Let me implement this.
    """
    n, m, p = 3, 4, 8
    
    # Split n=3 into top (2 rows) and bottom (1 row)
    n_top = 2
    n_bottom = 1
    
    # Split p=8 into 4 groups of 2
    p_groups = 4
    p_group_size = 2
    
    # Split m=4 into 2 groups of 2 (for Strassen in the top part)
    m_groups = 2
    m_group_size = 2
    
    # For the top part (2x4 * 4x8 = 2x8):
    # Split into 4 blocks of (2x4 * 4x2 = 2x2)
    # For each block, use Strassen (rank 7)
    
    # For the bottom part (1x4 * 4x8 = 1x8):
    # Split into 4 blocks of (1x4 * 4x2 = 1x2)
    # Each block uses schoolbook (rank 8)
    
    # Total rank: 4*7 + 4*8 = 28 + 32 = 60
    
    rank = 60
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Top part: 2x4 * 4x8 = 2x8
    # Split p into 4 groups of 2
    for kg in range(p_groups):
        k_start = kg * p_group_size
        
        # For each k group, we have a 2x2 * 2x2 = 2x2 multiplication
        # A_top is 2x4, B_sub is 4x2 (columns k_start to k_start+1)
        # We need to use Strassen for 2x2x2
        
        # Get Strassen factors for this block
        U_s, V_s, W_s = _strassen_2x2()
        
        for r in range(7):
            # U_s has shape (4, 7), corresponding to 2x2 matrix A_top
            # A_top[i, j] where i in {0,1}, j in {0,1,2,3}
            # But Strassen is for 2x2, so we need to map j in {0,1,2,3} to 2x2 blocks
            
            # Actually, A_top is 2x4, not 2x2. We need to split A_top into 2x2 blocks.
            # A_top = [A_top_left (2x2), A_top_right (2x2)]
            # B_sub = [B_sub_top (2x2); B_sub_bottom (2x2)]
            # A_top * B_sub = A_top_left * B_sub_top + A_top_right * B_sub_bottom
            
            # This is getting complicated. Let me use a different approach.
            # For 2x4 * 4x2 = 2x2, we can use schoolbook (16 terms) or find a better way.
            # Schoolbook: 2*4*2 = 16 terms per block, 4 blocks = 64 terms.
            # Plus bottom: 4*8 = 32 terms. Total: 96 terms. No improvement.
            
            # Let me just use schoolbook for the top part too.
            pass
    
    # This approach isn't working well. Let me just return the schoolbook decomposition
    # and try to improve it with a search.
    
    return _schoolbook_decomposition(n, m, p)

def _try_rank_reduction(n, m, p, deadline):
    """
    Try to find a decomposition with rank less than n*m*p.
    Uses a combination of block decomposition and random search.
    """
    schoolbook_rank = n * m * p
    best_U, best_V, best_W = _schoolbook_decomposition(n, m, p)
    best_rank = schoolbook_rank
    
    # Build true tensor
    T_true = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    lb = max(n*m, m*p, p*n)
    
    # Try a few ranks using gradient descent
    time_budget = deadline - time.time()
    
    # Try ranks from lb+10 up to schoolbook_rank-10
    for r in range(lb + 10, min(lb + 60, schoolbook_rank)):
        if time.time() > deadline:
            break
        
        # Try to find a rank-r decomposition
        success = _try_rank_r(r, n, m, p, T_true, time_budget / 50)
        if success:
            return success[0], success[1], success[2], r
    
    # If no improvement found, return schoolbook
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        raise RuntimeError("Schoolbook decomposition failed verification")
    
    return best_U, best_V, best_W, best_rank

def _try_rank_r(r, n, m, p, T_true, time_limit):
    """
    Try to find a rank-r decomposition using gradient descent.
    Returns (U, V, W) if successful, None otherwise.
    """
    if r < max(n*m, m*p, p*n):
        return None
    
    # Try multiple random initializations
    for init in range(5):
        if time.time() > time_limit:
            break
        
        rng = np.random.RandomState(42 + init)
        
        # Initialize U, V, W
        U = rng.randn(n*m, r) * 0.1
        V = rng.randn(m*p, r) * 0.1
        W = rng.randn(p*n, r) * 0.1
        
        # Gradient descent with Adam-like updates
        eta = 0.01
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        for it in range(500):
            if time.time() > time_limit:
                break
            
            # Compute residual
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            R = T_true - T_recon
            
            # Compute gradients
            grad_U = np.einsum('abc,br,cr->ar', R, V, W)
            grad_V = np.einsum('abc,ar,cr->br', R, U, W)
            grad_W = np.einsum('abc,ar,br->cr', R, U, V)
            
            # Adam update
            t = it + 1
            m_U = beta1 * m_U + (1 - beta1) * grad_U
            v_U = beta2 * v_U + (1 - beta2) * grad_U**2
            m_U_hat = m_U / (1 - beta1**t)
            v_U_hat = v_U / (1 - beta2**t)
            U += eta * m_U_hat / (np.sqrt(v_U_hat) + eps)
            
            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V**2
            m_V_hat = m_V / (1 - beta1**t)
            v_V_hat = v_V / (1 - beta2**t)
            V += eta * m_V_hat / (np.sqrt(v_V_hat) + eps)
            
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * grad_W**2
            m_W_hat = m_W / (1 - beta1**t)
            v_W_hat = v_W / (1 - beta2**t)
            W += eta * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            # Check convergence
            if it % 50 == 0:
                err = np.max(np.abs(T_recon - T_true))
                if err < 1e-10:
                    # Verify exactly
                    if _verify_decomposition(U, V, W, n, m, p):
                        return U, V, W
    
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4x8 matrix multiplication.
    Returns dict with 'U', 'V', 'W' keys, each a 2D list.
    """
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    
    # Try to find an improved decomposition
    U, V, W, rank = _try_rank_reduction(n, m, p, deadline)
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
