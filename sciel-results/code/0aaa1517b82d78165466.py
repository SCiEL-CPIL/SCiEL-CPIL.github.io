"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify if U, V, W satisfy the tensor identity exactly.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Identity: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta_{j,j'}
    """
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += W[k * n + i, c] * U[i * m + j, c] * V[jp * p + k, c]
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification using tensor operations."""
    r = U.shape[1]
    # Build the full tensor: T[i, j, jp, k] = sum_c U[i*m+j, c] * V[jp*p+k, c] * W[k*n+i, c]
    # We need T[i, j, jp, k] = delta_{j,jp} for all i, k
    
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    # For each (i, k), compute sum over j, jp: check if sum_c U[i,j,c]*V[jp,k,c]*W[k,i,c] = delta_{j,jp}
    # Actually, the identity is: for each (i, j, jp, k): sum_c U[i*m+j,c]*V[jp*p+k,c]*W[k*n+i,c] = delta_{j,jp}
    
    # Let's compute the 4D tensor
    # T[i, j, jp, k] = sum_c U[i,j,c] * V[jp,k,c] * W[k,i,c]
    
    # Use einsum: U[i,j,c] * V[jp,k,c] * W[k,i,c] -> sum over c
    # T[i,j,jp,k] = np.einsum('ijc,jkc,kic->ijjk', U3, V3, W3)
    
    T = np.einsum('ijc,jkc,kic->ijjk', U3, V3, W3)
    
    # Expected: for each i, k: T[i, j, jp, k] = delta_{j,jp}
    for i in range(n):
        for k in range(p):
            expected = np.eye(m)
            if not np.allclose(T[i, :, :, k], expected, atol=tol):
                return False
    return True

def _get_strassen_222():
    """Returns U, V, W for 2x2x2 multiplication using Strassen's algorithm (Rank 7)."""
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_school_221():
    """Returns U, V, W for 2x2x1 multiplication using schoolbook (Rank 4)."""
    n, m, p = 2, 2, 1
    r = 4
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for c in range(r):
        i = c // m
        j = c % m
        k = 0
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
        
    return U, V, W

def _build_block_decomposition(n, m, p, m_blocks, p_blocks, U_strassen, V_strassen, W_strassen, U_school, V_school, W_school):
    """
    Build a block-structured decomposition.
    m_blocks: list of (start, end) for m dimension
    p_blocks: list of (start, end, 'strassen' or 'school') for p dimension
    Returns U, V, W and total rank.
    """
    total_rank = 0
    for j_start, j_end in m_blocks:
        for k_start, k_end, p_type in p_blocks:
            if p_type == 'strassen':
                r_sub = 7
            else:
                r_sub = 4
            total_rank += r_sub
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for j_start, j_end in m_blocks:
        m_sub = j_end - j_start
        for k_start, k_end, p_type in p_blocks:
            p_sub = k_end - k_start
            if p_type == 'strassen':
                U_sub, V_sub, W_sub = U_strassen, V_strassen, W_strassen
                r_sub = 7
            else:
                U_sub, V_sub, W_sub = U_school, V_school, W_school
                r_sub = 4
            
            for i in range(n):
                for j_sub in range(m_sub):
                    j = j_start + j_sub
                    U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_sub, :]
            for j_sub in range(m_sub):
                j = j_start + j_sub
                for k_sub in range(p_sub):
                    k = k_start + k_sub
                    V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_sub + k_sub, :]
            for k_sub in range(p_sub):
                k = k_start + k_sub
                for i in range(n):
                    W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]
            col_offset += r_sub
    
    return U, V, W, total_rank

def _try_vectorized_descent(n, m, p, rank, deadline, seed=42, max_iter=2000):
    """
    Vectorized gradient descent to find a rank-r decomposition.
    Uses efficient tensor operations.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with small random values
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    # Target tensor: T[i,j,jp,k] = delta_{j,jp}
    # Error: err[i,j,jp,k] = sum_c U[i*m+j,c]*V[jp*p+k,c]*W[k*n+i,c] - delta_{j,jp}
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.05
    lr_decay = 0.99
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute error tensor T[i,j,jp,k] = sum_c U[i,j,c]*V[jp,k,c]*W[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijjk', U3, V3, W3)
        
        # Target: delta_{j,jp} for each (i,k)
        # err[i,j,jp,k] = T[i,j,jp,k] - delta_{j,jp}
        err = T.copy()
        for i in range(n):
            for k in range(p):
                err[i, :, :, k] -= np.eye(m)
        
        cur_err = np.sum(err**2)
        if cur_err < best_err:
            best_err = cur_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-12:
            break
        
        # Gradient: dU[i,j,c] = sum_{jp,k} err[i,j,jp,k] * V[jp,k,c] * W[k,i,c]
        # dV[jp,k,c] = sum_{i,j} err[i,j,jp,k] * U[i,j,c] * W[k,i,c]
        # dW[k,i,c] = sum_{j,jp} err[i,j,jp,k] * U[i,j,c] * V[jp,k,c]
        
        dU = np.einsum('ijjk,jkc,kic->ijc', err, V3, W3)
        dV = np.einsum('ijjk,ijc,kic->jkc', err, U3, W3)
        dW = np.einsum('ijjk,ijc,jkc->kic', err, U3, V3)
        
        U -= lr * dU.reshape(n*m, rank)
        V -= lr * dV.reshape(m*p, rank)
        W -= lr * dW.reshape(p*n, rank)
        
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        lr *= lr_decay
    
    if best_err < 1e-8:
        # Try rounding to nearest integers/half-integers
        for scale in [2, 1]:
            U_r = np.round(best_U * scale) / scale
            V_r = np.round(best_V * scale) / scale
            W_r = np.round(best_W * scale) / scale
            if _verify(U_r, V_r, W_r, n, m, p):
                return U_r, V_r, W_r
        # Also try integer rounding
        U_r = np.round(best_U)
        V_r = np.round(best_V)
        W_r = np.round(best_W)
        if _verify(U_r, V_r, W_r, n, m, p):
            return U_r, V_r, W_r
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin for 150s limit
    
    U_strassen, V_strassen, W_strassen = _get_strassen_222()
    U_school, V_school, W_school = _get_school_221()
    
    # Build the known rank-50 decomposition
    m_blocks = [(0, 2), (2, 4)]
    p_blocks = [
        (0, 2, 'strassen'),
        (2, 4, 'strassen'),
        (4, 6, 'strassen'),
        (6, 7, 'school')
    ]
    
    U, V, W, total_rank = _build_block_decomposition(n, m, p, m_blocks, p_blocks, 
                                                       U_strassen, V_strassen, W_strassen,
                                                       U_school, V_school, W_school)
    
    # Verify the rank-50 decomposition
    if _verify(U, V, W, n, m, p):
        known_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        known_rank = total_rank
    else:
        # Fallback to schoolbook
        r = n * m * p
        U_fb = np.zeros((n * m, r))
        V_fb = np.zeros((m * p, r))
        W_fb = np.zeros((p * n, r))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, column] = 1
                    V_fb[j * p + k, column] = 1
                    W_fb[k * n + i, column] = 1
                    column += 1
        known_result = {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        known_rank = r
    
    best_result = known_result
    best_rank = known_rank
    
    # Try different block structures to find lower rank
    # Strategy: try various ways to split m and p into blocks of size 2 (Strassen) and 1 (school)
    
    # Generate all possible block structures
    def generate_m_splits(m):
        """Generate all ways to split m into blocks of size 2 and 1."""
        splits = []
        def helper(remaining, current):
            if remaining == 0:
                splits.append(current[:])
                return
            if remaining >= 2:
                current.append(2)
                helper(remaining - 2, current)
                current.pop()
            if remaining >= 1:
                current.append(1)
                helper(remaining - 1, current)
                current.pop()
        helper(m, [])
        return splits
    
    def generate_p_splits(p):
        """Generate all ways to split p into blocks of size 2 and 1."""
        splits = []
        def helper(remaining, current):
            if remaining == 0:
                splits.append(current[:])
                return
            if remaining >= 2:
                current.append(2)
                helper(remaining - 2, current)
                current.pop()
            if remaining >= 1:
                current.append(1)
                helper(remaining - 1, current)
                current.pop()
        helper(p, [])
        return splits
    
    m_splits = generate_m_splits(m)
    p_splits = generate_p_splits(p)
    
    # Try all combinations of m and p splits
    tried_ranks = set()
    
    for m_split in m_splits:
        # Convert block sizes to (start, end) pairs
        m_blocks_list = []
        start = 0
        for size in m_split:
            m_blocks_list.append((start, start + size))
            start += size
        
        for p_split in p_splits:
            p_blocks_list = []
            start = 0
            for size in p_split:
                if size == 2:
                    p_blocks_list.append((start, start + 2, 'strassen'))
                else:
                    p_blocks_list.append((start, start + 1, 'school'))
                start += size
            
            # Calculate rank for this structure
            rank = 0
            for j_start, j_end in m_blocks_list:
                m_sub = j_end - j_start
                for k_start, k_end, p_type in p_blocks_list:
                    if p_type == 'strassen':
                        rank += 7
                    else:
                        rank += 4
            
            if rank >= best_rank or rank in tried_ranks:
                continue
            tried_ranks.add(rank)
            
            if time.time() > deadline - 30:
                break
            
            U_try, V_try, W_try, _ = _build_block_decomposition(n, m, p, m_blocks_list, p_blocks_list,
                                                                 U_strassen, V_strassen, W_strassen,
                                                                 U_school, V_school, W_school)
            
            if _verify(U_try, V_try, W_try, n, m, p):
                best_rank = rank
                best_result = {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
    
    # Try vectorized descent for ranks below the best found
    if time.time() < deadline - 20:
        for rank in range(best_rank - 1, max(best_rank - 10, 20), -1):
            if time.time() > deadline - 15:
                break
            
            result = _try_vectorized_descent(n, m, p, rank, deadline - 10, seed=rank * 7919 + 42, max_iter=500)
            if result is not None:
                U_r, V_r, W_r = result
                if _verify(U_r, V_r, W_r, n, m, p):
                    best_rank = rank
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    break
    
    return best_result

# EVOLVE-BLOCK-END
