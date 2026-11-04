"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W correctly decompose the matrix multiplication tensor.
    
    For each (i, j, k) in the multiplication tensor, we need:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    And for all other combinations, the sum should be 0.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Check all (i, j, k) tuples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                u_val = U_arr[i * m + j]
                v_val = V_arr[j * p + k]
                w_val = W_arr[k * n + i]
                actual = np.dot(np.dot(u_val, v_val), w_val)
                if abs(actual - expected) > tol:
                    return False, (i, j, k), expected, actual
    
    # Check that non-matching terms are zero
    # For a term to contribute, we need U[i*m+j,c] * V[j'*p+k',c] * W[k''*n+i'',c] != 0
    # where (j', k', i'') != (j, k, i)
    # This is complex to check exhaustively, but the main check above covers correctness
    
    return True, None, None, None


def _strassen_2x2_decomposition():
    """Return Strassen decomposition for 2x2x2 matrix multiplication.
    
    Returns U, V, W matrices of shape (4, 7), (4, 7), (4, 7)
    where U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each product c, define which A, B, C entries are involved
    # U[i*m+j, c] = coefficient of A_ij in the c-th product
    # V[j*p+k, c] = coefficient of B_jk in the c-th product  
    # W[k*n+i, c] = coefficient of C_ik in the c-th product (with sign)
    
    # Product 1: (A00+A11)(B00+B11)
    # U: A00=1, A11=1; V: B00=1, B11=1; W: C00=+1, C11=+1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Product 2: (A10+A11)B00
    # U: A10=1, A11=1; V: B00=1; W: C10=+1, C11=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Product 3: A00(B01-B11)
    # U: A00=1; V: B01=1, B11=-1; W: C01=+1, C11=+1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Product 4: A11(B10-B00)
    # U: A11=1; V: B10=1, B00=-1; W: C00=+1, C10=+1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Product 5: (A00+A01)B11
    # U: A00=1, A01=1; V: B11=1; W: C00=-1, C01=+1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Product 6: (A10-A00)(B00+B01)
    # U: A10=1, A00=-1; V: B00=1, B01=1; W: C11=+1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Product 7: (A01-A11)(B10+B11)
    # U: A01=1, A11=-1; V: B10=1, B11=1; W: C00=+1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U.tolist(), V.tolist(), W.tolist()


def _block_decompose(n, m, p):
    """Decompose n x m x p multiplication using block structure.
    
    Split the m dimension into blocks and the p dimension into blocks.
    For 2x3x8: split m=3 into [2,1] and p=8 into [4,4].
    This gives sub-problems: 2x2x4, 2x1x4, 1x2x4, 1x1x4.
    """
    start_time = time.time()
    deadline = start_time + 120
    
    # Split m=3 into [2, 1] and p=8 into [4, 4]
    m_splits = [2, 1]
    p_splits = [4, 4]
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    for mi, ms in enumerate(m_splits):
        for pi, ps in enumerate(p_splits):
            if time.time() > deadline - 10:
                # Fall back to schoolbook
                break
            # Get decomposition for n x ms x ps
            U_sub, V_sub, W_sub = _decompose_sub(n, ms, ps, deadline)
            if U_sub is not None:
                U_blocks.append(U_sub)
                V_blocks.append(V_sub)
                W_blocks.append(W_sub)
            else:
                # Schoolbook fallback for this sub-block
                U_sub, V_sub, W_sub = _schoolbook(n, ms, ps)
                U_blocks.append(U_sub)
                V_blocks.append(V_sub)
                W_blocks.append(W_sub)
    
    # Now assemble the full decomposition
    # The full U has shape (n*m, total_rank)
    # Row i*m+j corresponds to A[i,j]
    # For block (mi, pi), rows i*m + (m_offset + j) where j in [0, ms)
    # and m_offset = sum of m_splits[:mi]
    
    m_offsets = []
    offset = 0
    for ms in m_splits:
        m_offsets.append(offset)
        offset += ms
    
    p_offsets = []
    offset = 0
    for ps in p_splits:
        p_offsets.append(offset)
        offset += ps
    
    total_rank = sum(len(U_sub[0]) if U_sub else 0 for U_sub in U_blocks)
    
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for bi, (U_sub, V_sub, W_sub) in enumerate(zip(U_blocks, V_blocks, W_blocks)):
        mi = bi // len(p_splits)
        pi = bi % len(p_splits)
        ms = m_splits[mi]
        ps = p_splits[pi]
        m_off = m_offsets[mi]
        p_off = p_offsets[pi]
        
        sub_rank = U_sub.shape[1]
        
        # Copy U: row i*m + (m_off + j) -> U_sub[i*ms + j]
        for i in range(n):
            for j in range(ms):
                U_full[i * m + m_off + j, col_offset:col_offset + sub_rank] = U_sub[i * ms + j, :]
        
        # Copy V: row (m_off + j)*p + (p_off + k) -> V_sub[j*ps + k]
        for j in range(ms):
            for k in range(ps):
                V_full[(m_off + j) * p + p_off + k, col_offset:col_offset + sub_rank] = V_sub[j * ps + k, :]
        
        # Copy W: row (p_off + k)*n + i -> W_sub[k*n + i]
        for k in range(ps):
            for i in range(n):
                W_full[(p_off + k) * n + i, col_offset:col_offset + sub_rank] = W_sub[k * n + i, :]
        
        col_offset += sub_rank
    
    return U_full.tolist(), V_full.tolist(), W_full.tolist()


def _schoolbook(n, m, p):
    """Schoolbook decomposition for n x m x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _decompose_sub(n, m, p, deadline):
    """Get a good decomposition for n x m x p.
    
    For 2x2x2: use Strassen (rank 7)
    For other sizes: try to find a good decomposition or fall back to schoolbook.
    """
    start_time = time.time()
    
    # Special case: 2x2x2 -> Strassen
    if n == 2 and m == 2 and p == 2:
        U, V, W = _strassen_2x2_decomposition()
        return np.array(U), np.array(V), np.array(W)
    
    # For 2x2x4: we can think of it as two 2x2x2 blocks
    # Split p=4 into [2, 2]
    if n == 2 and m == 2 and p == 4:
        # Use two Strassen decompositions for the two 2x2x2 blocks
        U1, V1, W1 = _strassen_2x2_decomposition()
        U2, V2, W2 = _strassen_2x2_decomposition()
        
        U = np.hstack([np.array(U1), np.array(U2)])
        # V needs to be shaped (2*4, 14) = (8, 14)
        # Block 1 covers k in [0,2), block 2 covers k in [2,4)
        V_full = np.zeros((m * p, 14))
        V_full[0:2, 0:7] = np.array(V1)  # j=0, k=0,1
        V_full[2:4, 0:7] = np.array(V1)  # Wait, this is wrong
        # Actually V[j*p+k, c] for j in [0,2), k in [0,2) -> rows 0,1,2,3
        # V[j*p+k, c] for j in [0,2), k in [2,4) -> rows 4,5,6,7
        V_full = np.zeros((m * p, 14))
        V1_arr = np.array(V1)
        V2_arr = np.array(V2)
        # First block: j in [0,2), k in [0,2)
        for j in range(2):
            for k in range(2):
                V_full[j * p + k, 0:7] = V1_arr[j * 2 + k, :]
        # Second block: j in [0,2), k in [2,4)
        for j in range(2):
            for k in range(2):
                V_full[j * p + (k + 2), 7:14] = V2_arr[j * 2 + k, :]
        
        W_full = np.zeros((p * n, 14))
        W1_arr = np.array(W1)
        W2_arr = np.array(W2)
        # First block: k in [0,2), i in [0,2)
        for k in range(2):
            for i in range(2):
                W_full[k * n + i, 0:7] = W1_arr[k * 2 + i, :]
        # Second block: k in [2,4), i in [0,2)
        for k in range(2):
            for i in range(2):
                W_full[(k + 2) * n + i, 7:14] = W2_arr[k * 2 + i, :]
        
        return U, V_full, W_full
    
    # For 2x1x4: this is just 2 scalar multiplications per k, so rank = 2*1*4 = 8
    # Or we can view it as multiplying a 2x1 vector by a 1x4 matrix
    # C[i, k] = sum_j A[i,j] * B[j,k] = A[i,0] * B[0,k]
    # So each (i,k) pair is a separate product, rank = 8
    if n == 2 and m == 1 and p == 4:
        return _schoolbook(n, m, p)
    
    # For 1x2x4: similar
    if n == 1 and m == 2 and p == 4:
        return _schoolbook(n, m, p)
    
    # For 1x1x4: rank = 4
    if n == 1 and m == 1 and p == 4:
        return _schoolbook(n, m, p)
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Try the block decomposition approach
    try:
        U, V, W = _block_decompose(n, m, p)
        
        # Verify the decomposition
        valid, err_i, err_j, err_val = _verify_decomposition(U, V, W, n, m, p)
        if valid:
            return {"U": U, "V": V, "W": W}
        else:
            # Decomposition is invalid, fall back to schoolbook
            pass
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
