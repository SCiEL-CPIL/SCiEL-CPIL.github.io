"""
Rank-minimal exact tensor decomposition for 4x4 matrix multiplication.

Uses the known 49-term decomposition (Kronecker product of two Strassen rank-7
decompositions) for 4x4 matrix multiplication, which is the current best-known
upper bound on the tensor rank of M_{4,4,4}.

The decomposition is built from Strassen's 7-term 2x2 multiplication applied
at both the block level and the element level, giving 7*7 = 49 terms.
"""

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_START = time.monotonic()
_DEADLINE_LIMIT = 120.0
_SAFETY_MARGIN = 10.0

def _check_deadline():
    return (time.monotonic() - _DEADLINE_START) > (_DEADLINE_LIMIT - _SAFETY_MARGIN)

def _strassen_2x2():
    """
    Return (U, V, W) for the rank-7 Strassen decomposition of 2x2 matrix multiplication.
    
    U has shape (4, 7): U[a_idx, r] where a_idx = i*2+j for A[i,j]
    V has shape (4, 7): V[b_idx, r] where b_idx = j*2+k for B[j,k]
    W has shape (4, 7): W[c_idx, r] where c_idx = i*2+k for C[i,k] (row-major)
    
    Strassen terms (0-indexed):
        p0 = (a00+a11)*(b00+b11)
        p1 = (a10+a11)*b00
        p2 = a00*(b01-b11)
        p3 = a11*(b10-b00)
        p4 = (a00+a01)*b11
        p5 = (a10-a00)*(b00+b01)
        p6 = (a01-a11)*(b10+b11)
    
    c00 = p0 + p3 - p4 + p6
    c01 = p2 + p4
    c10 = p1 + p3
    c11 = p0 - p1 + p2 + p5
    """
    U = np.array([
        [ 1,  0,  1,  0,  1, -1,  0],  # a00
        [ 0,  0,  0,  0,  1,  0,  1],  # a01
        [ 0,  1,  0,  0,  0,  1,  0],  # a10
        [ 1,  1,  0,  1,  0,  0, -1]   # a11
    ], dtype=np.int64)
    
    V = np.array([
        [ 1,  1,  0, -1,  0,  1,  0],  # b00
        [ 0,  0,  1,  0,  0,  1,  0],  # b01
        [ 0,  0,  0,  1,  0,  0,  1],  # b10
        [ 1,  0, -1,  0,  1,  0,  1]   # b11
    ], dtype=np.int64)
    
    W = np.array([
        [ 1,  0,  0,  1, -1,  0,  1],  # c00
        [ 0,  0,  1,  0,  1,  0,  0],  # c01
        [ 0,  1,  0,  1,  0,  0,  0],  # c10
        [ 1, -1,  1,  0,  0,  1,  0]   # c11
    ], dtype=np.int64)
    
    return U, V, W

def _verify_decomposition(U, V, W, num_trials=10):
    """
    Verify the full 4x4x4 tensor decomposition.
    Checks that for random A, B: C = A@B is recovered.
    Uses the trace convention W[k*n+i, r] for C[i,k].
    """
    np.random.seed(2024)
    n, m, p = 4, 4, 4
    rank = U.shape[1]
    
    for trial in range(num_trials):
        A = np.random.randint(-10, 10, (n, m)).astype(np.int64)
        B = np.random.randint(-10, 10, (m, p)).astype(np.int64)
        C_expected = A @ B
        
        a_vec = A.flatten().astype(np.int64)
        b_vec = B.flatten().astype(np.int64)
        
        c_vec = np.zeros(p * n, dtype=np.int64)
        for s in range(rank):
            a_part = np.dot(U[:, s], a_vec)
            b_part = np.dot(V[:, s], b_vec)
            c_vec += W[:, s] * (a_part * b_part)
        
        # Reconstruct C from c_vec using trace convention: C[i,k] = c_vec[k*p + i]
        C_computed = np.zeros((n, p), dtype=np.int64)
        for i in range(n):
            for k in range(p):
                C_computed[i, k] = c_vec[k * n + i]
        
        if not np.array_equal(C_computed, C_expected):
            return False
    
    return True

def _build_4x4_decomposition():
    """
    Build the rank-49 decomposition for 4x4 matrix multiplication
    using the Kronecker product structure of Strassen's algorithm.
    
    The key idea: partition 4x4 matrices into 2x2 blocks of 2x2 sub-matrices.
    Use Strassen at the block level (7 terms), and Strassen within each block (7 terms).
    Total: 7 * 7 = 49 terms.
    """
    U_local, V_local, W_local = _strassen_2x2()
    
    # Outer Strassen coefficients
    O_A = U_local.T  # (7, 4): O_A[s][block]
    O_B = V_local.T  # (7, 4): O_B[s][block]
    O_C = W_local    # (4, 7): O_C[block][s]
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # U: A part
            for i in range(2):
                for j in range(2):
                    block_idx = i * 2 + j
                    outer_coeff = O_A[s, block_idx]
                    if outer_coeff == 0:
                        continue
                    for u in range(2):
                        for v in range(2):
                            global_a_idx = (2 * i + u) * m + (2 * j + v)
                            local_idx = u * 2 + v
                            local_coeff = U_local[local_idx, t]
                            if local_coeff != 0:
                                U[global_a_idx, col] += outer_coeff * local_coeff
            
            # V: B part
            for k in range(2):
                for l in range(2):
                    block_idx = k * 2 + l
                    outer_coeff = O_B[s, block_idx]
                    if outer_coeff == 0:
                        continue
                    for u in range(2):
                        for v in range(2):
                            global_b_idx = (2 * k + u) * p + (2 * l + v)
                            local_idx = u * 2 + v
                            local_coeff = V_local[local_idx, t]
                            if local_coeff != 0:
                                V[global_b_idx, col] += outer_coeff * local_coeff
            
            # W: C part
            for i_block in range(2):
                for k_block in range(2):
                    block_idx = i_block * 2 + k_block
                    outer_coeff = O_C[block_idx, s]
                    if outer_coeff == 0:
                        continue
                    for u in range(2):
                        for v in range(2):
                            global_c_idx = (2 * k_block + v) * n + (2 * i_block + u)
                            local_idx = u * 2 + v
                            local_coeff = W_local[local_idx, t]
                            if local_coeff != 0:
                                W[global_c_idx, col] += outer_coeff * local_coeff
    
    return U, V, W

def _build_4x4_decomposition_alt():
    """
    Alternative: use row-major W convention W[i*4+k, r] for C[i,k]
    (in case the trace convention interpretation differs)
    """
    U, V, W = _build_4x4_decomposition()
    
    W_alt = np.zeros_like(W)
    for i in range(4):
        for k in range(4):
            W_alt[i * 4 + k, :] = W[k * 4 + i, :]
    
    return U, V, W_alt

def _schoolbook_fallback():
    """Rank-64 schoolbook decomposition as fallback."""
    n, m, p = 4, 4, 4
    rank = 64
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    """
    Return a rank-49 tensor decomposition for 4x4 matrix multiplication.
    """
    if _check_deadline():
        return _schoolbook_fallback()
    
    # Build the rank-49 decomposition
    U, V, W = _build_4x4_decomposition()
    
    # Verify correctness with integer arithmetic
    if _verify_decomposition(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try alternative index convention for W
    U2, V2, W2 = _build_4x4_decomposition_alt()
    if _verify_decomposition(U2, V2, W2):
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    
    # Fallback
    return _schoolbook_fallback()
# EVOLVE-BLOCK-END
