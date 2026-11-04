"""
Rank-minimal exact tensor decomposition for 4x4 matrix multiplication.

Uses a known 49-term decomposition for 4x4 matrix multiplication,
which is the current best-known upper bound on the tensor rank of
M_{4,4,4}.

The decomposition is built from Strassen's 7-term 2x2 multiplication
combined with an optimal 7-term 2x2 combination scheme (the "7x7"
Strassen variant), giving 7*7 = 49 terms.
"""

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_START = time.monotonic()
_DEADLINE_LIMIT = 120.0  # seconds; evaluator kills at 150s, use 120s as hard limit
_SAFETY_MARGIN = 10.0    # return at least 10s before the hard limit

def _check_deadline():
    """Return True if we should stop and return current best."""
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

def _verify_2x2(U, V, W):
    """Verify the 2x2 Strassen decomposition is correct."""
    np.random.seed(42)
    for _ in range(20):
        A = np.random.randint(-10, 10, (2, 2)).astype(np.int64)
        B = np.random.randint(-10, 10, (2, 2)).astype(np.int64)
        C_expected = A @ B
        # Flatten A, B to vectors
        a_vec = A.flatten()  # [a00, a01, a10, a11]
        b_vec = B.flatten()  # [b00, b01, b10, b11]
        # Compute C via tensor decomposition: C_flat[r] = sum_s U[a_idx, s] * V[b_idx, s] * W[c_idx, s]
        # Actually: C[c_idx] = sum_s W[c_idx, s] * (sum_a U[a_idx, s] * A[a_idx]) * (sum_b V[b_idx, s] * B[b_idx])
        c_vec = np.zeros(4, dtype=np.int64)
        for s in range(7):
            a_part = np.dot(U[:, s], a_vec)
            b_part = np.dot(V[:, s], b_vec)
            c_vec += W[:, s] * (a_part * b_part)
        C_computed = c_vec.reshape(2, 2)
        if not np.array_equal(C_computed, C_expected):
            return False
    return True

def _verify_4x4(U, V, W):
    """Verify the 4x4 decomposition is correct.
    
    U has shape (16, rank): U[i*4+j, r] for A[i,j]
    V has shape (16, rank): V[j*4+k, r] for B[j,k]  
    W has shape (16, rank): W[k*4+i, r] for C[i,k] (trace convention)
    """
    np.random.seed(123)
    for _ in range(10):
        A = np.random.randint(-5, 5, (4, 4)).astype(np.int64)
        B = np.random.randint(-5, 5, (4, 4)).astype(np.int64)
        C_expected = A @ B
        a_vec = A.flatten()  # row-major: A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], ...
        b_vec = B.flatten()  # row-major: B[0,0], B[0,1], B[0,2], B[0,3], B[1,0], ...
        
        rank = U.shape[1]
        c_vec = np.zeros(16, dtype=np.int64)
        for s in range(rank):
            a_part = np.dot(U[:, s], a_vec)
            b_part = np.dot(V[:, s], b_vec)
            c_vec += W[:, s] * (a_part * b_part)
        
        # c_vec is indexed by W[k*4+i] which corresponds to C[i,k]
        # So C[i,k] = c_vec[k*4+i]
        C_computed = np.zeros((4, 4), dtype=np.int64)
        for i in range(4):
            for k in range(4):
                C_computed[i, k] = c_vec[k * 4 + i]
        
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
    
    For the outer level (block-level Strassen):
        Outer term s uses:
            A_outer[s] = sum of certain 2x2 blocks of A (with coefficients from Strassen)
            B_outer[s] = sum of certain 2x2 blocks of B (with coefficients from Strassen)
            C_outer[block][s] = coefficients for reconstructing C blocks from M_s
    
    For the inner level (within each 2x2 block):
        Inner term t uses Strassen coefficients for 2x2 multiplication.
    
    The combined term (s, t) has rank index s*7 + t.
    
    U[global_a_idx, s*7+t] = O_A[s][block] * U_local[t][local]
    V[global_b_idx, s*7+t] = O_B[s][block] * V_local[t][local]
    W[global_c_idx, s*7+t] = O_C[block][s] * W_local[t][local]
    
    Where:
        global_a_idx = (2*i+u)*4 + (2*j+v) for block (i,j), local (u,v)
        global_b_idx = (2*k+u)*4 + (2*l+v) for block (k,l), local (u,v)
        global_c_idx = (2*k+v)*4 + (2*i+u) for C[i,k] -> W index k*4+i
                     where block (i,k) and local (u,v)
    """
    U_local, V_local, W_local = _strassen_2x2()
    
    # Outer Strassen coefficients (same as inner for 2x2 block structure)
    # O_A[s][block] where block = i*2+j for block (i,j) of the 2x2 block matrix
    O_A = U_local  # Same pattern: row = block index, col = term
    # Wait, U_local has shape (4, 7) where row = local index (u*2+v), col = term s
    # We need O_A[s][block] which is (7, 4)
    O_A = U_local.T  # (7, 4): O_A[s][block]
    
    O_B = V_local.T  # (7, 4): O_B[s][block]
    
    # O_C: for reconstruction, C_block[i,j] = sum_s O_C[block(i,j)][s] * M_s
    # W_local has shape (4, 7) where row = local c index, col = term
    # We need O_C[block][s] where block = i*2+j
    O_C = W_local  # (4, 7): O_C[block][s]
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # U: A part
            # Block (i,j) of A, local (u,v)
            # global_a_idx = (2*i+u) * 4 + (2*j+v)
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
            # Block (k,l) of B, local (u,v)
            # global_b_idx = (2*k+u) * 4 + (2*l+v)
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
            # C[i,k] is in block (i_block, k_block) with local (u, v)
            # where i = 2*i_block + u, k = 2*k_block + v
            # W index: k * n + i = (2*k_block + v) * 4 + (2*i_block + u)
            for i_block in range(2):
                for k_block in range(2):
                    block_idx = i_block * 2 + k_block
                    outer_coeff = O_C[block_idx, s]
                    if outer_coeff == 0:
                        continue
                    for u in range(2):
                        for v in range(2):
                            # C row = 2*i_block + u, C col = 2*k_block + v
                            global_c_idx = (2 * k_block + v) * n + (2 * i_block + u)
                            local_idx = u * 2 + v
                            local_coeff = W_local[local_idx, t]
                            if local_coeff != 0:
                                W[global_c_idx, col] += outer_coeff * local_coeff
    
    return U, V, W

def _verify_decomposition(U, V, W):
    """
    Verify the full 4x4x4 tensor decomposition.
    Checks that for random A, B: C = A@B is recovered.
    Uses the trace convention W[k*n+i, r] for C[i,k].
    """
    np.random.seed(2024)
    for trial in range(5):
        A = np.random.randint(-10, 10, (4, 4)).astype(np.int64)
        B = np.random.randint(-10, 10, (4, 4)).astype(np.int64)
        C_expected = A @ B
        
        a_vec = A.flatten().astype(np.float64)
        b_vec = B.flatten().astype(np.float64)
        
        rank = U.shape[1]
        c_vec = np.zeros(16, dtype=np.float64)
        for s in range(rank):
            a_part = np.dot(U[:, s], a_vec)
            b_part = np.dot(V[:, s], b_vec)
            c_vec += W[:, s] * (a_part * b_part)
        
        # Reconstruct C from c_vec using trace convention: C[i,k] = c_vec[k*4+i]
        C_computed = np.zeros((4, 4), dtype=np.float64)
        for i in range(4):
            for k in range(4):
                C_computed[i, k] = c_vec[k * 4 + i]
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            print(f"Verification FAILED at trial {trial}")
            print(f"Expected:\n{C_expected}")
            print(f"Computed:\n{C_computed}")
            print(f"Diff:\n{C_computed - C_expected}")
            return False
    
    return True

def construct():
    """
    Return a rank-49 tensor decomposition for 4x4 matrix multiplication.
    """
    if _check_deadline():
        # Fallback: return schoolbook rank-64 (shouldn't happen, but safety)
        return _schoolbook_fallback()
    
    # Build the rank-49 decomposition
    U, V, W = _build_4x4_decomposition()
    
    # Verify correctness
    if not _verify_decomposition(U, V, W):
        print("Rank-49 decomposition failed verification, trying fallback...")
        if _check_deadline():
            return _schoolbook_fallback()
        # Try alternative index convention for W
        U2, V2, W2 = _build_4x4_decomposition_alt()
        if _verify_decomposition(U2, V2, W2):
            return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
        else:
            print("Alt decomposition also failed, using schoolbook fallback")
            return _schoolbook_fallback()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_4x4_decomposition_alt():
    """
    Alternative: use row-major W convention W[i*4+k, r] for C[i,k]
    (in case the trace convention interpretation differs)
    """
    U, V, W = _build_4x4_decomposition()
    # If the trace convention means W is indexed differently, try permuting
    # Actually, let's try: maybe W[i*4+k, r] is the standard row-major for C
    # The task says W[k*n+i, r], which we used. Let's also try the other way.
    
    # Try: C[i,k] stored at W[i*4+k, r] instead of W[k*4+i, r]
    W_alt = np.zeros_like(W)
    for i in range(4):
        for k in range(4):
            W_alt[i * 4 + k, :] = W[k * 4 + i, :]
    
    return U, V, W_alt

def _schoolbook_fallback():
    """Rank-64 schoolbook decomposition as fallback."""
    n, m, p = 4, 4, 4
    rank = n * m  # 16... no, schoolbook is n*m*p = 64
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
                # W[k*n + i, r] = 1 for C[i,k]
                W[k * n + i, r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
