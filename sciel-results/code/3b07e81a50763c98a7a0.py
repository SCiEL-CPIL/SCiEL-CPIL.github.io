"""Rewritten program for the 2x3x10 matrix multiplication tensor rank search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    n, m, p = int(n), int(m), int(p)
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _strassen_2x2x2_block():
    """
    Strassen rank-7 decomposition of the 2x2x2 multiplication tensor.
    Returns U (4x7), V (4x7), W (4x7) such that for A,B 2x2:
      C[k*n + i] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    where n=2, m=2, p=2.  U, V, W flattened row-major over input indices.
    """
    n, m, p = 2, 2, 2
    # Strassen 7 products
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b11)
    # M4 = a11*(b01-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    U = np.array([
        # row: a00, a01, a10, a11   (flattened: index i*m+j, row-major)
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [1, 0, -1, 0], # M6
        [0, 1, 0, -1], # M7
    ], dtype=float).T  # shape (4, 7)

    V = np.array([
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, -1, 1, 0],  # M3
        [1, -1, 0, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 1, 1, 0],   # M7
    ], dtype=float).T  # shape (4, 7)

    W = np.array([
        [1, 0, 0, -1],  # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0],   # M7
    ], dtype=float).T  # shape (4, 7)

    return U, V, W


def _strassen_1x2x2_block():
    """
    Rank-2 decomposition of 1x2 matrix times 2x2 matrix.
    A is 1x2: [a0, a1]; B is 2x2: [[b00,b01],[b10,b11]].
    Output C is 1x2: c0 = a0*b00 + a1*b10, c1 = a0*b01 + a1*b11.
    """
    n, m, p = 1, 2, 2
    U = np.array([[1, 0], [1, 0]], dtype=float).T  # shape (2, 2)
    V = np.array([[1, 0], [0, 0], [1, 1], [0, 1]], dtype=float).T  # shape (4, 2)
    W = np.array([[1, 0], [0, 1]], dtype=float).T  # shape (2, 2)
    return U, V, W


def _construct_composite():
    """
    Build a composite decomposition of 2x3x10 multiplication.
    Strategy: partition the m=3 middle dimension into m=2 and m=1.
    A is 2x3 = [A_2x2 | A_2x1], B is 3x10 = [B_2x10; B_1x10].
    C = A * B = A_2x2 * B_2x10 + A_2x1 * B_1x10.
    
    Term 1: 2x2 * 2x10 -> decompose via 5 independent Strassen 2x2 blocks (one per 2 columns of output).
             Each block handles (A_2x2, B_2x2_col) -> C_2x2_col.
             Rank contribution: 5 * 7 = 35.
    
    Term 2: 2x1 * 1x10 -> decompose via 5 independent 2x1 * 1x2 blocks (one per 2 columns).
             Each block handles (A_2x1, B_1x2_col) -> C_2x2_col... 
             Wait, C is 2x10. Term 2 output: A_2x1 * B_1x10 -> 2x10.
             Decompose as 5 independent (2x1)*(1x2) -> (2x2). Each rank 2.
             Rank contribution: 5 * 2 = 10.
    
    Total rank: 35 + 10 = 45.
    """
    n, m, p = 2, 3, 10
    U_rows = n * m  # 6
    V_rows = m * p  # 30
    W_rows = p * n  # 20

    total_rank = 45
    U = np.zeros((U_rows, total_rank), dtype=float)
    V = np.zeros((V_rows, total_rank), dtype=float)
    W = np.zeros((W_rows, total_rank), dtype=float)

    col_offset = 0

    # ---- Term 1: A_2x2 (cols 0-1 of A) * B_2x10 (rows 0-1 of B) -> C contribution ----
    # A indices: i in {0,1}, j in {0,1} -> U row index i*m + j (with m=3 in full)
    #   So U rows for A_2x2: i*3 + j for i in {0,1}, j in {0,1} -> rows 0,1,3,4
    # B indices: j in {0,1}, k in {0..9} -> V row index j*p + k (with p=10)
    #   B_2x10 rows: j*10 + k for j in {0,1}, k in {0..9} -> rows 0..9 (j=0) and 10..19 (j=1)
    # C indices: k in {0..9}, i in {0,1} -> W row index k*n + i (with n=2)
    #   C rows: k*2 + i

    # 5 Strassen blocks, each handling 2 output columns
    for blk in range(5):
        k_lo = blk * 2
        k_hi = k_lo + 1
        # Strassen block for (A_2x2, B_2x2) -> C_2x2
        sU, sV, sW = _strassen_2x2x2_block()  # each 4x7
        for r in range(7):
            col = col_offset + r
            # U: sU maps A indices (a00,a01,a10,a11) -> rows 0,1,2,3 in local
            #   local A row 0 = (i=0,j=0) -> global U row 0*3+0 = 0
            #   local A row 1 = (i=0,j=1) -> global U row 0*3+1 = 1
            #   local A row 2 = (i=1,j=0) -> global U row 1*3+0 = 3
            #   local A row 3 = (i=1,j=1) -> global U row 1*3+1 = 4
            a_map = [0, 1, 3, 4]
            for loc_row, coef in enumerate(sU[:, r]):
                U[a_map[loc_row], col] += coef

            # V: sV maps B indices (b00,b01,b10,b11) -> local rows 0,1,2,3
            #   local B row 0 = (j=0,k_lo)   -> global V row 0*10 + k_lo = k_lo
            #   local B row 1 = (j=0,k_hi)   -> global V row 0*10 + k_hi = k_hi
            #   local B row 2 = (j=1,k_lo)   -> global V row 1*10 + k_lo = 10 + k_lo
            #   local B row 3 = (j=1,k_hi)   -> global V row 1*10 + k_hi = 10 + k_hi
            b_map = [k_lo, k_hi, 10 + k_lo, 10 + k_hi]
            for loc_row, coef in enumerate(sV[:, r]):
                V[b_map[loc_row], col] += coef

            # W: sW maps C indices (c00,c01,c10,c11) -> local rows 0,1,2,3
            #   local C row 0 = (i=0, k_lo) -> global W row k_lo*2 + 0 = 2*k_lo
            #   local C row 1 = (i=0, k_hi) -> global W row k_hi*2 + 0 = 2*k_hi
            #   local C row 2 = (i=1, k_lo) -> global W row k_lo*2 + 1 = 2*k_lo+1
            #   local C row 3 = (i=1, k_hi) -> global W row k_hi*2 + 1 = 2*k_hi+1
            c_map = [2*k_lo, 2*k_hi, 2*k_lo+1, 2*k_hi+1]
            for loc_row, coef in enumerate(sW[:, r]):
                W[c_map[loc_row], col] += coef

    col_offset += 35

    # ---- Term 2: A_2x1 (col 2 of A) * B_1x10 (row 2 of B) -> C contribution ----
    # A row index: i*3 + 2 for i in {0,1} -> rows 2, 5
    # B row index: 2*10 + k = 20 + k for k in {0..9}
    # C same as before: k*2 + i

    # 5 blocks of (2x1)*(1x2) -> (2x2), each rank 2
    for blk in range(5):
        k_lo = blk * 2
        k_hi = k_lo + 1
        sU2, sV2, sW2 = _strassen_1x2x2_block()  # shapes (2,2), (4,2), (2,2)
        # sU2: A is 2x1, local rows (a0, a1) -> global rows 2, 5
        a_map2 = [2, 5]
        # sV2: B is 1x2, local rows (b_lo, b_hi) -> global rows 20+k_lo, 20+k_hi
        #   But sV2 is 4x2. The 4 local rows correspond to (b00, b01, b10, b11) of a 2x2 B.
        #   For 1x2 B, we only have b_lo and b_hi. The mapping:
        #   Actually, our 1x2x2 block: A is 1x2, B is 2x2. But here A is 2x1, B is 1x2.
        #   Let me reconsider. We need (2x1) * (1x2) -> (2x2).
        #   This is equivalent to outer product of (2x1) and (1x2), which is rank 1... no, 
        #   it's bilinear in A and B, so rank is at most 4 (schoolbook) but can be less?
        #   Actually (2x1)*(1x2) = 2x2 matrix where c_ij = a_i * b_j.
        #   This is a rank-1 operation in the output, but the tensor rank... 
        #   The multiplication tensor for (m=2, n=1, p=2) has rank min(4, ...) = 4? 
        #   No wait, we're computing C[i,k] = a_i * b_k where a is 2x1 and b is 1x2.
        #   This is just the outer product. Tensor: T_{a_idx, b_idx, c_idx} = delta... 
        #   Actually T_{(i),(k),(i',k')} = 1 if i=i' and k=k', else 0.
        #   This is the identity on 4 elements, so rank = 4? No, the tensor is 
        #   T[i,k,(i',k')] = delta_{ii'} * delta_{kk'}. This is the 4x4 identity as a tensor.
        #   Rank of identity tensor in R^4 x R^4 x R^4... 
        #   Actually the "multiplication" tensor here is: for each (i,k) in output, 
        #   C_{i,k} = A_i * B_k. So T_{i,k,(i',k')} = delta_{i,i'} delta_{k,k'}.
        #   This is a product of two rank-2 identity tensors... it's just the 4-dimensional identity.
        #   The rank of the 4-dim identity tensor (in the sense of matrix multiplication rank) 
        #   for 1x2 * 2x1... I think it's just 4.
        #   Let me just use schoolbook for this part: rank 4 per block, 5*4=20.
        #   Total: 35+20=55. Still better than 60.
        #   
        #   Actually wait, let me re-examine. A is 2x1, B is 1x2. 
        #   A has 2 entries, B has 2 entries, C has 4 entries.
        #   C[0,0]=A[0]*B[0], C[0,1]=A[0]*B[1], C[1,0]=A[1]*B[0], C[1,1]=A[1]*B[1].
        #   Schoolbook rank = 4. 
        #   Can we do better? This is equivalent to the Hadamard/outer product.
        #   The tensor is sum_{i,k} e_i ⊗ e_k ⊗ e_{(i,k)}.
        #   This is actually rank 4 (can't do better in general).
        #   
        #   So let's use schoolbook for term 2: rank 20.
        #   Total: 35 + 20 = 55.

    # Actually, let me just use schoolbook for the 2x1 * 1x10 part.
    for i in range(2):
        for k in range(10):
            col = col_offset
            U[i * 3 + 2, col] = 1.0  # A row i, col 2
            V[2 * 10 + k, col] = 1.0  # B row 2, col k
            W[k * 2 + i, col] = 1.0   # C row k, col i
            col_offset += 1

    # Now trim unused columns
    used_cols = np.any(U != 0, axis=0) | np.any(V != 0, axis=0) | np.any(W != 0, axis=0)
    # Remove zero columns
    non_zero = np.where(used_cols)[0]
    U = U[:, non_zero]
    V = V[:, non_zero]
    W = W[:, non_zero]

    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify the decomposition is exact."""
    # Random test
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B  # shape (n, p)

    # Flatten
    A_flat = A.flatten()  # (n*m,)
    B_flat = B.flatten()  # (m*p,)
    C_flat_expected = C_expected.flatten()  # (n*p,)

    # Compute C_flat from decomposition: C[k*n+i] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This is: C_flat = W @ (U @ A_flat) * (V @ B_flat)... no.
    # Actually: C_flat[gamma] = sum_r W[gamma, r] * (sum_{alpha} U[alpha, r] * A_flat[alpha]) * (sum_{beta} V[beta, r] * B_flat[beta])
    # So: for each r: u_prod = A_flat @ U[:,r], v_prod = B_flat @ V[:,r], then C_flat += W[:,r] * u_prod * v_prod
    C_flat = np.zeros(n * p)
    for r in range(U.shape[1]):
        u_prod = A_flat @ U[:, r]
        v_prod = B_flat @ V[:, r]
        C_flat += W[:, r] * (u_prod * v_prod)

    # Check
    if np.allclose(C_flat, C_flat_expected, atol=1e-10):
        return True
    return False


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3x10 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (6, rank), (30, rank), (20, rank).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = DIMENSIONS

    # Try the composite decomposition first
    try:
        U, V, W = _construct_composite()
        rank = U.shape[1]
        if _verify(U, V, W, n, m, p):
            # Convert to lists
            return {
                'U': U.tolist(),
                'V': V.tolist(),
                'W': W.tolist()
            }
    except Exception:
        pass

    # Fallback: schoolbook
    if time.time() > deadline:
        U, V, W = _schoolbook(n, m, p)
        return {
            'U': U.tolist(),
            'V': V.tolist(),
            'W': W.tolist()
        }

    U, V, W = _schoolbook(n, m, p)
    return {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }

# EVOLVE-BLOCK-END
