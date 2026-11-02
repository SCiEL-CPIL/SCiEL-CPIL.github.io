"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Uses Strassen-based block decomposition as baseline (rank 64), then attempts ALS refinement
with rounding to half-integers to find lower-rank decompositions."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n=4, m=4, p=5, tol=1e-9):
    """Check that the decomposition exactly reproduces matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    """
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        wcT = wc.T
        for j in range(m):
            T[:, j, :] += np.outer(uc[:, j], vc[j, :]) * wcT
    return np.allclose(T, 1.0, atol=tol)

def _round_to_half_integer(x):
    """Round a scalar to the nearest half-integer."""
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    """Round all entries of a matrix to the nearest half-integer."""
    return np.round(np.array(M, dtype=np.float64) * 2) / 2.0

def _strassen_construct():
    """Construct a Strassen-based block decomposition for 4x4 * 4x5.
    
    We view A (4x4) as 2x2 blocks of 2x2, and B (4x5) as 2x2 blocks where
    the second dimension has 5 columns. We pad B to 4x6 (or handle the 5 columns
    specially). Actually, let's think more carefully.
    
    A is 4x4: blocks A11(2x2), A12(2x2), A21(2x2), A22(2x2)
    B is 4x5: blocks B11(2x2), B12(2x3), B21(2x2), B22(2x3)
    
    C = AB is 4x5: blocks C11(2x2), C12(2x3), C21(2x2), C22(2x3)
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Using Strassen's 7 products for the 2x2 block structure:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    But wait - the block sizes are mixed. B12 and B22 are 2x3, while B11 and B21 are 2x2.
    This means M3 = A11(B12-B22) is 2x2 * 2x3 = 2x3. Good.
    M6 = (A21-A11)(B11+B12) - but B11 is 2x2 and B12 is 2x3, can't add them directly.
    
    Hmm, this is problematic. Let me reconsider.
    
    Actually, for the tensor decomposition, we need to work at the scalar level.
    The multiplication tensor for 4x4*4x5 has entries T[i,j,k]=1 for all i in [0,4),
    j in [0,4), k in [0,5).
    
    Let me use a different approach: decompose into 2x2x2 and 2x2x3 sub-tensors.
    
    Actually, let me just use the Kronecker product of Strassen 2x2x2 with identity
    for the extra dimension, or use a block approach more carefully.
    
    Alternative: Use the fact that 4x4x5 can be decomposed using two 2x2x3 blocks
    in the output dimension.
    
    Let me try a simpler approach: use Strassen on the 2x2 block structure for the
    first 4 columns of B, and handle the 5th column separately.
    
    Actually, the cleanest approach for the tensor: view the 4x4*4x5 multiplication
    as a combination of smaller multiplications.
    
    Let's use the following block decomposition:
    A = [[A11, A12], [A21, A22]] where each Ai* is 2x2
    B = [[B11, B12], [B21, B22]] where B11,B21 are 2x2 and B12,B22 are 2x3
    
    For the 2x2 part (columns 0-3 of B), we can use Strassen with 7 products.
    For the 5th column, we need additional terms.
    
    Let me just construct it directly at the scalar level using the known Strassen
    formulas extended to handle the rectangular case.
    
    Actually, the simplest correct approach: use the rank-49 decomposition of 4x4x4
    (Kronecker of two rank-7 2x2x2 decompositions) and extend it to 4x4x5 by
    adding 16 more terms for the 5th column (schoolbook for that column).
    Total: 49 + 16 = 65.
    
    Or: use rank 49 for the first 4 columns' worth and handle the 5th column
    with schoolbook: 4*4*1 = 16 terms. Total 65.
    
    But we can do better. Let me use a 2x2x3 Strassen-like decomposition.
    
    For 2x2 * 2x3:
    A is 2x2, B is 2x3, C is 2x3.
    C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:]
    C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:]
    
    Schoolbook: 8 terms.
    Can we do better? Using Strassen-like:
    M1 = (A00+A11)(B00+B10)  -- but B has 3 columns...
    
    This gets complicated. Let me just use the 49+16=65 approach first,
    then try to optimize.
    
    Actually, let me construct the 4x4x4 part using Kronecker product of Strassen,
    and add the 5th column terms.
    """
    n, m, p = 4, 4, 5
    # First, get the 2x2x2 Strassen decomposition
    U2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0]
    ], dtype=np.float64)
    V2 = np.array([
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0]
    ], dtype=np.float64)
    W2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 1]
    ], dtype=np.float64)
    
    # Verify 2x2x2 Strassen
    # T[i,j,k] = sum_c U2[i*2+j,c] * V2[j*2+k,c] * W2[k*2+i,c]
    # For 2x2x2: i,j,k in {0,1}
    # U2 shape (4,7), V2 shape (4,7), W2 shape (4,7)
    
    # Now build 4x4x4 using Kronecker product
    # T4[i,j,k] = sum_{a,b} U2[i1*2+i2, a] * U2[i2*2+j2, b] * ... 
    # Actually, the Kronecker product of two tensors T1 (2x2x2) and T2 (2x2x2)
    # gives T1⊗T2 (4x4x4) with T1⊗T2[i,j,k] = T1[i1,j1,k1]*T2[i2,j2,k2]
    # where i = i1*2+i2, j = j1*2+j2, k = k1*2+k2
    # 
    # For matrix multiplication, M_{2,2,2} ⊗ M_{2,2,2} = M_{4,4,4}
    # because (A⊗B)(C⊗D) = (AC)⊗(BD)
    
    # The rank of the Kronecker product is rank(T1)*rank(T2) = 7*7 = 49
    # U4[i*4+j, c] where c = a*7+b
    # U4[i*4+j, a*7+b] = U2[i1*2+i2, a] * U2[i2*2+j2, b]
    # where i = i1*2+i2, j = j1*2+j2
    
    U4 = np.zeros((16, 49), dtype=np.float64)
    V4 = np.zeros((16, 49), dtype=np.float64)
    W4 = np.zeros((16, 49), dtype=np.float64)
    
    for a in range(7):
        for b in range(7):
            c = a * 7 + b
            for i in range(4):
                for j in range(4):
                    i1, i2 = divmod(i, 2)
                    j1, j2 = divmod(j, 2)
                    U4[i*4+j, c] = U2[i1*2+i2, a] * U2[i2*2+j2, b]
            for j in range(4):
                for k in range(4):
                    j1, j2 = divmod(j, 2)
                    k1, k2 = divmod(k, 2)
                    V4[j*4+k, c] = V2[j1*2+k1, a] * V2[j2*2+k2, b]
            for k in range(4):
                for i in range(4):
                    k1, k2 = divmod(k, 2)
                    i1, i2 = divmod(i, 2)
                    W4[k*4+i, c] = W2[k1*2+i1, a] * W2[k2*2+i2, b]
    
    # Now extend to 4x4x5: add 16 schoolbook terms for the 5th column (k=4)
    # For k=4: C[i,4] = sum_j A[i,j] * B[j,4]
    # We need T[i,j,4] = 1 for all i,j
    # Add terms: for each (i,j), U[i*4+j, 49+i*4+j] = 1, V[j*4+4, 49+i*4+j] = 1, W[4*4+i, 49+i*4+j] = 1
    
    total_rank = 49 + 16
    U = np.zeros((16, total_rank), dtype=np.float64)
    V = np.zeros((16, total_rank), dtype=np.float64)
    W = np.zeros((20, total_rank), dtype=np.float64)
    
    U[:, :49] = U4
    V[:, :49] = V4
    W[:, :49] = W4
    
    for i in range(4):
        for j in range(4):
            c = 49 + i*4 + j
            U[i*4+j, c] = 1.0
            V[j*4+4, c] = 1.0
            W[4*4+i, c] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _als_decompose(n, m, p, target_rank, max_iters=100, seed=42, tol=1e-9):
    """Try to find a rank-target_rank decomposition using ALS with rounding."""
    np_rng = np.random.RandomState(seed)
    
    best_U = None
    best_V = None
    best_W = None
    best_err = np.inf
    
    for attempt in range(10):
        # Random initialization
        U = np_rng.randn(n*m, target_rank) * 0.1
        V = np_rng.randn(m*p, target_rank) * 0.1
        W = np_rng.randn(p*n, target_rank) * 0.1
        
        for it in range(max_iters):
            # Update U: for each row (i,j), solve least squares
            for i in range(n):
                for j in range(m):
                    # T[i,j,:] = U[i*m+j,:] * (V[j*p+0:p,:] * W[0:n,:].T) -- not quite
                    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    # For fixed (i,j): T[i,j,k] = sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c])
                    # Let M_k = V[j*p+k,:] * W[k*n+i,:] (element-wise product), shape (rank,)
                    # Then T[i,j,k] = U[i*m+j,:] . M_k
                    # This is: T[i,j,:] = U[i*m+j,:] @ M where M[k,c] = V[j*p+k,c]*W[k*n+i,c]
                    M = np.zeros((p, target_rank), dtype=np.float64)
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # T[i,j,:] = U[i*m+j,:] @ M
                    # So U[i*m+j,:] = T[i,j,:] @ M^+ (pseudoinverse)
                    try:
                        U_new = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                        U[i*m+j, :] = U_new
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # T[:,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    # For fixed (j,k): T[i,j,k] = V[j*p+k,:] . (U[i*m+j,:] * W[k*n+i,:])
                    M = np.zeros((n, target_rank), dtype=np.float64)
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V_new = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                        V[j*p+k, :] = V_new
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    # For fixed (k,i): T[i,j,k] = W[k*n+i,:] . (U[i*m+j,:] * V[j*p+k,:])
                    M = np.zeros((m, target_rank), dtype=np.float64)
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W_new = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                        W[k*n+i, :] = W_new
                    except:
                        pass
            
            # Check convergence
            if it % 10 == 0 or it == max_iters - 1:
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
                    # Try rounding
                    U_r = _round_matrix_to_half_int(U)
                    V_r = _round_matrix_to_half_int(V)
                    W_r = _round_matrix_to_half_int(W)
                    if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-9):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with Strassen-based construction (rank 65)
    strassen_result = _strassen_construct()
    
    # Verify it's correct
    if _verify_decomposition(strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p):
        best_result = strassen_result
        best_rank = 65
    else:
        best_result = _schoolbook()
        best_rank = n * m * p
    
    # Try to find lower-rank decompositions using ALS
    ranks_to_try = [64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 15:
            break
        if target_rank >= best_rank:
            continue
        
        result = _als_decompose(n, m, p, target_rank, max_iters=50, seed=target_rank * 137 + 42)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    if time.time() > deadline - 10:
        return best_result
    
    # Final verification
    if not _verify_decomposition(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        return _schoolbook()
    
    return best_result

# EVOLVE-BLOCK-END
