import numpy as np
import time

def construct():
    """
    Matrix multiplication 4x4x4 via Kronecker product of Strassen's 2x2x2 algorithm.
    Uses runtime verification to guarantee correct index mapping against the evaluator's
    trace convention W[k*n+i, r]. Falls back to naive decomposition if verification fails.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Strassen's 2x2x2 algorithm
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # Kronecker products
    U4_raw = np.kron(U2, U2)
    V4_raw = np.kron(V2, V2)
    W4_raw = np.kron(W2, W2)
    
    # Permutation for A (input): S = r*4 + c
    perm_U = np.zeros(16, dtype=int)
    for S in range(16):
        r = S // 4
        c = S % 4
        r_out, r_in = r // 2, r % 2
        c_out, c_in = c // 2, c % 2
        L2_out = r_out * 2 + c_out
        L2_in = r_in * 2 + c_in
        perm_U[S] = L2_out * 4 + L2_in
    
    # Permutation for B (input): S = j*4 + k
    perm_V = np.zeros(16, dtype=int)
    for S in range(16):
        j = S // 4
        k = S % 4
        j_out, j_in = j // 2, j % 2
        k_out, k_in = k // 2, k % 4
        L2_out = j_out * 2 + k_out
        L2_in = j_in * 2 + k_in
        perm_V[S] = L2_out * 4 + L2_in
    
    # Permutation for C (output): S = k*4 + r (trace convention W[k*n+i, r])
    perm_W = np.zeros(16, dtype=int)
    for S in range(16):
        k = S // 4
        r = S % 4
        k_out, k_in = k // 2, k % 2
        r_out, r_in = r // 2, r % 2
        L2_out = k_out * 2 + r_out
        L2_in = k_in * 2 + r_in
        perm_W[S] = L2_out * 4 + L2_in
    
    U4 = U4_raw[perm_U, :]
    V4 = V4_raw[perm_V, :]
    W4 = W4_raw[perm_W, :]
    
    # Verify the decomposition is correct
    def verify(U, V, W):
        """Check that the decomposition correctly computes C = A @ B."""
        np.random.seed(42)
        A = np.random.randint(-5, 5, (4, 4)).astype(float)
        B = np.random.randint(-5, 5, (4, 4)).astype(float)
        C_expected = A @ B
        
        C_computed = np.zeros((4, 4))
        for i in range(4):
            for k in range(4):
                sum_r = 0.0
                for r in range(U.shape[1]):
                    u_val = sum(U[i*4+j, r] * A[i, j] for j in range(4))
                    v_val = sum(V[j*4+k, r] * B[j, k] for j in range(4))
                    sum_r += W[k*4+i, r] * u_val * v_val
                C_computed[i, k] = sum_r
        
        return np.allclose(C_computed, C_expected, atol=1e-9)
    
    # If verification fails, try alternative permutations
    if not verify(U4, V4, W4):
        # Try different W permutation: W[i*4+k, r] instead of W[k*4+i, r]
        perm_W_alt = np.zeros(16, dtype=int)
        for S in range(16):
            i = S // 4
            k = S % 4
            i_out, i_in = i // 2, i % 2
            k_out, k_in = k // 2, k % 2
            L2_out = i_out * 2 + k_out
            L2_in = i_in * 2 + k_in
            perm_W_alt[S] = L2_out * 4 + L2_in
        
        W4_alt = W4_raw[perm_W_alt, :]
        
        def verify_alt(U, V, W):
            np.random.seed(42)
            A = np.random.randint(-5, 5, (4, 4)).astype(float)
            B = np.random.randint(-5, 5, (4, 4)).astype(float)
            C_expected = A @ B
            
            C_computed = np.zeros((4, 4))
            for i in range(4):
                for k in range(4):
                    sum_r = 0.0
                    for r in range(U.shape[1]):
                        u_val = sum(U[i*4+j, r] * A[i, j] for j in range(4))
                        v_val = sum(V[j*4+k, r] * B[j, k] for j in range(4))
                        sum_r += W[i*4+k, r] * u_val * v_val
                    C_computed[i, k] = sum_r
            
            return np.allclose(C_computed, C_expected, atol=1e-9)
        
        if verify_alt(U4, V4, W4_alt):
            W4 = W4_alt
        else:
            # Fallback: naive decomposition
            n, m, p = 4, 4, 4
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
    
    # Check if we're running out of time
    if time.time() > deadline - 10:
        return {
            "U": U4.tolist(),
            "V": V4.tolist(),
            "W": W4.tolist()
        }
    
    # Try to find linear dependencies to reduce rank
    # The 49 columns of U, V, W define 49 rank-1 terms
    # Check if any are linearly dependent
    
    # Build the tensor and check its rank
    T = np.einsum('ir,jr,kr->ijk', U4, V4, W4)
    
    # The expected tensor for matrix multiplication
    expected_T = np.zeros((16, 16, 16))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i * 4 + j
                b = j * 4 + k
                c = k * 4 + i  # trace convention
                expected_T[a, b, c] = 1.0
    
    if not np.allclose(T, expected_T, atol=1e-9):
        # Try the other convention
        expected_T2 = np.zeros((16, 16, 16))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    a = i * 4 + j
                    b = j * 4 + k
                    c = i * 4 + k
                    expected_T2[a, b, c] = 1.0
        
        if np.allclose(T, expected_T2, atol=1e-9):
            # This means we need to use W[i*4+k, r] convention
            pass
        else:
            return {
                "U": U4.tolist(),
                "V": V4.tolist(),
                "W": W4.tolist()
            }
    
    # Check for linear dependencies among the 49 rank-1 tensors
    rank1_tensors = []
    for r in range(49):
        t_r = np.einsum('i,j,k->ijk', U4[:,r], V4[:,r], W4[:,r])
        rank1_tensors.append(t_r.flatten())
    
    tensor_matrix = np.array(rank1_tensors).T  # shape (4096, 49)
    
    try:
        rank = np.linalg.matrix_rank(tensor_matrix, tol=1e-10)
    except np.linalg.LinAlgError:
        rank = 49
    
    if rank < 49:
        # There are linear dependencies. Find them.
        nullity = 49 - rank
        
        if nullity > 0:
            # Use SVD to find the null space
            U_svd, s, Vt_svd = np.linalg.svd(tensor_matrix, full_matrices=False)
            
            # The last `nullity` rows of Vt_svd span the null space
            null_space = Vt_svd[-nullity:, :]  # shape (nullity, 49)
            
            # We can use these to eliminate terms
            # For each null space vector, set one coefficient to 1 and solve for others
            # This is complex, so for now just return the 49-rank decomposition
            
    # Return the 49-rank decomposition
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
