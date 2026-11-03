"""
Matrix multiplication 3x4 by 4x7 with rank-minimal tensor decomposition.
Uses block decomposition with Strassen and optimized sub-block decompositions.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Uses block decomposition with optimized sub-block decompositions,
    then applies tensor peeling to reduce rank.
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 4, 7

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        """Add a single rank-1 tensor term to the decomposition."""
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)

        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val

        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def strassen_222(i0, j0, k0):
        """
        Strassen decomposition for 2x2 * 2x2 block starting at (i0, j0, k0).
        Adds 7 rank-1 terms.
        """
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        # M1: (A11+A22)*(B11+B22)
        add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0, k1*n+i1], [1, 1])

        # M2: (A21+A22)*B11
        add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0], [1],
                       [k0*n+i1, k1*n+i1], [1, -1])

        # M3: A11*(B12-B22)
        add_rank1_term([i0*m+j0], [1],
                       [j0*p+k1, j1*p+k1], [1, -1],
                       [k1*n+i0, k1*n+i1], [1, 1])

        # M4: A22*(B21-B11)
        add_rank1_term([i1*m+j1], [1],
                       [j1*p+k0, j0*p+k0], [1, -1],
                       [k0*n+i0, k0*n+i1], [1, 1])

        # M5: (A11+A12)*B22
        add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                       [j1*p+k1], [1],
                       [k0*n+i0, k1*n+i0], [-1, 1])

        # M6: (A21-A11)*(B11+B12)
        add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                       [j0*p+k0, j0*p+k1], [1, 1],
                       [k1*n+i1], [1])

        # M7: (A12-A22)*(B21+B22)
        add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                       [j1*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0], [1])

    def m227_decomp(i0, j0, k0):
        """
        Decomposition for 2x2 * 2x7 block.
        Split p=7 as 2+2+2+1: Strassen + Strassen + Strassen + schoolbook = 7+7+7+4 = 25
        """
        strassen_222(i0, j0, k0)
        strassen_222(i0, j0, k0 + 2)
        strassen_222(i0, j0, k0 + 4)
        # Schoolbook for k = k0+6
        k6 = k0 + 6
        for i in range(2):
            for j in range(2):
                ii = i0 + i
                jj = j0 + j
                add_rank1_term([ii*m+jj], [1.0],
                               [jj*p+k6], [1.0],
                               [k6*n+ii], [1.0])

    def m127_decomp(i0, j0, k0):
        """
        Decomposition for 1x2 * 2x7 block.
        Schoolbook rank = 2*7 = 14.
        """
        j1 = j0 + 1
        for k in range(7):
            kk = k0 + k
            add_rank1_term([i0*m+j0], [1.0],
                           [j0*p+kk], [1.0],
                           [kk*n+i0], [1.0])
            add_rank1_term([i0*m+j1], [1.0],
                           [j1*p+kk], [1.0],
                           [kk*n+i0], [1.0])

    def m327_decomp(i0, j0, k0):
        """
        Decomposition for 3x2 * 2x7 block.
        Split n=3 as 2+1:
          M_{2,2,7}: 25
          M_{1,2,7}: 14
        Total: 39
        """
        m227_decomp(i0, j0, k0)
        m127_decomp(i0 + 2, j0, k0)

    def m347_decomp(i0, j0, k0):
        """
        Decomposition for 3x4 * 4x7 block.
        Split m=4 as 2+2:
          M_{3,2,7}: 39
          M_{3,2,7}: 39
        Total: 78
        """
        m327_decomp(i0, j0, k0)
        m327_decomp(i0, j0 + 2, k0)

    # Build the full structured decomposition
    if time.time() - start_time < deadline:
        m347_decomp(0, 0, 0)

    initial_rank = len(U_vecs)

    # Convert to numpy arrays
    U_init = np.array(U_vecs).T  # shape (n*m, initial_rank)
    V_init = np.array(V_vecs).T  # shape (m*p, initial_rank)
    W_init = np.array(W_vecs).T  # shape (p*n, initial_rank)

    # Form the full tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = sum_r U_init[a, r] * V_init[b, r] * W_init[c, r]
    # But we only need the matmul tensor: T[i*m+j, j*p+k, k*n+i]
    # Actually, let's form the full tensor and then extract the relevant part.
    
    # More efficient: directly compute the matmul tensor
    # T_matmul[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The tensor we need to decompose has entries:
    #   T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some valid i,j,k
    #                0 otherwise
    
    # Build the target tensor
    dim_a = n * m  # 12
    dim_b = m * p  # 28
    dim_c = p * n  # 21
    
    T = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Now apply iterative SVD peeling to find minimal rank decomposition
    # This is the "subspace method" - project onto the actual tensor
    U_out = []
    V_out = []
    W_out = []
    
    T_residual = T.copy()
    max_iter = 80  # upper bound on rank
    
    for r in range(max_iter):
        if time.time() - start_time > deadline - 10.0:
            break
            
        # Check if residual is (nearly) zero
        if np.max(np.abs(T_residual)) < 1e-10:
            break
        
        # Find the best rank-1 approximation via SVD
        # Reshape T_residual to (dim_a, dim_b*dim_c) and do SVD
        T_mat = T_residual.reshape(dim_a, dim_b * dim_c)
        try:
            U_svd, s_vals, Vt_svd = np.linalg.svd(T_mat, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        
        if len(s_vals) == 0 or s_vals[0] < 1e-12:
            break
        
        # Best rank-1: u = U_svd[:,0], v = Vt_svd[0,:].reshape(dim_b, dim_c)
        u_vec = U_svd[:, 0]
        v_c_mat = Vt_svd[0, :].reshape(dim_b, dim_c)
        
        # Normalize: scale so that the rank-1 term matches
        # We want u * v_mat * w_mat to approximate T_residual
        # Actually for tensor: term = np.einsum('a,bc->abc', u_vec, v_c_mat)
        # But we need it as outer product of 3 vectors for the decomposition format
        # The SVD gives us a rank-1 in the (a, bc) unfolding.
        # We need to further decompose v_c_mat into v_vec and w_vec.
        
        # SVD of v_c_mat to get its best rank-1
        try:
            U2, s2, Vt2 = np.linalg.svd(v_c_mat, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        
        if len(s2) == 0 or s2[0] < 1e-12:
            break
        
        v_vec = U2[:, 0]
        w_vec = Vt2[0, :]
        
        # Scale: the term is u_vec * v_vec * w_vec
        # The coefficient from SVD chain: s_vals[0] * s2[0]
        # Actually, let's just compute the residual subtraction
        term = np.einsum('a,b,c->abc', u_vec, v_vec, w_vec)
        
        # Find optimal scaling
        # Minimize ||T_residual - alpha * term||_F
        # alpha = <T_residual, term> / <term, term>
        numerator = np.sum(T_residual * term)
        denominator = np.sum(term * term)
        
        if abs(denominator) < 1e-15:
            break
        
        alpha = numerator / denominator
        T_residual -= alpha * term
        
        # Store the scaled vectors
        U_out.append(alpha * u_vec)
        V_out.append(v_vec)
        W_out.append(w_vec)
        
        # Check convergence
        if np.max(np.abs(T_residual)) < 1e-10:
            break
    
    final_rank = len(U_out)
    
    # Verify correctness
    if final_rank > 0 and time.time() - start_time < deadline - 5.0:
        U_arr = np.array(U_out).T
        V_arr = np.array(V_out).T
        W_arr = np.array(W_out).T
        
        # Verify: for random A, B, check A@B = sum_r (A*U_r) * (B*V_r) * W_r
        # Actually verify the tensor directly
        T_reconstructed = np.zeros((dim_a, dim_b, dim_c))
        for r in range(final_rank):
            T_reconstructed += np.einsum('a,b,c->abc', U_out[r], V_out[r], W_out[r])
        
        error = np.max(np.abs(T_reconstructed - T))
        
        if error > 1e-8:
            # Fallback to the structured decomposition
            U_arr = U_init
            V_arr = V_init
            W_arr = W_init
            final_rank = initial_rank
        else:
            U_arr = np.array(U_out).T
            V_arr = np.array(V_out).T
            W_arr = np.array(W_out).T
    else:
        U_arr = U_init
        V_arr = V_init
        W_arr = W_init
        final_rank = initial_rank

    # If peeling didn't help or failed, use the structured decomposition
    if final_rank >= initial_rank:
        U_arr = U_init
        V_arr = V_init
        W_arr = W_init
        final_rank = initial_rank

    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
# EVOLVE-BLOCK-END
