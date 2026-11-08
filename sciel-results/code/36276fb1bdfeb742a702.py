import time
import sys

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

def _check_deadline():
    return (time.time() - _start_time) > (_DEADLINE - _SAFETY_MARGIN)

def _naive_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}

def _rank28_decomposition(n, m, p):
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4) - 4 terms
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen - 7 terms
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen - 7 terms
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomp(n, m, p, U, V, W):
    """Verify that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] equals the matmul tensor."""
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    # Build tensor: T[i*m+j, j'*p+k, k'*n+i'] should be delta
    # Actually: sum_r U[i,j,r] * V[j',k,r] * W[k',i',r] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # Wait, the tensor is: T_{(i,j),(j',k),(k',i')} = 1 if i=i', j=j', k=k' else 0
    # And we need: sum_r U[i,j,r] * V[j',k,r] * W[k',i',r] = T
    
    # Let's just check with random matrices
    import random
    random.seed(42)
    for trial in range(20):
        A = [[random.uniform(-1, 1) for _ in range(m)] for _ in range(n)]
        B = [[random.uniform(-1, 1) for _ in range(p)] for _ in range(m)]
        
        # True product
        C_true = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        # Decomposed product
        C_dec = [[0.0] * p for _ in range(n)]
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    u_val = U_arr[i * m + j, r]
                    if u_val != 0:
                        for j2 in range(m):
                            for k in range(p):
                                v_val = V_arr[j2 * p + k, r]
                                if v_val != 0:
                                    for i2 in range(n):
                                        w_val = W_arr[k * n + i2, r]
                                        if w_val != 0:
                                            C_dec[i][k] += u_val * v_val * w_val
        
        for i in range(n):
            for k in range(p):
                if abs(C_true[i][k] - C_dec[i][k]) > 1e-9:
                    return False
    return True

def construct():
    if _check_deadline():
        n, m, p = DIMENSIONS
        return _naive_decomposition(n, m, p)
    
    n, m, p = DIMENSIONS
    
    # Try rank 27 first
    # Check if rank-28 decomposition is valid
    decomp28 = _rank28_decomposition(n, m, p)
    
    if _check_deadline():
        return decomp28
    
    # Verify rank-28 is correct
    valid28 = _verify_decomp(n, m, p, decomp28["U"], decomp28["V"], decomp28["W"])
    
    if _check_deadline():
        if valid28:
            return decomp28
        return _naive_decomposition(n, m, p)
    
    if not valid28:
        # Fallback to naive
        return _naive_decomposition(n, m, p)
    
    # Try to find rank-27 by checking linear dependence
    # The 28 rank-1 tensors must be linearly independent for rank to be 28
    # If one is a linear combination of others, rank is 27
    import numpy as np
    
    U_arr = np.array(decomp28["U"], dtype=float)
    V_arr = np.array(decomp28["V"], dtype=float)
    W_arr = np.array(decomp28["W"], dtype=float)
    
    rank = 28
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Each rank-1 term is a tensor T_r with T_r[i,j,k'] = U[i,j,r]*V[j',k,r]*W[k',i',r]
    # Actually the tensor index is ((i,j),(j',k),(k',i')) but with constraints i=i', j=j', k=k'
    # The tensor lives in F^{nm} ⊗ F^{mp} ⊗ F^{pn}
    # Each term r gives a vector in this space: vec_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    
    # Stack all 28 vectors into a matrix and check rank
    vecs = []
    for r in range(rank):
        vec = np.kron(np.kron(U_arr[:, r], V_arr[:, r]), W_arr[:, r])
        vecs.append(vec)
    
    mat = np.column_stack(vecs)  # shape (nm*mp*pn, rank)
    
    # Check if columns are linearly independent
    # Use SVD
    try:
        sv = np.linalg.svd(mat, full_matrices=False)
        # Count singular values > 1e-9
        num_indep = np.sum(sv > 1e-9)
    except:
        num_indep = rank
    
    if _check_deadline():
        return decomp28
    
    if num_indep < rank:
        # There's linear dependence, we might be able to reduce rank
        # But for exact decomposition, we need to find which term to remove
        # and verify the remaining terms still span the tensor
        # For now, just report and return rank-28
        pass
    
    return decomp28
# EVOLVE-BLOCK-END
