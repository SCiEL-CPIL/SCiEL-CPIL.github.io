"""Matrix multiplication tensor decomposition for 2x3 * 3x9.

Strategy: The schoolbook rank is 54. We attempt to find a lower-rank
decomposition by exploiting the block structure (3 blocks of 2x3*3x3)
and using algebraic tensoring with the algebra of 3x3 matrices.

For 2x3 * 3x3, the schoolbook rank is 18. If we can find a rank-r
decomposition for 2x3*3x3, then tensoring with the 9-dim algebra of 3x3
matrices gives rank 9*r for 2x3*3x9.

We also try direct numerical search for low-rank decompositions.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]


def _build_schoolbook(n=2, m=3, p=9):
    """Build the schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _build_2x3x3_schoolbook():
    """Rank-18 schoolbook for 2x3 * 3x3."""
    n, m, p = 2, 3, 3
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    return U, V, W


def _algebra_tensoring(base_U, base_V, base_W, algebra_dim):
    """Tensor a base decomposition with an algebra of given dimension.
    
    For matrix multiplication over an algebra of dimension d,
    the rank is rank_base * d.
    
    The algebra here is the full 3x3 matrix algebra (dim 9).
    B is 3x9, split into 3 blocks of 3 columns. Each block is a 3x3 matrix.
    The multiplication C = A * B where A is 2x3, B is 3x9.
    
    View B as a 3x3 matrix of 3x3 blocks: B[k, block_b] is a 3x3 matrix.
    C[i, block_b, c] = sum_k A[i,k] * B[k, block_b, c]
    
    This is just 2x3 * 3x3 done independently for each block, so rank = 3*18 = 54.
    No improvement from full algebra.
    
    For upper triangular 3x3 (dim 6): only works if B blocks are upper triangular.
    """
    n, m = 2, 3
    p_base = 3  # base p
    p_full = 9
    rank_base = len(base_U[0])
    d = algebra_dim
    rank = rank_base * d
    
    # This is just the schoolbook for the full problem if d=9 and p_base=3
    # Actually, the tensoring with full matrix algebra gives rank = 18 * 9 = 162? No.
    # Let me reconsider.
    
    # The correct view: 2x3 * 3x9 = (2x3) * (3x3 of 3x3 matrices)
    # Over the algebra M_3(R) (dim 9):
    # A is 2x3 over R (scalars)
    # B is 3x1 over M_3(R)? No, B is 3x9 = 3 rows, 9 columns.
    # Split columns: B[:, 0:3], B[:, 3:6], B[:, 6:9] are three 3x3 matrices.
    # C[:, 0:3] = A * B[:, 0:3], etc.
    # Each is 2x3 * 3x3, rank 18. Total = 54.
    
    # Alternatively: view as 2x3 * 3x3 over the 3-dim algebra of "column blocks"
    # This doesn't help either.
    
    # The only way algebra helps is if we can find a rank < 18 decomposition
    # for 2x3 * 3x3, then use it 3 times (rank < 54).
    pass
    return None


def _try_strassen_2x3x3(deadline):
    """Try to find a rank < 18 decomposition for 2x3 * 3x3.
    
    Known: 2x2*2x2 has rank 7 (Strassen).
    2x3*3x3: not well-studied, but rank 18 is schoolbook.
    
    We can try: split 2x3 as [2x2 | 2x1] and 3x3 as [[3x2, 3x1]]
    This gets complicated. Let's try a direct numerical search.
    """
    n, m, p = 2, 3, 3
    # Lower bound: max(n*m, m*p, p*n) = max(6, 9, 6) = 9
    # Try ranks from 9 to 17
    
    for rank in range(9, 18):
        if time.time() > deadline:
            break
        
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(rank * 100 + attempt)
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Iterative refinement: alternate solving for one factor
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Solve for W: for each (i,j), sum_k sum_t U[i*m+k,t]*V[k*p+j,t]*W[j*n+i,t] = 1
                # For fixed (i,j): sum_t (sum_k U[i*m+k,t]*V[k*p+j,t]) * W[j*n+i,t] = 1
                # Let M[i,j,t] = sum_k U[i*m+k,t] * V[k*p+j,t]
                # Then for each (i,j): M[i,j,:] . W[j*n+i,:] = 1
                # This is underdetermined (1 eq, rank unknowns). Use least squares with regularization.
                
                M = np.zeros((n, p, rank))
                for i in range(n):
                    for j in range(p):
                        for k in range(m):
                            M[i, j] += U[i * m + k] * V[k * p + j]
                
                W_new = np.zeros((p * n, rank))
                for i in range(n):
                    for j in range(p):
                        # M[i,j,:] . W[j*n+i,:] = 1
                        # Least squares: minimize ||M[i,j,:] . x - 1||^2 + lambda*||x||^2
                        a = M[i, j]
                        # x = a^T (a a^T + lambda I)^{-1} * 1
                        # Since it's a single equation: x = a / (a.a + lambda)
                        aa = a @ a
                        lam = 1e-6
                        W_new[j * n + i] = a / (aa + lam)
                
                W = W_new
                
                # Solve for U similarly
                # For each (i,k): sum_j sum_t U[i*m+k,t]*V[k*p+j,t]*W[j*n+i,t] = 1
                # For fixed (i,k): sum_t U[i*m+k,t] * (sum_j V[k*p+j,t]*W[j*n+i,t]) = 1
                # Let N[i,k,t] = sum_j V[k*p+j,t] * W[j*n+i,t]
                # Then: N[i,k,:] . U[i*m+k,:] = 1
                N = np.zeros((n, m, rank))
                for i in range(n):
                    for k in range(m):
                        for j in range(p):
                            N[i, k] += V[k * p + j] * W[j * n + i]
                
                U_new = np.zeros((n * m, rank))
                for i in range(n):
                    for k in range(m):
                        a = N[i, k]
                        aa = a @ a
                        U_new[i * m + k] = a / (aa + lam)
                U = U_new
                
                # Solve for V
                # For each (k,j): sum_i sum_t U[i*m+k,t]*V[k*p+j,t]*W[j*n+i,t] = 1
                # For fixed (k,j): sum_t V[k*p+j,t] * (sum_i U[i*m+k,t]*W[j*n+i,t]) = 1
                # Let P[k,j,t] = sum_i U[i*m+k,t] * W[j*n+i,t]
                P = np.zeros((m, p, rank))
                for k in range(m):
                    for j in range(p):
                        for i in range(n):
                            P[k, j] += U[i * m + k] * W[j * n + i]
                
                V_new = np.zeros((m * p, rank))
                for k in range(m):
                    for j in range(p):
                        a = P[k, j]
                        aa = a @ a
                        V_new[k * p + j] = a / (aa + lam)
                V = V_new
                
                # Check error
                if it % 20 == 0:
                    err = 0.0
                    for i in range(n):
                        for k in range(m):
                            for j in range(p):
                                val = sum(U[i*m+k, t] * V[k*p+j, t] * W[j*n+i, t] for t in range(rank))
                                err += (val - 1.0) ** 2
                    if err < 1e-4:
                        # Try to round
                        U_r = np.round(U)
                        V_r = np.round(V)
                        W_r = np.round(W)
                        # Verify
                        err2 = 0.0
                        for i in range(n):
                            for k in range(m):
                                for j in range(p):
                                    val = sum(U_r[i*m+k, t] * V_r[k*p+j, t] * W_r[j*n+i, t] for t in range(rank))
                                    err2 += (val - 1.0) ** 2
                        if err2 < 1e-6:
                            return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None


def _verify(U, V, W, n=2, m=3, p=9):
    """Verify the decomposition."""
    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)
    C_expected = A @ B
    
    C_recon = np.zeros((n, p))
    rank = len(U[0])
    for t in range(rank):
        alpha = np.array([sum(U[i * m + k][t] * A[i, k] for k in range(m)) for i in range(n)])
        beta = np.array([sum(V[k * p + j][t] * B[k, j] for k in range(m)) for j in range(p)])
        for i in range(n):
            for j in range(p):
                C_recon[i, j] += W[j * n + i][t] * alpha[i] * beta[j]
    
    return np.allclose(C_recon, C_expected, atol=1e-9)


def _verify_tensor(U, V, W, n=2, m=3, p=9):
    """Verify tensor identity directly."""
    rank = len(U[0])
    for i in range(n):
        for k in range(m):
            for j in range(p):
                val = sum(U[i * m + k][t] * V[k * p + j][t] * W[j * n + i][t] for t in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _try_direct_search(n, m, p, deadline):
    """Direct search for low-rank decomposition of 2x3*3x9."""
    # Lower bound: max(6, 27, 18) = 27
    # Try a few ranks
    for rank in range(27, 54):
        if time.time() > deadline:
            break
        
        # Only try a few ranks to stay within time
        if rank not in [27, 30, 36, 42, 48, 50, 52, 53]:
            continue
        
        for attempt in range(2):
            if time.time() > deadline:
                break
            
            np.random.seed(rank * 7 + attempt)
            U = np.random.randn(n * m, rank) * 0.3
            V = np.random.randn(m * p, rank) * 0.3
            W = np.random.randn(p * n, rank) * 0.3
            
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Alternate solving
                lam = 1e-5
                
                # Solve W
                M = np.zeros((n, p, rank))
                for i in range(n):
                    for j in range(p):
                        for k in range(m):
                            M[i, j] += U[i * m + k] * V[k * p + j]
                W_new = np.zeros((p * n, rank))
                for i in range(n):
                    for j in range(p):
                        a = M[i, j]
                        aa = a @ a
                        W_new[j * n + i] = a / (aa + lam)
                W = W_new
                
                # Solve U
                N = np.zeros((n, m, rank))
                for i in range(n):
                    for k in range(m):
                        for j in range(p):
                            N[i, k] += V[k * p + j] * W[j * n + i]
                U_new = np.zeros((n * m, rank))
                for i in range(n):
                    for k in range(m):
                        a = N[i, k]
                        aa = a @ a
                        U_new[i * m + k] = a / (aa + lam)
                U = U_new
                
                # Solve V
                P = np.zeros((m, p, rank))
                for k in range(m):
                    for j in range(p):
                        for i in range(n):
                            P[k, j] += U[i * m + k] * W[j * n + i]
                V_new = np.zeros((m * p, rank))
                for k in range(m):
                    for j in range(p):
                        a = P[k, j]
                        aa = a @ a
                        V_new[k * p + j] = a / (aa + lam)
                V = V_new
                
                if it % 30 == 0:
                    err = 0.0
                    for i in range(n):
                        for k in range(m):
                            for j in range(p):
                                val = sum(U[i*m+k, t] * V[k*p+j, t] * W[j*n+i, t] for t in range(rank))
                                err += (val - 1.0) ** 2
                    if err < 1e-3:
                        # Try rounding to half-integers
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        err2 = 0.0
                        for i in range(n):
                            for k in range(m):
                                for j in range(p):
                                    val = sum(U_r[i*m+k, t] * V_r[k*p+j, t] * W_r[j*n+i, t] for t in range(rank))
                                    err2 += (val - 1.0) ** 2
                        if err2 < 1e-6:
                            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    Returns dict with 'U', 'V', 'W'.
    """
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 120
    
    # Start with schoolbook
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p  # 54
    
    # Try to find 2x3*3x3 decomposition with rank < 18
    if time.time() < deadline - 30:
        result_233 = _try_strassen_2x3x3(deadline - 30)
        if result_233 is not None:
            U_b, V_b, W_b = result_233
            r_b = len(U_b[0])
            # Tensor with 3 blocks: rank = 3 * r_b
            # Actually this just gives 3 separate 2x3*3x3 multiplications
            # Total rank = 3 * r_b
            if 3 * r_b < best_rank:
                # Build the full decomposition
                rank = 3 * r_b
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                for block in range(3):
                    for t in range(r_b):
                        col = block * r_b + t
                        for i in range(n):
                            for k in range(m):
                                U[i * m + k][col] = U_b[i * m + k][t]
                        for k in range(m):
                            for j in range(3):
                                V[k * p + block * 3 + j][col] = V_b[k * 3 + j][t]
                        for j in range(3):
                            for i in range(n):
                                W[(block * 3 + j) * n + i][col] = W_b[j * n + i][t]
                if _verify_tensor(U, V, W, n, m, p):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = rank
    
    # Try direct search
    if time.time() < deadline - 10:
        result = _try_direct_search(n, m, p, deadline - 10)
        if result is not None:
            r = len(result['U'][0])
            if r < best_rank:
                best = result
                best_rank = r
    
    return best
