"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np


def _verify(U, V, W, n=3, m=3, p=6):
    """Verify that the factorization exactly decomposes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    
    # Check on-diagonal entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check off-diagonal entries
    for a in range(n * m):
        i_a = a // m
        j_a = a % m
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
    
    return True


def _schoolbook(n=3, m=3, p=6):
    """Return the schoolbook decomposition of rank n*m*p."""
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


def _structured_rank36(n=3, m=3, p=6):
    """
    Build a rank-36 decomposition by exploiting the column-pairing structure.
    
    Key insight: For C = A * B where A is 3x3 and B is 3x6,
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can pair columns of B: (k=0,k=3), (k=1,k=4), (k=2,k=5)
    For each pair, define B'[j, pair] = B[j, 2*pair] + B[j, 2*pair+1]
    
    Then A * B' gives us the sum of paired columns.
    We also need the difference: B'[j, pair] = B[j, 2*pair] - B[j, 2*pair+1]
    
    This gives us a way to compute 6 columns using 3 "virtual" 3x3 multiplications
    with combined U factors, reducing rank from 54 to 36.
    
    More precisely:
    For each i (row of C) and each j (col of A / row of B):
      We need A[i,j] * B[j,0:6]
    
    Group the 6 columns into 3 pairs: (0,3), (1,4), (2,5)
    For pair g, define:
      S_g[j] = B[j, 2g] + B[j, 2g+3]  (sum)
      D_g[j] = B[j, 2g] - B[j, 2g+3]  (difference)
    
    Then:
      C[i, 2g]     = 0.5 * sum_j A[i,j] * (S_g[j] + D_g[j])
      C[i, 2g+3]   = 0.5 * sum_j A[i,j] * (S_g[j] - D_g[j])
    
    So for each (i, j, g) triple, we have two rank-1 terms:
      Term 1: A[i,j] * B[j, 2g]  -> contributes to C[i, 2g]
      Term 2: A[i,j] * B[j, 2g+3] -> contributes to C[i, 2g+3]
    
    That's still 54 terms. But we can combine:
      For fixed (i, g), define:
        U_i_g = A[i, :]  (the i-th row of A)
        V_g = [B[:, 2g], B[:, 2g+3]]  (the g-th column pair of B)
        W_i_g = [C[i, 2g], C[i, 2g+3]]  (the g-th column pair of C)
    
    This doesn't directly reduce rank. Let me think differently.
    
    Actually, the key is that for fixed i, C[i,:] = A[i,:] * B
    This is a 1x3 times 3x6 = 1x6 result.
    We can write this as:
      C[i, 2g]   = sum_j A[i,j] * B[j, 2g]
      C[i, 2g+3] = sum_j A[i,j] * B[j, 2g+3]
    
    For each i and g, we have 6 terms (3 for each column).
    Total: 3 * 3 * 6 = 54 terms (schoolbook).
    
    To reduce rank, we need to share computation. One approach:
    For fixed i, compute S[i, g] = sum_j A[i,j] * B[j, 2g] and T[i, g] = sum_j A[i,j] * B[j, 2g+3]
    These are just (A*B)[i, 2g] and (A*B)[i, 2g+3].
    
    The rank-36 approach: use the fact that the tensor has a product structure.
    For each (i, j), the contribution to C is A[i,j] * B[j, :]
    This is a rank-6 term (since B[j,:] has 6 elements).
    There are 9 such (i,j) pairs, giving rank 54.
    
    To get rank 36, we need to find 36 rank-1 terms that sum to the tensor.
    
    Alternative: use the structure that C = A * B can be written as
    C = sum_{j} A[:, j] * B[j, :]
    where A[:, j] is a 3x1 column and B[j, :] is a 1x6 row.
    Each outer product A[:, j] * B[j, :] is a 3x6 matrix of rank 1.
    The tensor rank of the sum of 9 such matrices is at most 9 * 6 = 54.
    
    But we can do better by noting that the 9 outer products share structure.
    """
    nm, mp, pn = n * m, m * p, p * n
    r = 36  # target rank
    
    # Build the decomposition using a structured approach
    # For each (i, j) pair, we have 6 rank-1 terms (one for each k)
    # Total: 9 * 6 = 54. We need to reduce to 36.
    
    # Idea: pair up the k indices. For each (i, j), combine k and k+3:
    #   Term A: A[i,j] * B[j, k] * (1 if C col = k else 0)
    #   Term B: A[i,j] * B[j, k+3] * (1 if C col = k+3 else 0)
    
    # These can be written as:
    #   A[i,j] * (B[j,k] * e_k + B[j,k+3] * e_{k+3})
    
    # This doesn't directly help. Let me try a different approach.
    
    # Use ALS to find a rank-36 decomposition with a structured initialization.
    # Initialize with the schoolbook decomposition and then project to rank 36.
    
    # First, build the full tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Reshape to matrix for SVD
    T_mat = T.reshape(nm, mp * pn)
    U_svd, S_vals, Vt_svd = np.linalg.svd(T_mat, full_matrices=False)
    
    # Keep top 36 components
    U_svd = U_svd[:, :r]
    S_vals = S_vals[:r]
    Vt_svd = Vt_svd[:r, :]
    
    # This gives a rank-36 approximation, but not exact.
    # We need an exact decomposition, so this won't work directly.
    
    # Instead, let's use a different strategy: exploit the block structure.
    # The tensor T has a specific structure: T[a,b,c] = 1 iff 
    #   a = i*m+j, b = j*p+k, c = k*n+i for some (i,j,k)
    
    # For fixed i, the sub-tensor T_i is a 3x6x18 tensor (over j, k, c)
    # where T_i[j, k, c] = 1 iff c = k*n + i
    
    # This sub-tensor has rank 3*6 = 18 (for each (j,k) pair, one c value).
    # But we can write it as a sum of 6 rank-3 terms (one for each k):
    #   For fixed k, T_i[:, k, :] has nonzero entries at j=0,1,2 and c=k*n+i
    #   This is a 3x18 matrix of rank 3 (one column is all 1s).
    
    # So for each i, we need rank 18, giving total rank 54.
    
    # To reduce rank, we need to share across i values.
    # For fixed (j, k), T[:, j*p+k, k*n+i] = 1 for all i.
    # This is a 9x18 tensor (over a and c) where a = i*m+j and c = k*n+i.
    # For fixed j,k, this is a 3x18 matrix (over i) where entry (i, c) = 1 iff c = k*n+i.
    # This has rank 3.
    
    # So for each (j,k), we have rank 3, giving total rank 3*6*3 = 54.
    
    # The rank is inherently 54 for this structure unless we find a clever combination.
    
    # Let's just return None and rely on ALS to find a better decomposition.
    return None


def _try_als_decomposition(n=3, m=3, p=6, deadline=None):
    """
    Try to find a low-rank decomposition using ALS with various initializations.
    """
    if deadline is None:
        deadline = time.time() + 100
    
    nm, mp, pn = n * m, m * p, p * n
    
    best_result = None
    best_rank = n * m * p
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # But the actual tensor rank is likely higher.
    # For 3x3 * 3x6, the rank is at least 18 (from dimension constraints).
    # Let's try ranks from 18 upward.
    
    for r in range(18, 40):
        if time.time() > deadline - 10:
            break
        
        n_restarts = max(2, int((deadline - time.time()) / 15))
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Different initializations
            if restart == 0:
                # Sparse random initialization
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                for col in range(r):
                    # Assign random nonzero entries
                    n_u = np.random.randint(1, 4)
                    n_v = np.random.randint(1, 4)
                    n_w = np.random.randint(1, 4)
                    u_idx = np.random.choice(nm, n_u, replace=False)
                    v_idx = np.random.choice(mp, n_v, replace=False)
                    w_idx = np.random.choice(pn, n_w, replace=False)
                    U[u_idx, col] = np.random.choice([-1, 1], n_u)
                    V[v_idx, col] = np.random.choice([-1, 1], n_v)
                    W[w_idx, col] = np.random.choice([-1, 1], n_w)
            elif restart == 1:
                # Dense random initialization
                U = np.random.randn(nm, r) * 0.5
                V = np.random.randn(mp, r) * 0.5
                W = np.random.randn(pn, r) * 0.5
            else:
                # Mixed initialization
                U = np.random.randn(nm, r) * 0.3
                V = np.random.randn(mp, r) * 0.3
                W = np.random.randn(pn, r) * 0.3
                # Add some structure: make U depend on i, V on j, W on k
                for a in range(nm):
                    i_a = a // m
                    U[a, :] *= (1.0 + 0.1 * i_a)
                for b in range(mp):
                    j_b = b // p
                    V[b, :] *= (1.0 + 0.1 * j_b)
                for c in range(pn):
                    k_c = c // n
                    W[c, :] *= (1.0 + 0.1 * k_c)
            
            # ALS iterations
            max_iter = 80
            prev_err = float('inf')
            for it in range(max_iter):
                if time.time() > deadline - 3:
                    break
                
                # Update U
                new_U = np.zeros_like(U)
                for a in range(nm):
                    rows = []
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if i * m + j == a:
                                    b_idx = j * p + k
                                    c_idx = k * n + i
                                    rows.append(V[b_idx, :] * W[c_idx, :])
                    if rows:
                        M = np.array(rows)
                        ones = np.ones(M.shape[0])
                        try:
                            new_U[a, :] = np.linalg.lstsq(M, ones, rcond=None)[0]
                        except:
                            new_U[a, :] = U[a, :]
                U = new_U
                
                # Update V
                new_V = np.zeros_like(V)
                for b in range(mp):
                    rows = []
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if j * p + k == b:
                                    a_idx = i * m + j
                                    c_idx = k * n + i
                                    rows.append(U[a_idx, :] * W[c_idx, :])
                    if rows:
                        M = np.array(rows)
                        ones = np.ones(M.shape[0])
                        try:
                            new_V[b, :] = np.linalg.lstsq(M, ones, rcond=None)[0]
                        except:
                            new_V[b, :] = V[b, :]
                V = new_V
                
                # Update W
                new_W = np.zeros_like(W)
                for c in range(pn):
                    rows = []
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if k * n + i == c:
                                    a_idx = i * m + j
                                    b_idx = j * p + k
                                    rows.append(U[a_idx, :] * V[b_idx, :])
                    if rows:
                        M = np.array(rows)
                        ones = np.ones(M.shape[0])
                        try:
                            new_W[c, :] = np.linalg.lstsq(M, ones, rcond=None)[0]
                        except:
                            new_W[c, :] = W[c, :]
                W = new_W
                
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            err += (val - 1.0) ** 2
                
                # Check off-support (sampled for speed)
                if it % 10 == 9:
                    for _ in range(100):
                        a = np.random.randint(0, nm)
                        b = np.random.randint(0, mp)
                        c = np.random.randint(0, pn)
                        i_a, j_a = a // m, a % m
                        j_b, k_b = b // p, b % p
                        k_c, i_c = c // n, c % n
                        if not (j_a == j_b and k_b == k_c and i_c == i_a):
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            err += val ** 2
                
                if err < 1e-6:
                    # Verify exactly
                    exact_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i * m + j
                                b = j * p + k
                                c = k * n + i
                                val = np.dot(U[a, :], V[b, :] * W[c, :])
                                exact_err += (val - 1.0) ** 2
                    for a in range(nm):
                        i_a, j_a = a // m, a % m
                        for b in range(mp):
                            j_b, k_b = b // p, b % p
                            for c in range(pn):
                                k_c, i_c = c // n, c % n
                                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                                    exact_err += val ** 2
                    
                    if exact_err < 1e-6:
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = r
                        break
                
                if abs(prev_err - err) < 1e-12 and it > 10:
                    break
                prev_err = err
        
        if best_result is not None:
            break
    
    return best_result


def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix."""
    start = time.time()
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    n, m, p = 3, 3, 6
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    
    # Verify it works
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        # Try to find better
        better = _try_als_decomposition(n, m, p, deadline)
        if better is not None:
            rank_better = len(better["U"][0]) if better["U"] else 0
            rank_current = len(result["U"][0]) if result["U"] else 0
            if rank_better < rank_current:
                result = better
    
    # Ensure we return before deadline
    if time.time() > deadline - 5:
        return result
    
    return result
# EVOLVE-BLOCK-END
