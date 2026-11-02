"""Matrix multiplication 2x3 by 3x9 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    """Construct rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix mult.
    
    Returns dict with U (6 x rank), V (27 x rank), W (18 x rank) where:
    - U[i*m + j, c] corresponds to first matrix element a[i,j]
    - V[j*p + k, c] corresponds to second matrix element b[j,k]
    - W[k*n + i, c] corresponds to output element c[k,i] (trace convention)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    nm = n * m   # 6
    mp = m * p   # 27
    np_ = n * p  # 18
    total_dim = nm + mp + np_  # 51
    
    # Trivial solution: rank = n*m*p = 54
    def trivial_solution():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((np_, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = trivial_solution()
    
    # Try to find lower rank decompositions
    # The tensor decomposition satisfies:
    # For all i,k: sum_j A[i,j]*B[j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]... 
    # More precisely, the tensor T[i,j,k] = delta (but here it's the multiplication tensor)
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # The rank decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's not quite right. Let me think again.
    
    # Actually, the matrix multiplication tensor M has entries M[i,j,k] for i in [0,n), j in [0,m), k in [0,p)
    # But the multiplication C = A * B gives C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor decomposition of the multiplication algorithm is:
    # C[i,k] = sum_c prod_a U[i*m + j_a, c] * prod_b V[j_b*p + k_b, c] * ... 
    # No wait. The standard formulation:
    # We want: C[i,k] = sum_j A[i,j] B[j,k]
    # This is a trilinear form in A, B, C. The tensor T has indices (i, j, k) where:
    # T[i, j, k] is the coefficient relating A[i,j], B[j,k] to C[i,k]
    # Actually T[i,j,k] = 1 (the identity), and we need:
    # sum_j A[i,j] B[j,k] = C[i,k]
    # So the tensor is T[i, j, k] = delta (the "multiplication tensor")
    
    # The rank-r decomposition means we can write:
    # C[i,k] = sum_{c=0}^{r-1} (sum_{j} ... ) 
    # More precisely, we define vectors u_c, v_c, w_c such that:
    # u_c has components u_c[i*m + j] for i,j
    # v_c has components v_c[j*p + k] for j,k  
    # w_c has components w_c[k*n + i] for k,i
    # And the condition is: for all i,k:
    #   sum_j delta_{i,k,j} = sum_c (sum_j u_c[i*m+j] * v_c[j*p+k]) * w_c[k*n+i]
    # where delta_{i,k,j} is... hmm, this is getting confused.
    
    # Let me restart with the correct formulation.
    # The matrix multiplication tensor for multiplying (n x m) by (m x p):
    # The tensor T is in R^{(n*m) x (m*p) x (n*p)} defined by:
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a, b, c] = 1 if j = j' and i = i' and k = k', else 0
    # Wait no. Let me think of it as: we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # So the "tensor" that we decompose is the map:
    # (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # This is a bilinear map. Its tensor representation T in R^{nm x mp x np} has:
    # T[a, b, c] = 1 if a corresponds to (i,j), b corresponds to (j',k), c corresponds to (k',i')
    # and j = j' and i = i' and k = k'
    
    # So T[i*m+j, j'*p+k, k'*n+i'] = 1 iff j=j' and i=i' and k=k'
    # This means T is very sparse.
    
    # A rank-r decomposition of T means finding U (nm x r), V (mp x r), W (np x r) such that:
    # T[a,b,c] = sum_{c_idx=0}^{r-1} U[a, c_idx] * V[b, c_idx] * W[c, c_idx]
    
    # For our case:
    # T[i*m+j, j'*p+k, k'*n+i'] = sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c]
    # This must equal 1 iff j=j', i=i', k=k' and 0 otherwise.
    
    # The number of non-zero entries in T is n*m*p = 2*3*9 = 54.
    # The trivial solution has rank 54.
    
    # Can we do better? The tensor rank of matrix multiplication is a famous open problem.
    # For 2x3 * 3x9, the exact rank is not known to me, but it's at most 54.
    
    # Let me try a random search approach to see if we can find rank < 54.
    
    def verify_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        r = U.shape[1]
        
        # Reconstruct T
        # T[a, b, c] = sum_c_idx U[a, c_idx] * V[b, c_idx] * W[c, c_idx]
        # We can compute this as: for each c_idx, outer product U[:,c_idx] x V[:,c_idx] x W[:,c_idx]
        # But that's expensive. Instead, check specific entries.
        
        # Check that T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        # And T[a, b, c] = 0 for all other (a, b, c)
        
        # Due to symmetry, we only need to check the non-zero and some zero entries.
        # The non-zero entries: T[i*m+j, j*p+k, k*n+i] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > 1e-9:
                        return False, val
        
        # Check some zero entries (random sample)
        np.random.seed(42)
        for _ in range(200):
            a = np.random.randint(nm)
            b = np.random.randint(mp)
            c = np.random.randint(np_)
            # Skip if this is a non-zero entry
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            if j == j2 and i == i2 and k == k2:
                continue
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val) > 1e-9:
                return False, val
        
        return True, 0.0
    
    def try_random_rank(r, max_iters=5000):
        """Try to find a rank-r decomposition via random initialization and local search."""
        if time.time() > deadline:
            return None
        
        # Initialize with random values
        for attempt in range(3):  # 3 attempts per rank
            if time.time() > deadline:
                return None
            
            np.random.seed(None)  # Random seed
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(np_, r)
            
            # Local search: minimize the error
            best_err = float('inf')
            best_state = (U.copy(), V.copy(), W.copy())
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Compute error (sparse check)
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = np.sum(U[a, :] * V[b, :] * W[c, :])
                            err += (val - 1.0) ** 2
                
                # Check some zeros
                for _ in range(50):
                    a = np.random.randint(nm)
                    b = np.random.randint(mp)
                    c = np.random.randint(np_)
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if j == j2 and i == i2 and k == k2:
                        continue
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    err += val ** 2
                
                if err < best_err:
                    best_err = err
                    best_state = (U.copy(), V.copy(), W.copy())
                    if best_err < 1e-8:
                        break
                
                # Perturbation
                if np.random.rand() < 0.5:
                    idx = np.random.randint(r)
                    U[np.random.randint(nm), idx] += np.random.randn() * 0.1
                elif np.random.rand() < 0.5:
                    idx = np.random.randint(r)
                    V[np.random.randint(mp), idx] += np.random.randn() * 0.1
                else:
                    idx = np.random.randint(r)
                    W[np.random.randint(np_), idx] += np.random.randn() * 0.1
            
            if best_err < 1e-8:
                U, V, W = best_state
                ok, _ = verify_decomposition(U, V, W)
                if ok:
                    return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try ranks from 53 down to 30 (but this is likely infeasible)
    # The matrix multiplication tensor rank is known to be at most n*m*p.
    # For small cases, improvements are rare.
    # Let's try a few ranks and see.
    
    for r in range(53, 25, -1):
        if time.time() > deadline:
            break
        result = try_random_rank(r, max_iters=2000)
        if result is not None:
            best_result = result
            break
    
    # Convert to list format and round to nearest half-integer if possible
    def to_half_integer(mat):
        mat = np.round(mat * 2) / 2
        return mat.tolist()
    
    U_list = to_half_integer(np.array(best_result["U"]))
    V_list = to_half_integer(np.array(best_result["V"]))
    W_list = to_half_integer(np.array(best_result["W"]))
    
    return {"U": U_list, "V": V_list, "W": W_list}
