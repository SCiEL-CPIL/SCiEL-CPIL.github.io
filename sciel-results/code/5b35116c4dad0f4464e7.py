import numpy as np
import time

def construct():
    """
    Find rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    
    Tensor convention: W[k*n+i, r]
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    The decomposition satisfies:
    sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    where T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k, and 0 elsewhere.
    """
    n, m, p = 3, 3, 8
    dim_u = n * m      # 9
    dim_v = m * p      # 24
    dim_w = p * n      # 24
    
    deadline = time.time() + 140  # 10s safety margin
    
    # Precompute the support structure
    # T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    support_a = []
    support_b = []
    support_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                support_a.append(a)
                support_b.append(b)
                support_c.append(c)
    support_a = np.array(support_a)
    support_b = np.array(support_b)
    support_c = np.array(support_c)
    num_support = len(support_a)  # 72
    
    # Build the full constraint matrix for alternating minimization
    # For each constraint (a,b,c, target), we need:
    # sum_r U[a,r]*V[b,r]*W[c,r] = target
    # 
    # When solving for W: fix U,V, then for each c:
    #   sum_r (U[a,r]*V[b,r]) * W[c,r] = target
    # Group by c: for each c, we have constraints from all (a,b) that pair with this c
    
    # Precompute: for each c, which (a,b) pairs are in the support
    c_to_constraints = {}
    for idx in range(num_support):
        c = support_c[idx]
        a = support_a[idx]
        b = support_b[idx]
        if c not in c_to_constraints:
            c_to_constraints[c] = []
        c_to_constraints[c].append((a, b))
    
    # Also precompute zero-entry constraints for verification
    # We'll sample zero entries for verification
    
    def verify_exact(U, V, W):
        """
        Verify exact decomposition using polynomial identity testing.
        Check all 72 support entries equal 1, and verify zero entries.
        """
        # Check support entries
        vals = np.array([
            np.dot(U[support_a[i], :], V[support_b[i], :]) * W[support_c[i], :].sum()
            for i in range(num_support)
        ])
        # Actually compute properly: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Vectorized: for each support entry, compute the sum
        err_support = 0.0
        for i in range(num_support):
            a, b, c = support_a[i], support_b[i], support_c[i]
            val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
            if abs(val - 1.0) > 1e-10:
                return False
        
        # Check zero entries - sample them
        # A zero entry is (a,b,c) where not (a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k)
        # Build set of support entries
        support_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    support_set.add((i*m+j, j*p+k, k*n+i))
        
        # Sample zero entries
        np.random.seed(12345)
        checked = 0
        for _ in range(1000):
            a = np.random.randint(dim_u)
            b = np.random.randint(dim_v)
            c = np.random.randint(dim_w)
            if (a, b, c) in support_set:
                continue
            val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
            if abs(val) > 1e-10:
                return False
            checked += 1
            if checked >= 500:
                break
        
        # Also check all zero entries in a structured way
        # For each a, b, check which c values should be zero
        for a in range(dim_u):
            for b in range(dim_v):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                if j != j2:
                    # All c are zero
                    for c in range(dim_w):
                        val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                        if abs(val) > 1e-10:
                            return False
                else:
                    # Only c = k*n+i should be 1, rest zero
                    valid_c = k * n + i
                    for c in range(dim_w):
                        if c == valid_c:
                            continue
                        val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                        if abs(val) > 1e-10:
                            return False
        
        return True
    
    def verify_fast(U, V, W, tol=1e-10):
        """Faster verification for search purposes."""
        # Check support entries
        for i in range(num_support):
            a, b, c = support_a[i], support_b[i], support_c[i]
            val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
            if abs(val - 1.0) > tol:
                return False
        # Spot check zeros
        np.random.seed(999)
        support_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    support_set.add((i*m+j, j*p+k, k*n+i))
        
        for _ in range(200):
            a = np.random.randint(dim_u)
            b = np.random.randint(dim_v)
            c = np.random.randint(dim_w)
            if (a, b, c) in support_set:
                continue
            val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
            if abs(val) > tol:
                return False
        return True
    
    # Schoolbook solution
    rank_school = n * m * p
    U_school = np.zeros((dim_u, rank_school))
    V_school = np.zeros((dim_v, rank_school))
    W_school = np.zeros((dim_w, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Alternating minimization with structured initialization
    def alternating_minimize(U, V, W, max_iters=50, tol=1e-12):
        """
        Alternating linear least squares to find exact decomposition.
        Each step solves a linear system for one factor.
        """
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Solve for W given U, V
            # For each c, we need: sum_r (sum over (a,b) paired with c) U[a,r]*V[b,r]*W[c,r] = target
            # Actually, each support entry (a,b,c) gives one equation:
            #   sum_r U[a,r]*V[b,r]*W[c,r] = 1
            # For fixed U,V, this is linear in W.
            # Build system: for each c, collect all equations involving W[c,:]
            
            # More efficient: build the full linear system
            # Number of equations = num_support = 72
            # Unknowns = dim_w * rank
            # A has shape (72, dim_w * rank)
            
            A_W = np.zeros((num_support, dim_w * W.shape[1]))
            b_W = np.ones(num_support)
            
            for idx in range(num_support):
                a, b, c = support_a[idx], support_b[idx], support_c[idx]
                for r in range(W.shape[1]):
                    A_W[idx, c * W.shape[1] + r] = U[a, r] * V[b, r]
            
            W_sol, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
            W = W_sol.reshape(dim_w, W.shape[1])
            
            # Solve for U given V, W
            A_U = np.zeros((num_support, dim_u * U.shape[1]))
            b_U = np.ones(num_support)
            
            for idx in range(num_support):
                a, b, c = support_a[idx], support_b[idx], support_c[idx]
                for r in range(U.shape[1]):
                    A_U[idx, a * U.shape[1] + r] = V[b, r] * W[c, r]
            
            U_sol, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
            U = U_sol.reshape(dim_u, U.shape[1])
            
            # Solve for V given U, W
            A_V = np.zeros((num_support, dim_v * V.shape[1]))
            b_V = np.ones(num_support)
            
            for idx in range(num_support):
                a, b, c = support_a[idx], support_b[idx], support_c[idx]
                for r in range(V.shape[1]):
                    A_V[idx, b * V.shape[1] + r] = U[a, r] * W[c, r]
            
            V_sol, _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
            V = V_sol.reshape(dim_v, V.shape[1])
            
            # Check convergence
            max_err = 0.0
            for idx in range(num_support):
                a, b, c = support_a[idx], support_b[idx], support_c[idx]
                val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                max_err = max(max_err, abs(val - 1.0))
            
            if max_err < tol:
                break
        
        return U, V, W
    
    def structured_init(rank):
        """
        Create a structured initialization based on the tensor's symmetry.
        The tensor has a cyclic structure: (i,j,k) -> (i*m+j, j*p+k, k*n+i)
        """
        np.random.seed(42)
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        
        # Assign each support entry to a "slot" in the decomposition
        # Use a hash-like assignment to spread load
        for idx in range(num_support):
            r = idx % rank
            a, b, c = support_a[idx], support_b[idx], support_c[idx]
            # Add contributions
            U[a, r] += 1.0 / 3.0
            V[b, r] += 1.0 / 3.0
            W[c, r] += 1.0 / 3.0
        
        # Add random perturbation to break symmetry
        U += np.random.randn(dim_u, rank) * 0.1
        V += np.random.randn(dim_v, rank) * 0.1
        W += np.random.randn(dim_w, rank) * 0.1
        
        return U, V, W
    
    def random_init(rank, seed):
        """Random initialization."""
        np.random.seed(seed)
        U = np.random.randn(dim_u, rank) * 0.5
        V = np.random.randn(dim_v, rank) * 0.5
        W = np.random.randn(dim_w, rank) * 0.5
        return U, V, W
    
    def try_rank(rank, max_seeds=5, max_iters=30):
        """Try to find a rank-r decomposition."""
        for seed in range(max_seeds):
            if time.time() > deadline - 5:
                break
            
            # Different initialization strategies
            if seed == 0:
                U, V, W = structured_init(rank)
            elif seed == 1:
                U, V, W = random_init(rank, seed * 1000 + rank)
            elif seed == 2:
                # Perturbed schoolbook (truncate)
                U = np.zeros((dim_u, rank))
                V = np.zeros((dim_v, rank))
                W = np.zeros((dim_w, rank))
                # Map schoolbook columns to fewer ranks
                for col_idx in range(min(rank, rank_school)):
                    src_col = col_idx % rank_school
                    U[:, col_idx] = U_school[:, src_col]
                    V[:, col_idx] = V_school[:, src_col]
                    W[:, col_idx] = W_school[:, src_col]
                # Add perturbation
                U += np.random.randn(dim_u, rank) * 0.05
                V += np.random.randn(dim_v, rank) * 0.05
                W += np.random.randn(dim_w, rank) * 0.05
            elif seed == 3:
                # Sparse random init
                U, V, W = random_init(rank, seed * 777 + rank)
                # Make sparse
                U[np.random.rand(*U.shape) < 0.5] = 0
                V[np.random.rand(*V.shape) < 0.5] = 0
                W[np.random.rand(*W.shape) < 0.5] = 0
            else:
                U, V, W = random_init(rank, seed * 333 + rank * 11)
            
            # Alternating minimization
            U, V, W = alternating_minimize(U, V, W, max_iters=max_iters)
            
            # Check if we found an exact solution
            if verify_fast(U, V, W, tol=1e-10):
                # Full verification
                if verify_exact(U, V, W):
                    return U, V, W
            
            # Try to polish: if close, do more iterations
            max_err = 0.0
            for idx in range(num_support):
                a, b, c = support_a[idx], support_b[idx], support_c[idx]
                val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 0.1:
                # Close - do more iterations
                U, V, W = alternating_minimize(U, V, W, max_iters=50, tol=1e-14)
                if verify_fast(U, V, W, tol=1e-10):
                    if verify_exact(U, V, W):
                        return U, V, W
        
        return None
    
    # Try ranks from low to high
    # Lower bound: at least max(dim_u, dim_v, dim_w) = 24
    # But realistically, we expect something lower than 72
    # Try a range of ranks
    ranks_to_try = [24, 25, 26, 27, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 64, 68]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        if target_rank >= best_rank:
            continue
        
        result = try_rank(target_rank, max_seeds=4, max_iters=25)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = target_rank
            best_U, best_V, best_W = U_r, V_r, W_r
            break  # Found a lower rank, stop
    
    # If we didn't find anything better, try a few more with more iterations
    if best_rank == rank_school and time.time() < deadline - 15:
        for target_rank in [55, 50, 45, 40, 35, 30, 28, 26, 25, 24]:
            if time.time() > deadline - 5:
                break
            if target_rank >= best_rank:
                continue
            result = try_rank(target_rank, max_seeds=6, max_iters=40)
            if result is not None:
                U_r, V_r, W_r = result
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break
    
    # Final: verify and format
    # Ensure the result is valid
    if not verify_fast(best_U, best_V, best_W, tol=1e-8):
        # Fall back to schoolbook
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = rank_school
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
