<<<<<<< SEARCH
def construct():
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook solution as baseline
    rank_school = n * m * p
    U_s = np.zeros((n * m, rank_school))
    V_s = np.zeros((m * p, rank_school))
    W_s = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
    
    lower_bound = max(n*m, m*p, p*n)
    
    # Try structured decompositions based on Kronecker products
    # Key idea: 2x3 x 3x8 can be decomposed using block structure
    
    # Strategy 1: Try rank values from lower_bound to best_rank
    # Use structured initializations based on known small-case results
    
    # For 2x3 x 3x8, try to use the fact that 3x8 = 3x(2x4)
    # and use a rank-5 Strassen-like decomposition for 2x2 x 2x2 blocks
    
    # Let's try a systematic search with better initializations
    
    vals = np.array([0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0])
    
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 15.0:
            break
        
        found = False
        
        # Try multiple random initializations with structure
        for trial in range(300):
            if time.time() > deadline - 5.0:
                break
            
            rng = np.random.RandomState(trial * 7919 + target_rank * 137 + 42)
            
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Sparse random initialization
            density = 0.3 + 0.3 * (target_rank / best_rank)
            for idx in range(n*m):
                mask = rng.random(target_rank) < density
                U[idx, mask] = rng.choice(vals, size=np.sum(mask))
            for idx in range(m*p):
                mask = rng.random(target_rank) < density
                V[idx, mask] = rng.choice(vals, size=np.sum(mask))
            for idx in range(p*n):
                mask = rng.random(target_rank) < density
                W[idx, mask] = rng.choice(vals, size=np.sum(mask))
            
            if _verify_numpy(U, V, W, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                found = True
                break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
=======
def construct():
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook solution as baseline
    rank_school = n * m * p
    U_s = np.zeros((n * m, rank_school))
    V_s = np.zeros((m * p, rank_school))
    W_s = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
    
    lower_bound = max(n*m, m*p, p*n)
    
    # Structured search using block decomposition:
    # 2x3 x 3x8 can be viewed as 4 blocks of 2x2 x 2x4
    # where the 3 columns are split as [col0, col1, col2] and 8 rows as [row0..row7]
    # Actually: 2x3 means 2 rows, 3 cols; 3x8 means 3 rows, 8 cols
    # Block structure: split the 3 middle dimension into 3 parts of size 1,
    # and the 8 output columns into 4 parts of size 2
    # This gives 3*4 = 12 blocks of 2x1 x 1x2, each of rank 4 schoolbook
    
    # Better: Use the fact that we can share structure across blocks
    # Try a Kronecker-like product structure
    
    # Strategy: Search over structured decompositions with parameters
    # U[i,j, r] = sum of products of factors
    
    # Let's try a different approach: use the known optimal rank for 2x2 x 2x4
    # which is at most 16, and try to find it via random search
    
    # First, let's try to find a good decomposition for 2x2 x 2x4
    # then use it to build 2x3 x 3x8
    
    def find_2x2_x_2x4(rank_target, max_trials=500):
        """Find rank decomposition for 2x2 x 2x4 matrix multiplication."""
        nn, mm, pp = 2, 2, 4
        vals = np.array([0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 0.0])
        
        for trial in range(max_trials):
            rng = np.random.RandomState(trial * 7919 + rank_target * 137 + 42)
            
            U = np.zeros((nn*mm, rank_target))
            V = np.zeros((mm*pp, rank_target))
            W = np.zeros((pp*nn, rank_target))
            
            density = 0.2 + 0.2 * (rank_target / 16.0)
            for idx in range(nn*mm):
                mask = rng.random(rank_target) < density
                if np.sum(mask) > 0:
                    U[idx, mask] = rng.choice(vals, size=np.sum(mask))
            for idx in range(mm*pp):
                mask = rng.random(rank_target) < density
                if np.sum(mask) > 0:
                    V[idx, mask] = rng.choice(vals, size=np.sum(mask))
            for idx in range(pp*nn):
                mask = rng.random(rank_target) < density
                if np.sum(mask) > 0:
                    W[idx, mask] = rng.choice(vals, size=np.sum(mask))
            
            # Verify
            valid = True
            for i in range(nn):
                for j in range(mm):
                    for k in range(pp):
                        s = np.dot(U[i*mm+j], V[j*pp+k] * W[k*nn+i])
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return U, V, W
        return None
    
    # Try to find 2x2 x 2x4 decomposition
    for r2224 in range(8, 17):
        if time.time() > deadline - 30.0:
            break
        result = find_2224(r2224, max_trials=200)
        if result is not None:
            print(f"Found 2x2 x 2x4 with rank {r2224}")
            break
    
    # Now try structured search for 2x3 x 3x8
    # Use a parameterized family of decompositions
    
    # Family 1: Kronecker-like structure
    # U[i,j,r] = A[i, r1] * B[j, r2] where r = (r1, r2)
    # This is too restrictive; let's try something more general
    
    # Family 2: Shared rank structure
    # Assume the rank-r decomposition has a block structure where
    # certain columns of U, V, W are related
    
    vals = np.array([0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 0.0])
    
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 15.0:
            break
        
        found = False
        
        for trial in range(500):
            if time.time() > deadline - 5.0:
                break
            
            rng = np.random.RandomState(trial * 7919 + target_rank * 137 + 42)
            
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # More structured initialization: group columns
            # Each "group" of columns handles a subset of (i,j,k) triples
            
            # Try sparse with higher density for smaller ranks
            density = 0.15 + 0.25 * (target_rank / best_rank)
            for idx in range(n*m):
                mask = rng.random(target_rank) < density
                if np.sum(mask) > 0:
                    U[idx, mask] = rng.choice(vals, size=np.sum(mask))
            for idx in range(m*p):
                mask = rng.random(target_rank) < density
                if np.sum(mask) > 0:
                    V[idx, mask] = rng.choice(vals, size=np.sum(mask))
            for idx in range(p*n):
                mask = rng.random(target_rank) < density
                if np.sum(mask) > 0:
                    W[idx, mask] = rng.choice(vals, size=np.sum(mask))
            
            if _verify_numpy(U, V, W, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                found = True
                break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
>>>>>>> REPLACE
