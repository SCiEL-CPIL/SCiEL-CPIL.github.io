import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def _build_schoolbook(n, m, p):
    """Build the schoolbook (naive) matrix multiplication decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the tensor decomposition reconstructs matrix multiplication exactly."""
    rank = U.shape[1]
    # Check all "one" entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                g = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[g, :])
                if abs(val - 1.0) > tol:
                    return False
    # Check zero constraints (sample)
    rng = random.Random(42)
    for _ in range(2000):
        a = rng.randint(0, n * m - 1)
        b = rng.randint(0, m * p - 1)
        g = rng.randint(0, p * n - 1)
        i = a // m
        j = a % m
        jp = b // p
        k = b % p
        kn = g // n
        ip = g % n
        if i != ip or j != jp or k != kn:
            val = np.sum(U[a, :] * V[b, :] * W[g, :])
            if abs(val) > tol:
                return False
    return True

def _round_to_half_integer(M):
    """Round matrix entries to nearest 0.5."""
    return np.round(M * 2) / 2

def _build_strassen_2x2():
    """Build the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    Returns U, V, W matrices for 2x2 * 2x2 multiplication with rank 7.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    
    Strassen's 7 products:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    Result:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # A indices: A11=0, A12=1, A21=2, A22=3
    # B indices: B11=0, B12=1, B21=2, B22=3
    # C indices: C11=0, C12=1, C21=2, C22=3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0, 0] = 1  # A11
    U[3, 0] = 1  # A22
    V[0, 0] = 1  # B11
    V[3, 0] = 1  # B22
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # M2 = (A21 + A22)B11
    U[2, 1] = 1  # A21
    U[3, 1] = 1  # A22
    V[0, 1] = 1  # B11
    W[2, 1] = 1  # C21
    W[3, 1] = -1 # C22
    
    # M3 = A11(B12 - B22)
    U[0, 2] = 1  # A11
    V[1, 2] = 1  # B12
    V[3, 2] = -1 # B22
    W[1, 2] = 1  # C12
    W[3, 2] = 1  # C22
    
    # M4 = A22(B21 - B11)
    U[3, 3] = 1  # A22
    V[2, 3] = 1  # B21
    V[0, 3] = -1 # B11
    W[0, 3] = 1  # C11
    W[2, 3] = 1  # C21
    
    # M5 = (A11 + A12)B22
    U[0, 4] = 1  # A11
    U[1, 4] = 1  # A12
    V[3, 4] = 1  # B22
    W[0, 4] = -1 # C11
    W[1, 4] = 1  # C12
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2, 5] = 1  # A21
    U[0, 5] = -1 # A11
    V[0, 5] = 1  # B11
    V[1, 5] = 1  # B12
    W[3, 5] = 1  # C22
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1  # A12
    U[3, 6] = -1 # A22
    V[2, 6] = 1  # B21
    V[3, 6] = 1  # B22
    W[0, 6] = 1  # C11
    
    return U, V, W

def _build_padded_strassen(n, m, p):
    """
    Build a decomposition using Strassen's algorithm applied to padded blocks.
    
    For 4x5 * 5x6:
    - Pad A (4x5) to 8x8 by embedding in top-left
    - Pad B (5x6) to 8x8 by embedding in top-left
    - Use Strassen recursively: 8x8 = 2x2 blocks of 4x4, each 4x4 = 2x2 blocks of 2x2
    - This gives rank 7^3 = 343 for full 8x8, but we only need the top-left 4x6 of C
    
    Actually, let's think more carefully. We can use a hybrid approach:
    - Split the 4x5 * 5x6 into blocks and apply Strassen where beneficial
    """
    # For 4x5 * 5x6, let's try a direct block approach
    # Split A into [A1; A2] where A1 is 2x5, A2 is 2x5
    # Split B into [B1 | B2] where B1 is 5x3, B2 is 5x3
    # C = [A1; A2] * [B1 | B2] = [[A1*B1, A1*B2], [A2*B1, A2*B2]]
    # Each sub-product is 2x5 * 5x3 = 2x3, rank <= 6
    # Total rank <= 4*6 = 24 if we do them independently
    # But we can share some structure
    
    # Let's try: split into 2x2 blocks of size 2x2 with some padding
    # Actually, the simplest improvement: use Strassen on 2x2 blocks
    # 4x5 * 5x6 can be seen as:
    # A = [[A11, A12], [A21, A22]] where each is 2x2 (with padding for 5->4)
    # But 5 is odd, so we need to handle the extra row/col
    
    # Let's use a different approach: Kronecker product with Strassen
    # 4x5 * 5x6: pad to 4x8 * 8x6? No, that changes the problem.
    
    # Better: just use the structure directly
    # The key insight: for small matrices, we can find explicit low-rank decompositions
    
    # For 4x5 * 5x6, let's try rank 80 (schoolbook is 120)
    # Strategy: group operations to share multiplications
    
    # Actually, let's just return None and use the search-based approach
    return None

def _structured_search(n, m, p, target_rank, max_time=10, seed=42):
    """
    Search for a rank-target_rank decomposition using structured patterns.
    
    Strategy:
    1. Try block-structured decompositions based on partitioning indices
    2. Use Strassen-like sharing where possible
    3. Solve for the remaining degrees of freedom
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    rng = random.Random(seed)
    
    # Try several structured patterns
    patterns_tried = 0
    
    # Pattern 1: Group by i (rows of A)
    # For each i, we have m*p products. Try to share within each i-group.
    # For fixed i, C[i,k] = sum_j A[i,j] * B[j,k] for k=0..p-1
    # This is a vector of length p for each i.
    
    # Pattern 2: Use the fact that n=4, m=5, p=6
    # 4*5*6 = 120 schoolbook
    # Try to reduce by sharing: for each pair (i,j), we compute p products
    # If we can share across i for the same j, we save
    
    # Let's try a systematic approach:
    # Assign each (i,j,k) triple to one or more rank indices
    # For a rank index r, define:
    #   I_r = set of i values where U[i*m+j, r] != 0 for some j
    #   J_r = set of j values where U[i*m+j, r] != 0 for some i  
    #   K_r = set of k values where V[j*p+k, r] != 0 for some j
    #   The contribution to (i,j,k) is U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # Simplified: try rank = n*p + m*p + n*m - something
    # For 4x6 + 5x6 + 4x5 = 24 + 30 + 20 = 74, minus overlaps
    
    # Let's try explicit constructions
    # Construction: for each j, compute A[:,j] * B[j,:] (outer product)
    # This uses n*p = 24 rank indices per j, total 5*24 = 120 (schoolbook)
    # To save, we need to share across j values
    
    # Alternative: use the "sum of outer products" structure
    # C = sum_j A[:,j] B[j,:]
    # Each term is rank n*p = 24, total rank 5*24 = 120
    
    # To reduce rank, we need a different decomposition
    # One approach: factor through a smaller intermediate
    
    # Let's try: C = A * B where we compute A*B via a rank-r factorization
    # If we can write A*B = (A*F) * (F^{-1}*B) for some F, that doesn't help directly
    
    # For now, let's try random structured patterns and solve
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Create sparse random U and V
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        
        # Each row of U gets ~2 nonzeros, each row of V gets ~2 nonzeros
        nnz_u = max(1, target_rank // (n * m) * 2)
        nnz_v = max(1, target_rank // (m * p) * 2)
        
        for a in range(n * m):
            cols = rng.sample(range(target_rank), min(nnz_u, target_rank))
            for c in cols:
                U[a, c] = rng.choice([1, -1])
        
        for b in range(m * p):
            cols = rng.sample(range(target_rank), min(nnz_v, target_rank))
            for c in cols:
                V[b, c] = rng.choice([1, -1])
        
        # Solve for W: for each g, A_g * W[g,:] = t_g
        # where A_g has rows for all (a,b) pairs, A_g[(a,b),r] = U[a,r]*V[b,r]
        # t_g[(a,b)] = 1 if (a,b,g) is a "one" entry, else 0
        
        W = np.zeros((p * n, target_rank))
        solved = True
        
        for g in range(p * n):
            k = g // n
            i = g % n
            
            # Build system: for each (a,b), equation is sum_r U[a,r]*V[b,r]*W[g,r] = T[a,b,g]
            # This is: (U_a * V_b) dot W_g = T[a,b,g]
            # We have (n*m)*(m*p) equations in target_rank unknowns
            
            # To make this tractable, only include equations where U[a,r]*V[b,r] is nonzero
            # for some r, and group by r
            
            # Actually, let's just use least squares with a subset of equations
            # Include all "one" equations and a random sample of "zero" equations
            
            A_rows = []
            t_vals = []
            
            # Add all "one" equations for this g
            for j in range(m):
                a = i * m + j
                b = j * p + k
                A_rows.append(U[a, :] * V[b, :])
                t_vals.append(1.0)
            
            # Add random "zero" equations
            num_zero = min(50, n * m * m * p - m)
            for _ in range(num_zero):
                a = rng.randint(0, n * m - 1)
                b = rng.randint(0, m * p - 1)
                ai = a // m
                aj = a % m
                bj = b // p
                bk = b % p
                if ai == i and aj == bj and bk == k:
                    continue  # Skip "one" entries
                A_rows.append(U[a, :] * V[b, :])
                t_vals.append(0.0)
            
            if not A_rows:
                continue
            
            A = np.array(A_rows)
            t = np.array(t_vals)
            
            try:
                w, _, _, _ = np.linalg.lstsq(A, t, rcond=None)
                W[g, :] = w
            except:
                solved = False
                break
        
        if not solved:
            continue
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
        
        patterns_tried += 1
    
    return None

def _fast_als(n, m, p, target_rank, max_time=10, seed=42):
    """
    Fast Alternating Least Squares with pre-computed structures.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    np.random.seed(seed)
    
    # Pre-compute index mappings
    a_to_bg = {}
    for i in range(n):
        for j in range(m):
            a = i * m + j
            pairs = []
            for k in range(p):
                b = j * p + k
                g = k * n + i
                pairs.append((b, g))
            a_to_bg[a] = pairs
    
    b_to_ag = {}
    for j in range(m):
        for k in range(p):
            b = j * p + k
            pairs = []
            for i in range(n):
                a = i * m + j
                g = k * n + i
                pairs.append((a, g))
            b_to_ag[b] = pairs
    
    g_to_ab = {}
    for k in range(p):
        for i in range(n):
            g = k * n + i
            pairs = []
            for j in range(m):
                a = i * m + j
                b = j * p + k
                pairs.append((a, b))
            g_to_ab[g] = pairs
    
    # Try multiple initializations
    for init_seed in range(3):
        if time.time() > deadline:
            break
        
        np.random.seed(seed + init_seed)
        
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(30):
            if time.time() > deadline:
                break
            
            # Update U
            G_V = V.T @ V
            G_W = W.T @ W
            A_mat = G_V * G_W
            A_mat += 1e-8 * np.eye(target_rank)
            
            for a in range(n * m):
                b_vec = np.zeros(target_rank)
                for b, g in a_to_bg[a]:
                    b_vec += V[b, :] * W[g, :]
                try:
                    U[a, :] = np.linalg.solve(A_mat, b_vec)
                except:
                    pass
            
            # Update V
            G_U = U.T @ U
            G_W = W.T @ W
            A_mat = G_U * G_W
            A_mat += 1e-8 * np.eye(target_rank)
            
            for b in range(m * p):
                b_vec = np.zeros(target_rank)
                for a, g in b_to_ag[b]:
                    b_vec += U[a, :] * W[g, :]
                try:
                    V[b, :] = np.linalg.solve(A_mat, b_vec)
                except:
                    pass
            
            # Update W
            G_U = U.T @ U
            G_V = V.T @ V
            A_mat = G_U * G_V
            A_mat += 1e-8 * np.eye(target_rank)
            
            for g in range(p * n):
                b_vec = np.zeros(target_rank)
                for a, b in g_to_ab[g]:
                    b_vec += U[a, :] * V[b, :]
                try:
                    W[g, :] = np.linalg.solve(A_mat, b_vec)
                except:
                    pass
            
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        g = k * n + i
                        val = np.sum(U[a, :] * V[b, :] * W[g, :])
                        error += (val - 1.0) ** 2
            
            # Sample zero constraints
            rng = np.random.RandomState(seed + init_seed + iteration)
            for _ in range(100):
                a = rng.randint(0, n * m)
                b = rng.randint(0, m * p)
                g = rng.randint(0, p * n)
                i = a // m
                j = a % m
                jp = b // p
                k = b % p
                kn = g // n
                ip = g % n
                if i != ip or j != jp or k != kn:
                    val = np.sum(U[a, :] * V[b, :] * W[g, :])
                    error += val ** 2
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < 1e-8:
                break
        
        # Try to round and verify
        U_rounded = _round_to_half_integer(best_U)
        V_rounded = _round_to_half_integer(best_V)
        W_rounded = _round_to_half_integer(best_W)
        
        if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
            return U_rounded, V_rounded, W_rounded
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    best_rank = best_U.shape[1]
    
    # Try structured search for various ranks
    for target_rank in [110, 100, 90, 80, 70, 60]:
        if time.time() > deadline - 30:
            break
        result = _structured_search(n, m, p, target_rank, max_time=10, seed=target_rank)
        if result is not None:
            U, V, W = result
            if U.shape[1] < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = U.shape[1]
    
    # Try ALS optimization
    if time.time() < deadline - 15:
        for target_rank in [100, 90, 80]:
            if time.time() > deadline - 10:
                break
            remaining = int(deadline - time.time() - 5)
            if remaining < 5:
                break
            result = _fast_als(n, m, p, target_rank, max_time=remaining, seed=target_rank)
            if result is not None:
                U, V, W = result
                if U.shape[1] < best_rank:
                    best_U, best_V, best_W = U, V, W
                    best_rank = U.shape[1]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
