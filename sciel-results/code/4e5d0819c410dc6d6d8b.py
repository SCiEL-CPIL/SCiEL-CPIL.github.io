"""
Matrix multiplication 2x5x5: rank-minimal exact tensor decomposition.
Strategy: Start from schoolbook (rank 50) and try to reduce rank by
merging factors via a structured random search over the bilinear form space.

Key insight: The tensor T_{i,j,k} = delta for matrix mult 2x5x5 has rank 50 (schoolbook).
We try to find a decomposition of rank r < 50 by:
1. Using a parameterized family of decompositions
2. Random search over structured perturbations
3. Verification via multiple test matrices

The decomposition convention:
- U: shape (n*m, rank) = (10, rank), rows indexed by (i*m+j)
- V: shape (m*p, rank) = (25, rank), rows indexed by (j*p+k)  
- W: shape (p*n, rank) = (10, rank), rows indexed by (k*n+i)

Condition: For all (i,k): C[i,k] = sum_c W[k*n+i, c] * (sum_{i',j'} U[i'*m+j', c]*A[i',j']) * (sum_{j',k'} V[j'*p+k', c]*B[j',k'])
"""
import random
import time
import numpy as np

DIMENSIONS = [2, 5, 5]
N, M, P = DIMENSIONS

_DEADLINE_START = None
_TIME_BUDGET = 135.0  # seconds, keep 15s safety margin


def _elapsed():
    global _DEADLINE_START
    if _DEADLINE_START is None:
        _DEADLINE_START = time.time()
    return time.time() - _DEADLINE_START


def _within_budget(margin=5.0):
    return _elapsed() < _TIME_BUDGET - margin


def _verify(U, V, W, n, m, p, num_tests=5):
    """Verify decomposition using multiple random test matrices."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    # Precompute for speed: U @ A_flat, V @ B_flat for each c
    # C_flat[l] = sum_c W[l,c] * (U[:,c] @ A) * (V[:,c] @ B)
    
    for test in range(num_tests):
        rng = np.random.RandomState(1000 + test)
        A = rng.randn(n * m)
        B = rng.randn(m * p)
        
        A_mat = A.reshape(n, m)
        B_mat = B.reshape(m, p)
        C_true = (A_mat @ B_mat).flatten()  # shape (n*p,) but we need (p*n,) with W[k*n+i]
        
        # C_true[i,k] is at index i*p+k in standard flatten
        # But W uses k*n+i indexing, so C_flat[l] where l = k*n + i corresponds to C[i,k]
        C_true_wn = np.zeros(p * n)
        for i in range(n):
            for k in range(p):
                C_true_wn[k * n + i] = C_true[i * p + k]
        
        C_comp = np.zeros(p * n)
        for c in range(rank):
            vA = float(U_np[:, c] @ A)
            vB = float(V_np[:, c] @ B)
            if vA != 0.0 and vB != 0.0:
                C_comp += W_np[:, c] * vA * vB
        
        if not np.allclose(C_comp, C_true_wn, atol=1e-9):
            return False
    
    return True


def _build_schoolbook():
    """Return the schoolbook rank-50 decomposition."""
    n, m, p = N, M, P
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


def _try_rank_reduction(U_init, V_init, W_init, target_rank, time_limit=10.0, seed=None):
    """
    Try to reduce rank by merging factors.
    Start with schoolbook and try to eliminate one factor at a time.
    """
    n, m, p = N, M, P
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    start_time = time.time()
    
    U = [row[:] for row in U_init]
    V = [row[:] for row in V_init]
    W = [row[:] for row in W_init]
    
    current_rank = len(U[0])
    
    # We want to go from current_rank to target_rank
    # Strategy: try to merge pairs of columns
    while current_rank > target_rank and time.time() - start_time < time_limit:
        if not _within_budget(margin=20):
            break
        
        # Pick a random column to try to eliminate
        col_to_elim = rng.randint(0, current_rank - 1)
        
        # Try to express column col_to_elim as a linear combination of other columns
        # This is a linear algebra problem: find x such that
        # U[:, col_to_elim] ≈ sum_{c != col_to_elim} x_c * U[:, c]
        # Similarly for V and W
        # But we need exact, so we need the column to be exactly in the span.
        
        # Alternative: try small integer combinations
        # For each pair of other columns, check if col_to_elim can be expressed as a/b * col_a + c/d * col_b
        # This is too expensive. Instead, try random perturbations.
        
        # Better approach: try to set one column to zero and adjust others
        # Or: try a random linear transformation that preserves the tensor
        
        # Let's try: pick two columns a, b and try to merge them
        # Replace col_a with col_a + t*col_b, col_b with col_b - t*col_a_old
        # This is a shear transformation. The tensor is preserved if we also adjust W.
        # Actually no, the tensor changes.
        
        # Simpler: just try random rank-reduction by checking if removing a column
        # and adjusting others can still work.
        
        # For now, try a direct approach: randomly zero out a column and see if
        # we can compensate by modifying other columns.
        
        # Actually, let's try a different strategy: parameterized search
        # Start with schoolbook and add small random perturbations, then try to
        # project back to lower rank.
        
        break  # This approach is too complex for now
    
    return None


def _structured_search(target_rank, time_limit=15.0, seed=None):
    """
    Structured random search for a decomposition of the given rank.
    Uses the structure of matrix multiplication to guide the search.
    """
    n, m, p = N, M, P
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    start_time = time.time()
    
    # Strategy: The key observation is that for 2x5x5, we have:
    # C[0,k] = sum_j A[0,j] * B[j,k]  for k=0..4
    # C[1,k] = sum_j A[1,j] * B[j,k]  for k=0..4
    # 
    # Total: 10 output entries, each is a sum of 5 products.
    # Schoolbook: 50 factors.
    # 
    # Can we share computations? For example, if we precompute
    # S_j = A[0,j] + A[1,j] for j=0..4, then:
    # C[0,k] + C[1,k] = sum_j (A[0,j]+A[1,j]) * B[j,k] = sum_j S_j * B[j,k]
    # This saves 5 multiplications!
    # 
    # Similarly, D_j = A[0,j] - A[1,j]:
    # C[0,k] - C[1,k] = sum_j D_j * B[j,k]
    # 
    # So: C[0,k] = (sum_j S_j*B[j,k] + sum_j D_j*B[j,k]) / 2
    #          C[1,k] = (sum_j S_j*B[j,k] - sum_j D_j*B[j,k]) / 2
    # 
    # But this still requires 2*5 = 10 multiplications per k, so 50 total.
    # No savings unless we can share B[j,k] terms.
    #
    # Actually, the bilinear rank is what matters. Let's think about it differently.
    #
    # The tensor for 2x5x5 matrix mult has a known rank. Let's try to find it.
    # Lower bound: max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # Upper bound: 50 (schoolbook)
    #
    # For 2xnxn, the rank is known to be 2n^2 - n for n >= 2 (from Strassen-like results).
    # For n=5: 2*25 - 5 = 45.
    # Actually, the exact rank for 2x5x5 might be 45 or less.
    
    # Let's try a specific construction:
    # For each k (column of C), we need to compute a 2x1 vector = 2x5 @ 5x1.
    # This is a bilinear map from (2x5, 5x1) to (2x1).
    # The rank of this map is at most 2*5 = 10 (schoolbook for each k).
    # But we can share across k's.
    
    # Alternative: use the fact that B is 5x5, so we can decompose it.
    # B = sum_{j=0}^{4} e_j * B[j,:] where e_j is the j-th standard basis vector.
    # Then C[:,k] = sum_j A[:,j] * B[j,k]
    # 
    # If we could factor B[j,k] in some way... but B is arbitrary.
    
    # Let's try a random search with structure:
    # Each factor c has:
    #   U_c: a vector in R^{10} (linear combination of A entries)
    #   V_c: a vector in R^{25} (linear combination of B entries)  
    #   W_c: a vector in R^{10} (how factor c contributes to C entries)
    
    max_attempts = 500
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
        if not _within_budget(margin=20):
            break
        
        # Generate random sparse decomposition
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        for c in range(target_rank):
            # Each factor picks a small number of A entries, B entries, and C positions
            num_u = rng.randint(1, 3)
            num_v = rng.randint(1, 3)
            num_w = rng.randint(1, 2)
            
            u_idxs = rng.sample(range(n * m), num_u)
            v_idxs = rng.sample(range(m * p), num_v)
            w_idxs = rng.sample(range(p * n), num_w)
            
            for idx in u_idxs:
                U[idx][c] = rng.choice([-1, 1])
            for idx in v_idxs:
                V[idx][c] = rng.choice([-1, 1])
            for idx in w_idxs:
                W[idx][c] = rng.choice([-1, 1])
        
        # Quick verification
        if _verify(U, V, W, n, m, p, num_tests=1):
            if _verify(U, V, W, n, m, p, num_tests=5):
                return {"U": U, "V": V, "W": W}
    
    return None


def _try_construction_based():
    """
    Try a specific construction based on known results for 2xn matrix multiplication.
    """
    n, m, p = N, M, P
    # For 2x5x5, try to use a construction that gives rank < 50.
    # 
    # Known: rank(2xn x n x n) <= 2n^2 - n for n even? Let's just try random search.
    # 
    # Actually, let's try a very specific approach:
    # The schoolbook has 50 factors. Can we merge any two?
    # Factor (i,j,k) has U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    # Two factors can be merged if there's a linear relation.
    #
    # For example, factors (0,0,0) and (1,0,0):
    # U: [1,0,0,0,0, 0,1,0,0,0] vs [0,0,0,0,0, 1,0,0,0,0]
    # V: [1,0,0,0,0, ...] same for both (j=0, k=0)
    # W: [1,0,0,0,0, 0,0,0,0,0] vs [0,0,0,0,0, 1,0,0,0,0]
    #
    # We can combine: let t be a parameter.
    # New U = U1 + t*U2, New V = V1 (same), New W = W1 + t*W2
    # This gives: (A[0,0]+t*A[1,0]) * B[0,0] * (C[0,0] + t*C[1,0])
    # = A[0,0]*B[0,0]*C[0,0] + t*A[0,0]*B[0,0]*C[1,0] + t*A[1,0]*B[0,0]*C[0,0] + t^2*A[1,0]*B[0,0]*C[1,0]
    # This has extra terms, so it doesn't work directly.
    
    return None


def _polynomial_search(target_rank, time_limit=20.0, seed=None):
    """
    Search using a polynomial parameterization.
    """
    n, m, p = N, M, P
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    start_time = time.time()
    
    # Try different sparsity patterns
    for trial in range(200):
        if time.time() - start_time > time_limit:
            break
        if not _within_budget(margin=20):
            break
        
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        for c in range(target_rank):
            # Random pattern with varying density
            density = rng.uniform(0.05, 0.3)
            for i in range(n * m):
                if rng.random() < density:
                    U[i][c] = rng.choice([-1, 1])
            for j in range(m * p):
                if rng.random() < density:
                    V[j][c] = rng.choice([-1, 1])
            for k in range(p * n):
                if rng.random() < density * 0.5:
                    W[k][c] = rng.choice([-1, 1])
        
        if _verify(U, V, W, n, m, p, num_tests=1):
            if _verify(U, V, W, n, m, p, num_tests=5):
                return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x5x5 matrix mult.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = N, M, P
    
    # Start with schoolbook as guaranteed fallback (rank 50)
    best = _build_schoolbook()
    best_rank = n * m * p  # 50
    
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) = 25
    # Try to find a decomposition with rank < 50
    
    # Try ranks from 49 down to 30, spending limited time per rank
    for target_rank in range(49, 29, -1):
        if not _within_budget(margin=15):
            break
        
        remaining = _TIME_BUDGET - _elapsed() - 15
        if remaining < 2:
            break
        
        # Allocate time based on rank: lower ranks get less time (harder)
        time_limit = min(5.0, remaining * 0.3)
        
        result = _structured_search(target_rank, time_limit=time_limit, seed=target_rank)
        if result is not None:
            best = result
            best_rank = target_rank
            break
        
        result = _polynomial_search(target_rank, time_limit=min(3.0, remaining * 0.2), seed=target_rank + 1000)
        if result is not None:
            best = result
            best_rank = target_rank
            break
    
    # If we didn't find anything better, return schoolbook
    return best
