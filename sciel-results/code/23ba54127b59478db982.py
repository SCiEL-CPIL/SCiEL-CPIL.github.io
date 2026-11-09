"""Matrix multiplication 2x3 x 3x8 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[a,b,c] where:
    a = i*m + j, b = j*p + k, c = k*n + i
    T[a,b,c] = 1 for all valid (i,j,k) triples.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _reconstruct(U, V, W):
    """Reconstruct tensor from factors."""
    r = U.shape[1]
    recon = np.zeros((U.shape[0], V.shape[0], W.shape[0]), dtype=np.float64)
    for rr in range(r):
        recon += np.einsum('a,b,c->abc', U[:, rr], V[:, rr], W[:, rr])
    return recon


def _verify_exact(T, U, V, W, tol=1e-6):
    """Check if decomposition is exact."""
    recon = _reconstruct(U, V, W)
    return np.max(np.abs(T - recon)) < tol


def _als_optimize(T, U, V, W, rank, max_iter=80, tol=1e-10, time_deadline=None):
    """ALS optimization for tensor decomposition."""
    a_size, b_size, c_size = T.shape
    start_time = time.time()
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        # Update U: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed a: T[a,:,:] = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
        M_U = np.einsum('br,cr->bcr', V, W).reshape(b_size * c_size, rank)
        T_flat = T.reshape(a_size, b_size * c_size)
        U_new, _, _, _ = np.linalg.lstsq(M_U, T_flat.T, rcond=None)
        U = U_new.T
        
        # Update V
        M_V = np.einsum('ar,cr->acr', U, W).reshape(a_size * c_size, rank)
        T_flat_v = T.transpose(0, 2, 1).reshape(b_size, a_size * c_size)
        V_new, _, _, _ = np.linalg.lstsq(M_V, T_flat_v.T, rcond=None)
        V = V_new.T
        
        # Update W
        M_W = np.einsum('ar,br->abr', U, V).reshape(a_size * b_size, rank)
        T_flat_w = T.transpose(2, 0, 1).reshape(c_size, a_size * b_size)
        W_new, _, _, _ = np.linalg.lstsq(M_W, T_flat_w.T, rcond=None)
        W = W_new.T
        
        # Check convergence
        recon = _reconstruct(U, V, W)
        residual = np.max(np.abs(T - recon))
        if residual < tol:
            break
    
    return U, V, W


def _structured_init(n, m, p, rank, strategy=0):
    """Create structured initializations based on the tensor structure."""
    a_size = n * m
    b_size = m * p
    c_size = p * n
    U = np.zeros((a_size, rank))
    V = np.zeros((b_size, rank))
    W = np.zeros((c_size, rank))
    
    if strategy == 0:
        # Schoolbook-like: assign columns to (i,j,k) triples
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= rank:
                        break
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        # Fill remaining columns with small random values
        for r in range(col, rank):
            U[:, r] = np.random.randn(a_size) * 0.01
            V[:, r] = np.random.randn(b_size) * 0.01
            W[:, r] = np.random.randn(c_size) * 0.01
    elif strategy == 1:
        # Random with structure: each column picks one a, b, c
        for r in range(rank):
            a = np.random.randint(0, a_size)
            b = np.random.randint(0, b_size)
            c = np.random.randint(0, c_size)
            U[a, r] = 1.0
            V[b, r] = 1.0
            W[c, r] = 1.0
    elif strategy == 2:
        # Block structure: group by i
        for r in range(rank):
            i = r % n
            j = (r // n) % m
            k = (r // (n * m)) % p
            if i < n and j < m and k < p:
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, r] = 1.0
                V[b, r] = 1.0
                W[c, r] = 1.0
    elif strategy == 3:
        # Dense random
        U = np.random.randn(a_size, rank) * 0.1
        V = np.random.randn(b_size, rank) * 0.1
        W = np.random.randn(c_size, rank) * 0.1
    elif strategy == 4:
        # Sparse random: 2 non-zeros per column
        for r in range(rank):
            for _ in range(2):
                U[np.random.randint(0, a_size), r] = np.random.choice([-1, 1])
            for _ in range(2):
                V[np.random.randint(0, b_size), r] = np.random.choice([-1, 1])
            for _ in range(2):
                W[np.random.randint(0, c_size), r] = np.random.choice([-1, 1])
    
    return U, V, W


def _try_rank(T, n, m, p, rank, time_budget, seed=0):
    """Try to find rank-r decomposition with multiple strategies."""
    a_size = n * m
    b_size = m * p
    c_size = p * n
    start_time = time.time()
    deadline = start_time + time_budget
    
    np.random.seed(seed)
    random_strategies = [0, 1, 2, 3, 4, 0, 1, 3, 4, 2, 0, 3, 1, 4, 2, 3]
    
    for attempt in range(len(random_strategies)):
        if time.time() > deadline:
            break
        
        np.random.seed(seed * 100 + attempt)
        strategy = random_strategies[attempt]
        
        U, V, W = _structured_init(n, m, p, rank, strategy)
        
        # Run ALS with time check
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        
        U, V, W = _als_optimize(T, U, V, W, rank, max_iter=60, 
                                 tol=1e-10, time_deadline=deadline - 0.5)
        
        if _verify_exact(T, U, V, W, tol=1e-6):
            return U, V, W
        
        # Try a second phase with different init if time permits
        if time.time() < deadline - 5.0:
            np.random.seed(seed * 200 + attempt + 1000)
            U2, V2, W2 = _structured_init(n, m, p, rank, (strategy + 3) % 5)
            U2, V2, W2 = _als_optimize(T, U2, V2, W2, rank, max_iter=40,
                                        tol=1e-10, time_deadline=deadline - 0.5)
            if _verify_exact(T, U2, V2, W2, tol=1e-6):
                return U2, V2, W2
    
    return None


def _to_list(mat):
    """Convert numpy matrix to list of lists, rounding to int/half-int if close."""
    result = []
    for row in mat:
        row_list = []
        for val in row:
            nearest_int = round(val)
            nearest_half = round(2 * val) / 2.0
            if abs(val - nearest_int) < 1e-5:
                row_list.append(int(nearest_int))
            elif abs(val - nearest_half) < 1e-5:
                row_list.append(nearest_half)
            else:
                row_list.append(float(val))
        result.append(row_list)
    return result


def _schoolbook_decomp(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    schoolbook_rank = n * m * p  # 48
    time_start = time.time()
    time_limit = 140  # safety margin for 150s limit
    
    # Lower bound: max(n*p, m*p, n*m) = max(16, 24, 6) = 24
    # But we know rank >= n*p = 16, and for rectangular it's more complex
    # Try from a reasonable lower bound upward
    lower_bound = max(n * p, m * p, n * m)  # 24
    
    best_decomp = None
    best_rank = schoolbook_rank
    
    # First, try to find a good decomposition at schoolbook rank or below
    # Try ranks from lower_bound to schoolbook_rank - 1
    # Allocate time proportionally
    
    ranks_to_try = list(range(lower_bound, schoolbook_rank))
    
    for rank in ranks_to_try:
        elapsed = time.time() - time_start
        if elapsed > time_limit:
            break
        
        remaining = time_limit - elapsed
        # Give more time to lower ranks (more likely to be minimal)
        # But ensure we can at least try a few
        num_remaining = schoolbook_rank - rank
        per_rank_time = min(remaining / max(1, num_remaining), 15.0)
        per_rank_time = max(per_rank_time, 2.0)
        
        result = _try_rank(T, n, m, p, rank, per_rank_time, seed=rank * 7 + 13)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_decomp = (U, V, W)
            # Continue trying lower ranks if possible
            # But don't break - try to find even lower rank
            # Actually, we want the MINIMUM rank, so we should try lower first
            # Let's restructure: try from lowest rank up
    
    # Actually, let's restructure: try from lowest rank upward
    # Reset
    best_decomp = None
    best_rank = schoolbook_rank
    time_start = time.time()
    
    for rank in range(lower_bound, schoolbook_rank):
        elapsed = time.time() - time_start
        if elapsed > time_limit:
            break
        
        remaining = time_limit - elapsed
        num_remaining = schoolbook_rank - rank
        per_rank_time = remaining / max(1, num_remaining)
        per_rank_time = max(min(per_rank_time, 20.0), 3.0)
        
        result = _try_rank(T, n, m, p, rank, per_rank_time, seed=rank * 31 + 7)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_decomp = (U, V, W)
            break  # Found minimum rank, stop
    
    if best_decomp is not None:
        U, V, W = best_decomp
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    else:
        # Fall back to schoolbook
        U, V, W = _schoolbook_decomp(n, m, p)
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

# EVOLVE-BLOCK-END
