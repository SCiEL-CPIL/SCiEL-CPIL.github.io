import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def _build_schoolbook(n, m, p):
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

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Rank 7.
    Indices:
    a = i*2 + j
    b = j*2 + k
    g = k*2 + i
    """
    R = 7
    U = np.zeros((4, R))
    V = np.zeros((4, R))
    W = np.zeros((4, R))
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C11 += M1, C22 += M1
    # C11 is g=0 (k=0, i=0). C22 is g=3 (k=1, i=1).
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21 + A22) B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C21 += M2. C21 is g=2 (k=1, i=0)? No.
    # C_{i,k}. i=1, k=0 -> C10?
    # g = k*2 + i.
    # C_{1,0} (i=1, k=0) -> g = 0*2 + 1 = 1.
    # C_{0,1} (i=0, k=1) -> g = 1*2 + 0 = 2.
    # C_{1,1} (i=1, k=1) -> g = 1*2 + 1 = 3.
    # C_{0,0} (i=0, k=0) -> g = 0.
    
    # M2 contributes to C_{2,1} in 1-based -> C_{1,0} in 0-based?
    # Strassen: C21 = M2 + M4.
    # C21 is row 2, col 1 (1-based). i=1, k=0 (0-based).
    # g = k*2 + i = 0*2 + 1 = 1.
    W[1, 1] = 1
    
    # M3 = A11 (B12 - B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # C12 = M3 + M5.
    # C12 is i=0, k=1. g = 1*2 + 0 = 2.
    W[2, 2] = 1
    
    # M4 = A22 (B21 - B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # C11 += M4, C21 += M4.
    # C11 (i=0, k=0) -> g=0.
    # C21 (i=1, k=0) -> g=1.
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A11 + A12) B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C12 += M5, C11 -= M5.
    # C12 (i=0, k=1) -> g=2.
    # C11 (i=0, k=0) -> g=0.
    W[2, 4] = 1; W[0, 4] = -1
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C22 -= M6.
    # C22 (i=1, k=1) -> g=3.
    W[3, 5] = -1
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C11 += M7.
    # C11 (i=0, k=0) -> g=0.
    W[0, 6] = 1
    
    return U, V, W

def _combine_decompositions(U1, V1, W1, U2, V2, W2, n, m1, m2, p):
    """
    Combines two decompositions for (n, m1, p) and (n, m2, p) into one for (n, m1+m2, p).
    """
    R1 = U1.shape[1]
    R2 = U2.shape[1]
    R = R1 + R2
    
    U = np.zeros((n * (m1 + m2), R))
    V = np.zeros(((m1 + m2) * p, R))
    W = np.zeros((p * n, R))
    
    # U mapping
    # A1 rows: 0 .. n*m1 - 1
    # A2 rows: n*m1 .. n*(m1+m2) - 1
    # U1 has rows indexed by a1 = i*m1 + j.
    # U2 has rows indexed by a2 = i*m2 + j.
    
    for i in range(n):
        for j in range(m1):
            a_global = i * (m1 + m2) + j
            a_local = i * m1 + j
            U[a_global, :R1] = U1[a_local, :]
            
        for j in range(m2):
            a_global = i * (m1 + m2) + (m1 + j)
            a_local = i * m2 + j
            U[a_global, R1:] = U2[a_local, :]
            
    # V mapping
    # B1 rows: 0 .. m1*p - 1
    # B2 rows: m1*p .. (m1+m2)*p - 1
    # V1 has rows indexed by b1 = j*p + k.
    # V2 has rows indexed by b2 = j*p + k.
    
    for j in range(m1):
        for k in range(p):
            b_global = j * p + k
            b_local = j * p + k
            V[b_global, :R1] = V1[b_local, :]
            
    for j in range(m2):
        for k in range(p):
            b_global = (m1 + j) * p + k
            b_local = j * p + k
            V[b_global, R1:] = V2[b_local, :]
            
    # W mapping
    # W is concatenated horizontally.
    W[:, :R1] = W1
    W[:, R1:] = W2
    
    return U, V, W

def _combine_split_n(U1, V1, W1, U2, V2, W2, n1, m, p):
    """Combine decompositions for (n1,m,p) and (n2,m,p) into (n1+n2,m,p)."""
    n2 = (U2.shape[0]) // m
    n = n1 + n2
    R1 = U1.shape[1]
    R2 = U2.shape[1]
    R = R1 + R2
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # U: stack vertically
    U[:n1*m, :R1] = U1
    U[n1*m:, R1:] = U2
    
    # V: concatenate horizontally (same B matrix)
    V[:, :R1] = V1
    V[:, R1:] = V2
    
    # W: stack vertically. g = k*n + i.
    # For first block (i in 0..n1-1): g = k*n + i, i < n1
    # For second block (i in n1..n-1): g = k*n + i, i >= n1
    # W1 has rows indexed by g1 = k*n1 + i (i in 0..n1-1)
    # W2 has rows indexed by g2 = k*n2 + i (i in 0..n2-1)
    # In combined: for k in 0..p-1, i in 0..n-1:
    #   if i < n1: use W1[k*n1 + i]
    #   else: use W2[k*n2 + (i-n1)]
    for k in range(p):
        for i in range(n1):
            W[k * n + i, :R1] = W1[k * n1 + i, :]
        for i in range(n2):
            W[k * n + (n1 + i), R1:] = W2[k * n2 + i, :]
    
    return U, V, W

def _combine_split_p(U1, V1, W1, U2, V2, W2, n, m, p1):
    """Combine decompositions for (n,m,p1) and (n,m,p2) into (n,m,p1+p2)."""
    p2 = (V2.shape[0]) // m
    p = p1 + p2
    R1 = U1.shape[1]
    R2 = U2.shape[1]
    R = R1 + R2
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # U: concatenate horizontally
    U[:, :R1] = U1
    U[:, R1:] = U2
    
    # V: stack vertically. b = j*p + k.
    # V1 rows: b1 = j*p1 + k (k in 0..p1-1)
    # V2 rows: b2 = j*p2 + k (k in 0..p2-1)
    # Combined: for j in 0..m-1, k in 0..p-1:
    #   if k < p1: use V1[j*p1 + k]
    #   else: use V2[j*p2 + (k-p1)]
    for j in range(m):
        for k in range(p1):
            V[j * p + k, :R1] = V1[j * p1 + k, :]
        for k in range(p2):
            V[j * p + (p1 + k), R1:] = V2[j * p2 + k, :]
    
    # W: stack vertically. g = k*n + i.
    for k in range(p1):
        for i in range(n):
            W[k * n + i, :R1] = W1[k * n + i, :]
    for k in range(p2):
        for i in range(n):
            W[(p1 + k) * n + i, R1:] = W2[k * n + i, :]
    
    return U, V, W

_rec_cache = {}

def _build_recursive(n, m, p):
    """
    Recursively builds a low-rank decomposition using Strassen-like splits.
    """
    if (n, m, p) in _rec_cache:
        return _rec_cache[(n, m, p)]
    
    if n == 1 and m == 1 and p == 1:
        U = np.array([[1.0]])
        V = np.array([[1.0]])
        W = np.array([[1.0]])
        _rec_cache[(n, m, p)] = (U, V, W)
        return U, V, W
        
    if n == 2 and m == 2 and p == 2:
        result = _strassen_2x2()
        _rec_cache[(n, m, p)] = result
        return result
    
    # Base cases: when one dimension is 1, schoolbook is optimal
    if n == 1:
        result = _build_schoolbook(n, m, p)
        _rec_cache[(n, m, p)] = result
        return result
    if m == 1:
        result = _build_schoolbook(n, m, p)
        _rec_cache[(n, m, p)] = result
        return result
    if p == 1:
        result = _build_schoolbook(n, m, p)
        _rec_cache[(n, m, p)] = result
        return result
    
    # Try all possible splits of each dimension and pick the one with minimum total rank
    best_rank = float('inf')
    best_result = None
    
    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1 = _build_recursive(n, m1, p)
        U2, V2, W2 = _build_recursive(n, m2, p)
        r = U1.shape[1] + U2.shape[1]
        if r < best_rank:
            best_rank = r
            best_result = _combine_decompositions(U1, V1, W1, U2, V2, W2, n, m1, m2, p)
    
    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1 = _build_recursive(n1, m, p)
        U2, V2, W2 = _build_recursive(n2, m, p)
        r = U1.shape[1] + U2.shape[1]
        if r < best_rank:
            best_rank = r
            best_result = _combine_split_n(U1, V1, W1, U2, V2, W2, n1, m, p)
    
    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1 = _build_recursive(n, m, p1)
        U2, V2, W2 = _build_recursive(n, m, p2)
        r = U1.shape[1] + U2.shape[1]
        if r < best_rank:
            best_rank = r
            best_result = _combine_split_p(U1, V1, W1, U2, V2, W2, n, m, p1)
    
    if best_result is not None:
        _rec_cache[(n, m, p)] = best_result
        return best_result
    
    # Fallback: schoolbook
    result = _build_schoolbook(n, m, p)
    _rec_cache[(n, m, p)] = result
    return result

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the tensor decomposition reconstructs matrix multiplication exactly."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    # Check zero constraints (sample)
    import random
    random.seed(42)
    for _ in range(1000):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        g = random.randint(0, p*n-1)
        i = a // m; j = a % m
        jp = b // p; k = b % p
        kn = g // n; ip = g % n
        if i != ip or j != jp or k != kn:
            val = 0.0
            for r in range(rank):
                val += U[a, r] * V[b, r] * W[g, r]
            if abs(val) > tol:
                return False
    return True

def _round_to_half_integer(M):
    """Round matrix entries to nearest 0.5."""
    return np.round(M * 2) / 2

def _optimize_decomposition_fast(n, m, p, initial_rank=None, max_time=100):
    """
    Fast Alternating Least Squares using pre-computed Gram matrices.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    if initial_rank is None:
        initial_rank = n * m * p
    
    # Try decreasing ranks
    for target_rank in range(initial_rank - 1, max(10, initial_rank - 50), -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for seed in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(seed)
            
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Pre-compute indices for fast t-vector calculation
            # For each a, store the list of (b, g) pairs where t=1
            # a = i*m + j
            # b = j*p + k
            # g = k*n + i
            # For fixed a (i,j), k varies 0..p-1
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

            # Similarly for b and g
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

            # Alternating least squares
            for iteration in range(50):
                if time.time() > deadline:
                    break
                
                # Update U
                # M^T M = (V^T V) o (W^T W)
                G_V = V.T @ V
                G_W = W.T @ W
                A_mat = G_V * G_W # Hadamard product
                # Regularize slightly to avoid singularity
                A_mat += 1e-6 * np.eye(target_rank)
                
                for a in range(n * m):
                    # Compute b_vec = sum_k V[b_k, :] * W[g_k, :]
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
                A_mat += 1e-6 * np.eye(target_rank)
                
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
                A_mat += 1e-6 * np.eye(target_rank)
                
                for g in range(p * n):
                    b_vec = np.zeros(target_rank)
                    for a, b in g_to_ab[g]:
                        b_vec += U[a, :] * V[b, :]
                    
                    try:
                        W[g, :] = np.linalg.solve(A_mat, b_vec)
                    except:
                        pass
                
                # Check convergence and round
                if iteration % 10 == 0:
                    # Calculate error on 1-positions
                    error = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i*m+j; b = j*p+k; g = k*n+i
                                val = np.sum(U[a, :] * V[b, :] * W[g, :])
                                error += (val - 1.0) ** 2
                    
                    # Sample zero constraints
                    np.random.seed(42)
                    for _ in range(200):
                        a = np.random.randint(0, n*m)
                        b = np.random.randint(0, m*p)
                        g = np.random.randint(0, p*n)
                        i = a // m; j = a % m
                        jp = b // p; k = b % p
                        kn = g // n; ip = g % n
                        if i != ip or j != jp or k != kn:
                            val = np.sum(U[a, :] * V[b, :] * W[g, :])
                            error += val ** 2
                    
                    if error < 1e-6:
                        # Try rounding
                        U_rounded = _round_to_half_integer(U)
                        V_rounded = _round_to_half_integer(V)
                        W_rounded = _round_to_half_integer(W)
                        
                        if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                            return U_rounded, V_rounded, W_rounded
    
    return None

def _greedy_set_cover(n, m, p, max_candidates=2000, max_time=120):
    """
    Greedy set-cover approach for tensor rank minimization.
    
    The matrix multiplication tensor T has T[i,j,k] = 1 for all valid (i,j,k).
    A rank-1 tensor is u * v * w where u, v, w are vectors.
    For the MM tensor, any rank-1 component must have support that is a
    "rectangle" in the 3D index space.
    
    We generate candidate rank-1 tensors with sparse support (at most one
    non-zero per factor dimension slice) and greedily select the minimum
    number to cover all required tensor entries.
    """
    start = time.time()
    deadline = start + max_time
    
    # The MM tensor has 1s at positions (i,j,k) for all i,j,k
    # and 0s elsewhere. We need to represent it as sum of rank-1 tensors.
    # 
    # Key insight: each rank-1 tensor u*v*w contributes u[i]*v[j]*w[k] to T[i,j,k].
    # For the MM tensor, we need the sum to equal 1 at all (i,j,k) positions.
    #
    # A sparse rank-1 tensor with support S_u x S_v x S_w where |S_u|=1, |S_v|=1, |S_w|=1
    # is just a single entry. But we can have larger supports.
    #
    # Strategy: Consider rank-1 tensors where each factor has limited support.
    # For factor U: support is a set of (i,j) indices.
    # For factor V: support is a set of (j,k) indices.
    # For factor W: support is a set of (k,i) indices.
    #
    # The product u[i,j]*v[j,k]*w[k,i] contributes to T[i,j,k] only if
    # (i,j) in supp(u), (j,k) in supp(v), (k,i) in supp(w).
    #
    # For a valid contribution to T[i,j,k]=1, we need the triple (i,j,k) to be
    # "covered" by the intersection pattern.
    #
    # Let's think about it differently. The MM tensor is:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # T[a,b,g] = 0 otherwise (where a,b,g don't form valid (i,j,k))
    #
    # Actually, the tensor is defined on indices a=i*m+j, b=j*p+k, g=k*n+i.
    # T[a,b,g] = 1 if a,b,g form a valid triple (i,j,k), else 0.
    #
    # Number of 1-entries: n*m*p = 120
    # Total tensor size: (n*m)*(m*p)*(p*n) = 20*30*24 = 14400
    #
    # For a rank-1 tensor with U vector (length 20), V vector (length 30), W vector (length 24):
    # It contributes U[a]*V[b]*W[g] to position (a,b,g).
    #
    # We need sum_r U_r[a] * V_r[b] * W_r[g] = 1 for valid triples, 0 for invalid.
    #
    # Greedy approach:
    # 1. Generate candidate rank-1 tensors
    # 2. For each candidate, compute which valid (i,j,k) triples it covers
    # 3. Greedily select candidates to cover all 120 required entries
    #
    # A "covers" relationship: candidate r covers triple (i,j,k) if
    # U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = 1 (or some specific value)
    #
    # Actually, we need the SUM to equal 1, not individual terms.
    # This makes set-cover less straightforward.
    #
    # Alternative: Use the fact that for half-integer entries, we can try
    # specific patterns. Let's generate candidates where each factor has
    # a small number of non-zero entries.
    
    # Generate candidate rank-1 tensors
    # Each candidate is (U_vec, V_vec, W_vec) where each vec has limited support
    
    candidates = []
    
    # Type 1: Single entry in each factor
    # U has one 1, V has one 1, W has one 1
    # This covers exactly one triple if they're consistent
    for i in range(n):
        for j in range(m):
            a = i * m + j
            for k in range(p):
                b = j * p + k
                g = k * n + i
                U_vec = np.zeros(n * m)
                V_vec = np.zeros(m * p)
                W_vec = np.zeros(p * n)
                U_vec[a] = 1
                V_vec[b] = 1
                W_vec[g] = 1
                candidates.append((U_vec, V_vec, W_vec, [(i, j, k)]))
    
    # Type 2: U has two entries (same i, different j), V has two entries (same j, different k), W has two entries (same k, different i)
    # This creates a 2x2x2 block pattern
    for i in range(n):
        for j1 in range(m):
            for j2 in range(j1 + 1, m):
                for k in range(p):
                    # U: (i,j1) and (i,j2)
                    # V: (j1,k) and (j2,k)
                    # W: (k,i) - only one entry needed
                    # This gives contributions to (i,j1,k) and (i,j2,k)
                    a1 = i * m + j1
                    a2 = i * m + j2
                    b1 = j1 * p + k
                    b2 = j2 * p + k
                    g = k * n + i
                    U_vec = np.zeros(n * m)
                    V_vec = np.zeros(m * p)
                    W_vec = np.zeros(p * n)
                    U_vec[a1] = 1
                    U_vec[a2] = 1
                    V_vec[b1] = 1
                    V_vec[b2] = 1
                    W_vec[g] = 1
                    candidates.append((U_vec, V_vec, W_vec, [(i, j1, k), (i, j2, k)]))
    
    # Type 3: Full row in U (all j for fixed i), corresponding in V and W
    for i in range(n):
        U_vec = np.zeros(n * m)
        V_vec = np.zeros(m * p)
        W_vec = np.zeros(p * n)
        for j in range(m):
            U_vec[i * m + j] = 1
        for j in range(m):
            for k in range(p):
                V_vec[j * p + k] = 1
        for k in range(p):
            W_vec[k * n + i] = 1
        covered = [(i, j, k) for j in range(m) for k in range(p)]
        candidates.append((U_vec, V_vec, W_vec, covered))
    
    # Type 4: Full column in V (all k for fixed j)
    for j in range(m):
        U_vec = np.zeros(n * m)
        V_vec = np.zeros(m * p)
        W_vec = np.zeros(p * n)
        for i in range(n):
            U_vec[i * m + j] = 1
        for k in range(p):
            V_vec[j * p + k] = 1
        for i in range(n):
            for k in range(p):
                W_vec[k * n + i] = 1
        covered = [(i, j, k) for i in range(n) for k in range(p)]
        candidates.append((U_vec, V_vec, W_vec, covered))
    
    # Type 5: Full "slice" in W (all i for fixed k)
    for k in range(p):
        U_vec = np.zeros(n * m)
        V_vec = np.zeros(m * p)
        W_vec = np.zeros(p * n)
        for i in range(n):
            for j in range(m):
                U_vec[i * m + j] = 1
        for j in range(m):
            V_vec[j * p + k] = 1
        for i in range(n):
            W_vec[k * n + i] = 1
        covered = [(i, j, k) for i in range(n) for j in range(m)]
        candidates.append((U_vec, V_vec, W_vec, covered))
    
    # Type 6: 2x2 Strassen-like patterns for sub-blocks
    # For 2x2x2 blocks within the larger matrix
    for i1 in range(0, n - 1):
        for j1 in range(0, m - 1):
            for k1 in range(0, p - 1):
                # 2x2x2 block starting at (i1, j1, k1)
                i2 = i1 + 1
                j2 = j1 + 1
                k2 = k1 + 1
                
                # Strassen's 7 multiplications for 2x2x2
                # M1 = (A11+A22)(B11+B22)
                U1 = np.zeros(n * m)
                V1 = np.zeros(m * p)
                W1 = np.zeros(p * n)
                U1[i1*m+j1] = 1; U1[i2*m+j2] = 1
                V1[j1*p+k1] = 1; V1[j2*p+k2] = 1
                W1[k1*n+i1] = 1; W1[k2*n+i2] = 1
                covered1 = [(i1,j1,k1), (i1,j1,k2), (i2,j2,k1), (i2,j2,k2)]
                candidates.append((U1, V1, W1, covered1))
                
                # M2 = (A21+A22)B11
                U2 = np.zeros(n * m)
                V2 = np.zeros(m * p)
                W2 = np.zeros(p * n)
                U2[i2*m+j1] = 1; U2[i2*m+j2] = 1
                V2[j1*p+k1] = 1
                W2[k1*n+i2] = 1
                covered2 = [(i2,j1,k1)]
                candidates.append((U2, V2, W2, covered2))
                
                # M3 = A11(B12-B22)
                U3 = np.zeros(n * m)
                V3 = np.zeros(m * p)
                W3 = np.zeros(p * n)
                U3[i1*m+j1] = 1
                V3[j1*p+k2] = 1; V3[j2*p+k2] = -1
                W3[k2*n+i1] = 1
                covered3 = [(i1,j1,k2)]
                candidates.append((U3, V3, W3, covered3))
                
                # M4 = A22(B21-B11)
                U4 = np.zeros(n * m)
                V4 = np.zeros(m * p)
                W4 = np.zeros(p * n)
                U4[i2*m+j2] = 1
                V4[j2*p+k1] = 1; V4[j1*p+k1] = -1
                W4[k1*n+i1] = 1; W4[k1*n+i2] = 1
                covered4 = [(i1,j1,k1), (i2,j1,k1)]
                candidates.append((U4, V4, W4, covered4))
                
                # M5 = (A11+A12)B22
                U5 = np.zeros(n * m)
                V5 = np.zeros(m * p)
                W5 = np.zeros(p * n)
                U5[i1*m+j1] = 1; U5[i1*m+j2] = 1
                V5[j2*p+k2] = 1
                W5[k2*n+i1] = 1; W5[k1*n+i1] = -1
                covered5 = [(i1,j1,k2), (i1,j1,k1)]
                candidates.append((U5, V5, W5, covered5))
                
                # M6 = (A21-A11)(B11+B12)
                U6 = np.zeros(n * m)
                V6 = np.zeros(m * p)
                W6 = np.zeros(p * n)
                U6[i2*m+j1] = 1; U6[i1*m+j1] = -1
                V6[j1*p+k1] = 1; V6[j1*p+k2] = 1
                W6[k2*n+i2] = -1
                covered6 = [(i2,j2,k2)]
                candidates.append((U6, V6, W6, covered6))
                
                # M7 = (A12-A22)(B21+B22)
                U7 = np.zeros(n * m)
                V7 = np.zeros(m * p)
                W7 = np.zeros(p * n)
                U7[i1*m+j2] = 1; U7[i2*m+j2] = -1
                V7[j2*p+k1] = 1; V7[j2*p+k2] = 1
                W7[k1*n+i1] = 1
                covered7 = [(i1,j1,k1)]
                candidates.append((U7, V7, W7, covered7))
    
    if time.time() > deadline:
        return None
    
    # All required triples
    all_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    triple_set = set(all_triples)
    
    # For each candidate, determine which triples it "contributes to"
    # A candidate contributes to triple (i,j,k) if the product U[i*m+j]*V[j*p+k]*W[k*n+i] != 0
    # and the sign pattern is consistent.
    
    # Actually, for the greedy set cover to work, we need candidates that
    # can be combined (summed) to give exactly 1 at each required position.
    # This is more complex than simple set cover.
    
    # Let's use a different approach: iterative refinement
    # Start with schoolbook, then try to merge/eliminate terms
    
    # Alternative: Use the candidates to build a linear system
    # and find the minimum number of candidates whose weighted sum gives the target
    
    # For now, let's try a simpler greedy: 
    # Select candidates that cover the most uncovered triples
    
    uncovered = set(all_triples)
    selected = []
    
    # Sort candidates by number of triples they cover (descending)
    # But we need to account for the actual values, not just coverage
    
    # Let's try a direct approach: for each candidate, check if adding it
    # helps reduce the error when we have a current decomposition
    
    # Actually, let's try yet another approach:
    # Start with the full tensor and try to express it as sum of few rank-1 terms
    # using a greedy algorithm that picks the best rank-1 approximation at each step
    
    # Compute the MM tensor as a 3D array
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                g = k * n + i
                T[a, b, g] = 1.0
    
    # Greedy rank-1 approximation
    residual = T.copy()
    rank1_terms = []
    
    for step in range(min(100, len(all_triples))):
        if time.time() > deadline:
            break
        
        if np.allclose(residual, 0, atol=1e-10):
            break
        
        # Find the best rank-1 approximation to the residual
        # This is an NP-hard problem in general, but we can use
        # power iteration or try the pre-generated candidates
        
        best_candidate = None
        best_error = float('inf')
        
        # Try all pre-generated candidates
        for U_c, V_c, W_c, _ in candidates:
            # Compute the rank-1 tensor from this candidate
            R1 = np.outer(U_c, V_c)
            R1 = R1 * W_c[np.newaxis, np.newaxis, :]
            
            # Compute error if we subtract this from residual
            # We want to minimize ||residual - alpha * R1|| for some alpha
            # Optimal alpha = <residual, R1> / ||R1||^2
            
            inner = np.sum(residual * R1)
            norm_sq = np.sum(R1 * R1)
            
            if norm_sq < 1e-10:
                continue
            
            alpha = inner / norm_sq
            
            # New residual
            new_residual = residual - alpha * R1
            error = np.sum(new_residual * new_residual)
            
            if error < best_error:
                best_error = error
                best_candidate = (U_c, V_c, W_c, alpha)
        
        if best_candidate is None:
            break
        
        U_c, V_c, W_c, alpha = best_candidate
        R1 = np.outer(U_c, V_c)
        R1 = R1 * W_c[np.newaxis, np.newaxis, :]
        
        residual = residual - alpha * R1
        rank1_terms.append((U_c, V_c, W_c, alpha))
        
        # Check if we're making progress
        if best_error >= 0.9999 * np.sum(residual * residual + np.sum(R1 * R1)):
            # Not making much progress
            break
    
    if len(rank1_terms) == 0:
        return None
    
    # Build the decomposition from the rank-1 terms
    rank = len(rank1_terms)
    U_out = np.zeros((n * m, rank))
    V_out = np.zeros((m * p, rank))
    W_out = np.zeros((p * n, rank))
    
    for r, (U_c, V_c, W_c, alpha) in enumerate(rank1_terms):
        U_out[:, r] = alpha * U_c
        V_out[:, r] = V_c
        W_out[:, r] = W_c
    
    # Try to round to half-integers
    U_rounded = _round_to_half_integer(U_out)
    V_rounded = _round_to_half_integer(V_out)
    W_rounded = _round_to_half_integer(W_out)
    
    # Verify
    if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
        return U_rounded, V_rounded, W_rounded
    
    # If rounding fails, try without rounding
    if _verify_decomposition(U_out, V_out, W_out, n, m, p, tol=1e-6):
        return U_out, V_out, W_out
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    best_rank = best_U.shape[1]
    
    # Try recursive Strassen-like decomposition
    try:
        U_rec, V_rec, W_rec = _build_recursive(n, m, p)
        if _verify_decomposition(U_rec, V_rec, W_rec, n, m, p):
            if U_rec.shape[1] < best_rank:
                best_U, best_V, best_W = U_rec, V_rec, W_rec
                best_rank = U_rec.shape[1]
        elif U_rec.shape[1] < best_rank:
            # Even if verification fails, check with relaxed tolerance
            if _verify_decomposition(U_rec, V_rec, W_rec, n, m, p, tol=1e-6):
                best_U, best_V, best_W = U_rec, V_rec, W_rec
                best_rank = U_rec.shape[1]
    except Exception as e:
        pass
    
    # Try greedy set-cover approach
    if time.time() < deadline - 5:
        try:
            U_gc, V_gc, W_gc = _greedy_set_cover(n, m, p, max_time=130)
            if U_gc is not None and U_gc.shape[1] < best_rank:
                best_U, best_V, best_W = U_gc, V_gc, W_gc
                best_rank = U_gc.shape[1]
        except Exception as e:
            pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
