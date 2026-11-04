import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is exact.
    
    The tensor T[i, j, k] for matrix multiplication A(n x m) * B(m x p) -> C(n x p)
    is: T[i, j, k, a, b, c, d] = 1 iff a==i, b==c, d==k
    (where a is row of A, b is col of A / row of B, c is row of B, d is col of B)
    
    Actually, the standard decomposition: C[i,k] = sum_j A[i,j] * B[j,k]
    So the tensor has non-zero entries at T[i, j, k, i, j, k] = 1
    (where we index A by (i,j), B by (j,k), C by (i,k))
    
    For the decomposition U, V, W:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal
    1 if we're looking at the "diagonal" term where the indices match appropriately.
    
    More precisely: For the matrix multiplication tensor, we need:
    sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c]
    where a indexes A's flattened (i,j) with i in [0,n), j in [0,m)
         b indexes B's flattened (j,k) with j in [0,m), k in [0,p)
         c indexes C's flattened (k,i) with k in [0,p), i in [0,n)  [using W[k*n+i,r] convention]
    
    And T[a, b, c] = 1 iff:
      Let (i, j) = (a // m, a % m)  -- A index
      Let (j2, k) = (b // p, b % p) -- B index
      Let (k2, i2) = (c // n, c % n) -- C index (W convention)
      Then T = 1 iff i == i2, j == j2, k == k2
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    # Check all entries
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            j2 = b // p
            k = b % p
            for c in range(p * n):
                k2 = c // n
                i2 = c % n
                
                expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                
                # Compute sum_r U[a,r] * V[b,r] * W[c,r]
                actual = np.dot(U_arr[a] * V_arr[b], W_arr[c])
                
                if abs(actual - expected) > tol:
                    return False
    return True

def _make_schoolbook(n, m, p):
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

def _strassen_block(U, V, W, n_offset, m_offset, p_offset, r_offset, size):
    """
    Add Strassen 2x2 block multiplication starting at given offsets.
    size should be 2 for standard Strassen.
    Returns the next r_offset.
    """
    if size == 2:
        # Split into 2x2 blocks
        # A = A11 A12; A21 A22  (n x m -> each block (n/2) x (m/2))
        # B = B11 B12; B21 B22  (m x p -> each block (m/2) x (p/2))
        # C = C11 C12; C21 C22  (n x p -> each block (n/2) x (p/2))
        
        # For simplicity, handle 2x2 case directly
        # A11 at (n_offset, m_offset), A12 at (n_offset, m_offset+1)
        # A21 at (n_offset+1, m_offset), A22 at (n_offset+1, m_offset+1)
        # etc.
        
        ns = n_offset  # n/2 = 1 (since total n=2, each half is 1)
        ms = m_offset  # m/2
        ps = p_offset  # p/2
        
        # Actually for n=2, m=5, p=5, this gets complicated.
        # Let's just do schoolbook for now and try other optimizations.
        pass
    
    return r_offset

def _try_strassen_2x2_2x2(n, m, p):
    """
    Try to use Strassen for 2x2 * 2x2 subproblems.
    For n=2, m=5, p=5:
    Split B into blocks along rows: 2+2+1 and along cols: 2+2+1
    A is 2x5, so along cols: 2+2+1
    
    This is complex. Let's try a simpler approach first.
    """
    if n != 2:
        return None
    
    # Schoolbook gives rank 50
    # Let's try to find a better decomposition
    
    # One approach: For each of the 5 columns of A and 5 rows of B,
    # we have 2*5 = 10 products. Can we do better?
    
    # Actually, let's try a direct optimization approach.
    # The bilinear complexity of (2,5,5) is at most 50 (schoolbook).
    # Known lower bound: For m x n matrix multiplication, rank >= 2mn - m - n
    # For (2,5,5): 2*2*5 - 2 - 5 = 20 - 7 = 13? That's not right.
    # Lower bound: rank >= m + n - 1 for (1,m,n). For general, it's more complex.
    
    # Let's just return schoolbook for now.
    return _make_schoolbook(n, m, p)

def _try_random_search(n, m, p, R, num_trials=100, seed=42):
    """
    Try to find a rank-R decomposition using random initialization and verification.
    This is a heuristic search.
    """
    rng = random.Random(seed)
    
    for trial in range(num_trials):
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        for r in range(R):
            # Each rank-1 term should contribute to some subset of the tensor
            # Initialize with small random values
            U[:, r] = rng.choice([-1, 0, 1], size=n * m)
            V[:, r] = rng.choice([-1, 0, 1], size=m * p)
            W[:, r] = rng.choice([-1, 0, 1], size=p * n)
        
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_greedy_construction(n, m, p, R):
    """
    Greedily construct a rank-R decomposition.
    Start with rank-0 and add rank-1 terms to satisfy tensor entries.
    """
    if n != 2:
        return None
    
    # The tensor has exactly n*m*p = 50 non-zero entries (all = 1)
    # at positions (i*m+j, j*p+k, k*n+i) for all i,j,k
    
    # Greedy: for each unsatisfied tensor entry, add a rank-1 term that covers it
    # and as many other entries as possible
    
    # Create a set of all tensor entries that need to be satisfied
    needed = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                needed.add((a, b, c))
    
    # Create a set of entries that should be zero (all others)
    # We'll track which entries are currently satisfied
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # For each rank term, try to find a pattern that covers multiple needed entries
    # A rank-1 term (u, v, w) satisfies entry (a, b, c) if u[a]*v[b]*w[c] contributes to it
    
    # This is a set cover problem, which is hard. Let's try a simpler approach.
    
    # Just do schoolbook for now.
    return _make_schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < deadline
    
    # Start with schoolbook
    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    random.seed(12345)
    
    # Try to improve below schoolbook
    for R in range(20, best_rank):
        if not still_going():
            break
        
        # Try random search
        result = _try_random_search(n, m, p, R, num_trials=50, seed=R * 100)
        if result is not None:
            best_rank = R
            best = result
            break  # Found the minimal rank
    
    return best
