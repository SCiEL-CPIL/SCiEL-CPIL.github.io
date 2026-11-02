"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 135  # Leave 15s margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    T_target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Trivial schoolbook decomposition (rank = n*m*p = 105)
    rank_trivial = n * m * p
    U_trivial = np.zeros((nm, rank_trivial), dtype=np.float64)
    V_trivial = np.zeros((mp, rank_trivial), dtype=np.float64)
    W_trivial = np.zeros((pn, rank_trivial), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, col] = 1
                V_trivial[b, col] = 1
                W_trivial[c, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_trivial.copy(), V_trivial.copy(), W_trivial.copy()
    best_rank = rank_trivial
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        T_computed = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_computed, T_target, atol=1e-9)
    
    def try_decomposition(U, V, W, target_rank):
        """Try to verify and update best if valid."""
        nonlocal best_U, best_V, best_W, best_rank
        if verify(U, V, W):
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = target_rank
            return True
        return False
    
    # Strategy 1: Structured decompositions using block structure
    # For matrix multiplication A(n×m) * B(m×p) = C(n×p),
    # we can exploit the fact that T[i,j,k] = delta_{i,j} * delta_{j,k}
    
    # Laderman-style: try to find structured patterns
    # First, try a Kronecker-product-like approach
    # Decompose indices into blocks
    
    # Strategy 2: Systematic rank reduction via structured search
    # The tensor T[a,b,c] = 1 iff a%m == b//p == c%n (with proper indexing)
    # Actually: a = i*m+j, b = j*p+k, c = k*n+i
    # So T[a,b,c] = 1 iff (a%m) == (b//p) == (c%n) ... let me recheck
    # a = i*m + j -> a mod m = j, a // m = i
    # b = j*p + k -> b mod p = k, b // p = j  
    # c = k*n + i -> c mod n = i, c // n = k
    # So T[a,b,c] = 1 iff (a%m) == (b//p) AND (b%p) == (c//n) AND (c%n) == (a//m)
    # Which simplifies to: the "middle" index j must match, and i, k must be consistent
    
    # Try structured decompositions based on index factorization
    # For 3×5×7: n=3, m=5, p=7
    
    # Attempt 1: Use the structure to build a better-than-schoolbook decomposition
    # Group terms by shared indices
    
    # Key insight: T[a,b,c] = sum over j of (a%m==j)*(b//p==j) * (b%p)*(c//n) * (c%n)*(a//m)
    # This suggests a decomposition where we sum over j
    
    # Let's try: for each j in 0..m-1, we need to handle the i-k coupling
    # T[a,b,c] = sum_j [ (a%m==j) * (b//p==j) * (b%p) * (c//n) * (c%n) * (a//m) ]
    
    # This can be written as:
    # U[a, (j,i)] = 1 if a%m==j and a//m==i
    # V[b, (j,k)] = 1 if b//p==j and b%p==k
    # W[c, (i,k)] = 1 if c%n==i and c//n==k
    # But this gives rank = m * n * p = 105 (same as schoolbook)
    
    # To reduce rank, we need to find linear dependencies or use algebraic tricks
    
    # Strategy 3: Randomized search with structured initialization
    rng = np.random.default_rng(42)
    
    # Try ranks in a focused range
    # Lower bound: max(nm, mp, pn) = 35
    # Upper bound: 105 (schoolbook)
    # Realistic minimum for 3x5x7 is likely around 50-80
    
    # Try a range of ranks with multiple random trials
    # Focus on ranks where we might find solutions
    
    ranks_to_try = list(range(35, 106))
    
    # Prioritize lower ranks but also try some higher ones for safety
    # Actually, let's try from low to high and stop when we find one
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 30:
            break
        
        if target_rank >= best_rank:
            break
        
        # Number of trials: more for lower ranks (harder to find)
        if target_rank < 50:
            max_trials = 100
        elif target_rank < 70:
            max_trials = 50
        elif target_rank < 90:
            max_trials = 30
        else:
            max_trials = 20
        
        found = False
        for trial in range(max_trials):
            if time.time() > deadline - 15:
                break
            
            # Generate random sparse matrices with small integer entries
            # Use a mix of dense and sparse patterns
            if trial % 3 == 0:
                # Dense with small values
                U = rng.integers(-2, 3, size=(nm, target_rank)).astype(np.float64)
                V = rng.integers(-2, 3, size=(mp, target_rank)).astype(np.float64)
                W = rng.integers(-2, 3, size=(pn, target_rank)).astype(np.float64)
            elif trial % 3 == 1:
                # Sparse with {-1, 0, 1}
                U = rng.integers(-1, 2, size=(nm, target_rank)).astype(np.float64)
                V = rng.integers(-1, 2, size=(mp, target_rank)).astype(np.float64)
                W = rng.integers(-1, 2, size=(pn, target_rank)).astype(np.float64)
                # Add sparsity
                for mat in [U, V, W]:
                    mask = rng.random(mat.shape) < 0.4
                    mat[~mask] = 0
            else:
                # Very sparse
                U = np.zeros((nm, target_rank), dtype=np.float64)
                V = np.zeros((mp, target_rank), dtype=np.float64)
                W = np.zeros((pn, target_rank), dtype=np.float64)
                for r in range(target_rank):
                    # Each rank-1 term touches a few entries
                    u_idx = rng.choice(nm, size=rng.integers(2, 5), replace=False)
                    v_idx = rng.choice(mp, size=rng.integers(2, 5), replace=False)
                    w_idx = rng.choice(pn, size=rng.integers(2, 5), replace=False)
                    U[u_idx, r] = rng.choice([-1, 1])
                    V[v_idx, r] = rng.choice([-1, 1])
                    W[w_idx, r] = rng.choice([-1, 1])
            
            if try_decomposition(U, V, W, target_rank):
                found = True
                break
        
        if found:
            # Found a decomposition at this rank; try to go lower
            continue
    
    # Strategy 4: Try to improve by removing redundant terms from current best
    if best_rank < rank_trivial and time.time() < deadline - 40:
        # Try to find a lower rank by perturbing and re-solving
        for attempt in range(20):
            if time.time() > deadline - 20:
                break
            
            target_rank = max(35, best_rank - rng.integers(1, 5))
            
            # Initialize near the current solution but truncated
            U_init = best_U[:, :target_rank].copy()
            V_init = best_V[:, :target_rank].copy()
            W_init = best_W[:, :target_rank].copy()
            
            # Add random perturbation
            U_pert = U_init + rng.normal(0, 0.1, size=(nm, target_rank))
            V_pert = V_init + rng.normal(0, 0.1, size=(mp, target_rank))
            W_pert = W_init + rng.normal(0, 0.1, size=(pn, target_rank))
            
            # Round to nearest half-integer
            U_pert = np.round(U_pert * 2) / 2
            V_pert = np.round(V_pert * 2) / 2
            W_pert = np.round(W_pert * 2) / 2
            
            if try_decomposition(U_pert, V_pert, W_pert, target_rank):
                break
    
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
