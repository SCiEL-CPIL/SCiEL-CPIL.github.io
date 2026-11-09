"""Program for matrix multiplication 3x3 * 3x7 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Construct a rank-minimal tensor decomposition for 3x3 * 3x7 matrix multiplication.
    Returns schoolbook rank-63 decomposition as a guaranteed valid solution.
    Uses slice rank lower bound to prune search and structured initialization.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0

    def make_schoolbook():
        """Schoolbook decomposition of rank n*m*p = 63."""
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank

    def verify_exact(U, V, W, tol=1e-10):
        """Verify that U,V,W form an exact decomposition."""
        r = U.shape[1]
        Ur = U.reshape(n, m, r)
        Vr = V.reshape(m, p, r)
        Wr = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)

    def compute_slice_rank_lower_bound():
        """
        Compute slice rank lower bound by flattening the tensor into matrices.
        For tensor T[i,j,k], the slice rank is at least the max of:
        - rank of T flattened as (i) x (j,k) matrix
        - rank of T flattened as (j) x (i,k) matrix
        - rank of T flattened as (k) x (i,j) matrix
        """
        # Create the all-ones tensor
        T = np.ones((n, m, p))
        
        # Slice 1: fix i, matrix is (m*p) x (n) -> actually (j,k) x (i)
        # T as matrix with rows indexed by (j,k) and cols by i
        T1 = T.transpose(1, 2, 0).reshape(m*p, n)
        r1 = np.linalg.matrix_rank(T1, tol=1e-10)
        
        # Slice 2: fix j, matrix is (i*p) x (m) -> rows (i,k), cols j
        T2 = T.transpose(0, 2, 1).reshape(n*p, m)
        r2 = np.linalg.matrix_rank(T2, tol=1e-10)
        
        # Slice 3: fix k, matrix is (n*m) x (p) -> rows (i,j), cols k
        T3 = T.transpose(0, 1, 2).reshape(n*m, p)
        r3 = np.linalg.matrix_rank(T3, tol=1e-10)
        
        return max(r1, r2, r3)

    def try_decomposition(target_rank, rng, max_attempts=10):
        """
        Try to find a decomposition at the given target rank using
        structured initialization and least-squares solving for W.
        """
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Structured initialization
            U = np.zeros((n*m, target_rank), dtype=float)
            V = np.zeros((m*p, target_rank), dtype=float)
            
            # Fill columns with schoolbook-like patterns
            # For each column c, assign (i,j,k) indices
            for c in range(target_rank):
                # Cycle through i, j, k in a structured way
                i = c % n
                j = (c // n) % m
                k = (c // (n*m)) % p
                
                # Add some randomization for later columns to break symmetry
                if c >= n*m*3 and rng.random() < 0.3:
                    # Randomly perturb the assignment
                    i = rng.integers(0, n)
                    j = rng.integers(0, m)
                    k = rng.integers(0, p)
                
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
            
            # Solve for W using least squares
            W = np.zeros((p*n, target_rank), dtype=float)
            valid = True
            
            for i in range(n):
                for k in range(p):
                    A_w = np.zeros((m, target_rank), dtype=float)
                    for j in range(m):
                        A_w[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    
                    try:
                        w, residuals, rank_w, sv = np.linalg.lstsq(A_w, np.ones(m), rcond=None)
                        # Check if solution is exact
                        residual = A_w @ w - np.ones(m)
                        if np.max(np.abs(residual)) > 1e-10:
                            valid = False
                            break
                        W[k*n+i, :] = w
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid and verify_exact(U, V, W, tol=1e-10):
                return U, V, W
        
        return None

    # Start with schoolbook as guaranteed fallback
    best_U, best_V, best_W, best_rank = make_schoolbook()
    
    # Compute slice rank lower bound to prune search
    slice_lb = compute_slice_rank_lower_bound()
    
    # The minimum possible rank is at least the slice rank lower bound
    # Also, for matrix multiplication, we know rank >= max(n*m, m*p, p*n) / something
    # But slice rank gives us a concrete lower bound
    min_possible_rank = max(slice_lb, 1)
    
    rng = np.random.default_rng(12345)
    
    # Try ranks from lower bound up to schoolbook
    # Focus on ranks that might be achievable
    candidate_ranks = list(range(min_possible_rank, best_rank + 1))
    
    # Prioritize ranks that are close to known good decompositions
    # For 3x3x3, Strassen gives rank 23
    # For 3x3x7, we can try to combine Strassen-like ideas
    
    # Sort candidates: try lower ranks first, but skip ranks that are clearly impossible
    # Based on slice rank, we know rank >= slice_lb
    # Also, for each (i,k) pair, we need rank >= 1 (at least one term)
    # For each j, the "middle" dimension constrains rank
    
    # Try a focused set of ranks
    focused_ranks = []
    for r in range(min_possible_rank, best_rank + 1):
        if r % 2 == 0 or r in [23, 24, 25, 26, 27, 28, 29, 30]:
            focused_ranks.append(r)
    
    # Also include some specific candidates
    for r in [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]:
        if r not in focused_ranks and r >= min_possible_rank and r < best_rank:
            focused_ranks.append(r)
    
    focused_ranks.sort()
    
    for target_rank in focused_ranks:
        if time.time() > deadline:
            break
        
        if target_rank >= best_rank:
            break
        
        result = try_decomposition(target_rank, rng, max_attempts=5)
        if result is not None:
            U, V, W = result
            if verify_exact(U, V, W, tol=1e-10):
                best_U, best_V, best_W, best_rank = U, V, W, target_rank
                break
    
    # Ensure we have a valid result
    if best_rank > 63:
        best_U, best_V, best_W, best_rank = make_schoolbook()
    
    # Final verification
    if not verify_exact(best_U, best_V, best_W, tol=1e-10):
        best_U, best_V, best_W, best_rank = make_schoolbook()
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
