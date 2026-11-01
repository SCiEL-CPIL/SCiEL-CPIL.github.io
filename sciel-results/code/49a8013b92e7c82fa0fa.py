"""Matrix multiplication 2,2,11 - rank-minimal decomposition via structured search."""
import time
import numpy as np

DIMENSIONS = [2, 2, 11]

def _build_target_tensor():
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    n, m, p = 2, 2, 11
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T=None):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    if T is None:
        T = _build_target_tensor()
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, T_decomp, atol=1e-9)

def _build_strassen_decomposition():
    """
    Build a decomposition using Strassen for 2x2 blocks.
    For 2x2 * 2x11: use Strassen on pairs of columns.
    
    11 columns = 5 pairs + 1 single column
    Each Strassen block (2x2 * 2x2) has rank 7
    Each single column (2x2 * 2x1) has rank 4 (schoolbook for that column)
    
    Total: 5*7 + 4 = 39
    """
    n, m, p = 2, 2, 11
    num_pairs = p // 2  # 5 pairs
    remainder = p % 2   # 1 remaining column
    
    rank = num_pairs * 7 + remainder * 4
    U = np.zeros((n * m, rank), dtype=np.float64)  # (4, rank)
    V = np.zeros((m * p, rank), dtype=np.float64)  # (22, rank)
    W = np.zeros((p * n, rank), dtype=np.float64)  # (22, rank)
    
    col = 0
    for b in range(num_pairs):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = p + k0   # B[1, k0]
        v_h = p + k1   # B[1, k1]
        
        w_c00 = k0 * n + 0   # C[0, k0]
        w_c01 = k1 * n + 0   # C[0, k1]
        w_c10 = k0 * n + 1   # C[1, k0]
        w_c11 = k1 * n + 1   # C[1, k1]
        
        u_a = 0  # A[0,0]
        u_b = 1  # A[0,1]
        u_c = 2  # A[1,0]
        u_d = 3  # A[1,1]
        
        # m1 = (a+d)(e+h)
        U[u_a, col] = 1.0; U[u_d, col] = 1.0
        V[v_e, col] = 1.0; V[v_h, col] = 1.0
        W[w_c00, col] = 1.0; W[w_c11, col] = 1.0
        col += 1
        
        # m2 = (c+d)e
        U[u_c, col] = 1.0; U[u_d, col] = 1.0
        V[v_e, col] = 1.0
        W[w_c10, col] = 1.0; W[w_c11, col] = -1.0
        col += 1
        
        # m3 = a(f-h)
        U[u_a, col] = 1.0
        V[v_f, col] = 1.0; V[v_h, col] = -1.0
        W[w_c01, col] = 1.0; W[w_c11, col] = 1.0
        col += 1
        
        # m4 = d(g-e)
        U[u_d, col] = 1.0
        V[v_g, col] = 1.0; V[v_e, col] = -1.0
        W[w_c00, col] = 1.0; W[w_c10, col] = 1.0
        col += 1
        
        # m5 = (a+b)h
        U[u_a, col] = 1.0; U[u_b, col] = 1.0
        V[v_h, col] = 1.0
        W[w_c00, col] = -1.0; W[w_c01, col] = 1.0
        col += 1
        
        # m6 = (c-a)(e+f)
        U[u_c, col] = 1.0; U[u_a, col] = -1.0
        V[v_e, col] = 1.0; V[v_f, col] = 1.0
        W[w_c11, col] = 1.0
        col += 1
        
        # m7 = (b-d)(g+h)
        U[u_b, col] = 1.0; U[u_d, col] = -1.0
        V[v_g, col] = 1.0; V[v_h, col] = 1.0
        W[w_c00, col] = 1.0
        col += 1
    
    if remainder == 1:
        k2 = num_pairs * 2  # = 10
        
        v_e2 = k2        # B[0, 10]
        v_g2 = p + k2    # B[1, 10]
        w_c02 = k2 * n + 0   # C[0, 10]
        w_c12 = k2 * n + 1   # C[1, 10]
        
        U[0, col] = 1.0
        V[v_e2, col] = 1.0
        W[w_c02, col] = 1.0
        col += 1
        
        U[1, col] = 1.0
        V[v_g2, col] = 1.0
        W[w_c02, col] = 1.0
        col += 1
        
        U[2, col] = 1.0
        V[v_e2, col] = 1.0
        W[w_c12, col] = 1.0
        col += 1
        
        U[3, col] = 1.0
        V[v_g2, col] = 1.0
        W[w_c12, col] = 1.0
        col += 1
    
    return U, V, W

def _build_schoolbook():
    """Build the schoolbook rank-44 decomposition."""
    n, m, p = 2, 2, 11
    rank = n * m * p  # 44
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _try_rank_reduction(U, V, W, T, max_time=60):
    """
    Try to reduce the rank by removing linearly dependent rank-1 terms.
    Uses a greedy approach: for each term, check if removing it still allows
    exact reconstruction. Uses SVD to detect numerical rank.
    """
    start = time.time()
    n_terms = U.shape[1]
    T_flat = T.flatten()
    
    # Precompute the tensor contribution of each rank-1 term
    # Term r contributes U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # Flatten each term's contribution
    term_tensors = []
    for r in range(n_terms):
        tt = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        term_tensors.append(tt.flatten())
    term_tensors = np.array(term_tensors).T  # shape (n_terms, n*m*p*n)
    
    # Try greedy removal: iterate, try to remove each term
    # A term can be removed if its contribution is in the span of the others
    # This is equivalent to checking if the SVD rank decreases when we remove it
    
    current_terms = list(range(n_terms))
    improved = False
    
    # Use a simpler approach: try removing each term one at a time
    # and check if the remaining terms still span the target tensor
    # We do this by solving a linear system
    
    # For efficiency, build the matrix M where M[:,r] = term_tensors[r]
    # We want to find the minimum subset of columns that still spans T_flat
    
    # Greedy forward selection won't help (we already have all terms)
    # Greedy backward elimination:
    # For each term, check if T_flat is in the span of all other terms
    # This is: is the system M_without_r * x = T_flat solvable?
    # Equivalently: is the augmented matrix [M | T_flat] the same rank as M?
    
    # More efficient: compute SVD of M, then for each column check if it's redundant
    # A column r is redundant if it's in the span of the other columns
    # AND T_flat is in the span of the other columns (which it is if it's in span of all)
    
    # Let's use a different approach: 
    # The set of all term tensors spans T_flat. We want the minimum subset.
    # This is the "minimum subset" problem which is NP-hard in general,
    # but for small dimensions we can be smarter.
    
    # Key insight: if term r is a linear combination of other terms, 
    # we can remove it. Check: is term_tensors[r] in span(term_tensors[other])?
    
    # For each term, check if it's in the span of the rest
    # Use QR with column pivoting on the matrix of all other terms
    
    n_dim = T_flat.shape[0]
    
    # First, find which terms are linearly dependent among themselves
    # Build matrix of all term tensors
    M = term_tensors  # (n_dim, n_terms)
    
    # Compute SVD to find numerical rank
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    # Numerical rank based on singular values
    tol = 1e-10
    num_rank = np.sum(s_vals > tol * s_vals[0] if s_vals.size > 0 else 0)
    
    if num_rank >= n_terms:
        # All terms are independent, can't reduce
        return None
    
    # Now try greedy removal: remove terms that are in the span of others
    # Start with all terms, try to remove one at a time
    
    active = set(range(n_terms))
    removed = []
    
    for attempt in range(n_terms):
        if time.time() - start > max_time:
            break
        if len(active) == 0:
            break
        
        found_removal = False
        for r in sorted(active):
            if time.time() - start > max_time:
                break
            # Check if term r is in span of other active terms
            others = [t for t in active if t != r]
            if not others:
                # Only one term left, can't remove
                break
            
            # Check if T_flat is in span of others
            # Build submatrix
            M_sub = M[:, others]
            
            # Check if T_flat is in the column space of M_sub
            # Use least squares: min ||M_sub x - T_flat||
            try:
                sol, residuals, rank_sub, sv = np.linalg.lstsq(M_sub, T_flat, rcond=1e-10)
            except np.linalg.LinAlgError:
                continue
            
            # Check residual
            residual = np.linalg.norm(M_sub @ sol - T_flat)
            if residual < 1e-8:
                # Term r can be removed!
                active.remove(r)
                removed.append(r)
                found_removal = True
                break
        
        if not found_removal:
            break
    
    if not removed:
        return None
    
    # Build reduced decomposition
    keep = sorted(active)
    U_red = U[:, keep]
    V_red = V[:, keep]
    W_red = W[:, keep]
    
    # Verify
    if _verify(U_red, V_red, W_red, T):
        return U_red, V_red, W_red
    
    return None

def _try_alternative_pairings(U, V, W, T, max_time=30):
    """
    Try different pairings of columns to see if we get a lower rank.
    The current pairing is (0,1),(2,3),(4,5),(6,7),(8,9),(10).
    Try other pairings that might yield better Strassen interactions.
    """
    start = time.time()
    n, m, p = 2, 2, 11
    
    # Generate some alternative pairings
    # The key is that Strassen works on 2x2 blocks of B
    # Different pairings might interact better
    
    # For now, just try a few specific alternatives
    # This is a limited search for now
    
    best = (U, V, W)
    best_rank = U.shape[1]
    
    # Try pairing the last column with one of the pairs
    # e.g., (0,1),(2,3),(4,5),(6,7),(8,10),(9)
    # This might not help directly but let's check
    
    # Actually, the main improvement path is through rank reduction
    # of the existing 39-term construction
    
    return best

def construct():
    start_time = time.time()
    DEADLINE = 145  # seconds, leave margin before 150s limit

    T = _build_target_tensor()

    # Try the Strassen-based construction (rank 39)
    U, V, W = _build_strassen_decomposition()

    if not _verify(U, V, W, T):
        # Fall back to schoolbook rank 44
        U, V, W = _build_schoolbook()
        if _verify(U, V, W, T):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
        raise RuntimeError("No valid decomposition found")
    
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Try rank reduction with generous time budget
    elapsed = time.time() - start_time
    remaining_time = DEADLINE - elapsed - 3
    if remaining_time > 5:
        reduced = _try_rank_reduction(U, V, W, T, max_time=remaining_time)
        if reduced is not None:
            rU, rV, rW = reduced
            if rU.shape[1] < best_rank:
                best_U, best_V, best_W = rU, rV, rW
                best_rank = rU.shape[1]
    
    # Try alternative constructions if time permits
    elapsed = time.time() - start_time
    remaining_time = DEADLINE - elapsed - 3
    if remaining_time > 10:
        alt = _try_alternative_pairings(best_U, best_V, best_W, T, max_time=remaining_time)
        if alt[0].shape[1] < best_rank:
            best_U, best_V, best_W = alt
    
    return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
