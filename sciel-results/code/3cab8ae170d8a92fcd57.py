"""Matrix multiplication 3,3,4 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=3, m=3, p=4, tol=1e-9):
    """Verify rank-r decomposition: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n=3, m=3, p=4):
    """Schoolbook rank n*m*p decomposition."""
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

def _try_merge_terms(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank by identifying pairs of terms that can be merged.
    Two terms c1 and c2 can potentially be merged if their combined contribution
    can be expressed as a single term. This is generally hard, but we can try
    a heuristic: for each pair, check if removing both and adding one new term
    with appropriately chosen values works.
    """
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each pair of columns, try to see if we can eliminate one
    # by adjusting the other. This is a local search.
    # A simpler approach: try to find linear dependencies among the rank-1 tensors.
    
    # Alternative: try random perturbations and gradient descent on the objective
    # f = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
    
    # Let's try a different approach: use the fact that we can add a small
    # perturbation to one term and adjust others to compensate.
    
    # For now, try a simple heuristic: for each pair of terms, check if we can
    # set one term's coefficients to zero by absorbing into the other.
    # This only works in special cases, so let's try something more general.
    
    # Try: for each column c, try to reduce its "weight" by distributing
    # its contribution to other columns.
    
    return None

def _construct_rank35(n=3, m=3, p=4):
    """
    Try to construct a rank-35 decomposition.
    
    Idea: The schoolbook has 36 terms. We can try to merge two terms that
    share structure. For example, terms (i,j,k) and (i,j,k') for the same
    (i,j) but different k might be partially mergeable.
    
    Actually, let's try a different approach. For 3x3 x 3x4, we can think of
    it as computing C = A * B where A is 3x3 and B is 3x4.
    
    One known approach: use the rank-23 decomposition for 3x3x3 and add
    9 more terms for the 4th column, giving 32. But we need to verify this
    works with the specific tensor structure.
    
    Let me try a direct construction with rank 35 by finding one pair of
    terms that can be merged.
    """
    # Start with schoolbook
    sb = _schoolbook(n, m, p)
    U = np.array(sb["U"], dtype=float)
    V = np.array(sb["V"], dtype=float)
    W = np.array(sb["W"], dtype=float)
    rank = 36
    
    # Try to merge terms: for each pair (c1, c2), see if we can set
    # U[:,c2] = U[:,c1] + delta, etc., and adjust W to compensate.
    # This is generally not possible, but let's try a specific construction.
    
    # Alternative: use a known algebraic identity. For 3x3x4, one can
    # potentially save one term by exploiting the rectangular structure.
    
    # Let's try: merge the last two terms (k=3, j=2, i=0 and k=3, j=2, i=1)
    # by introducing a shared computation.
    
    # Actually, let me try a more systematic approach: use random search
    # to find a rank-35 decomposition.
    
    return None

def _random_search(n, m, p, target_rank, deadline, seed=42):
    """
    Randomized search for a rank-r decomposition.
    Start from schoolbook and try to reduce rank by local moves.
    """
    rng = np.random.RandomState(seed)
    
    # Start with schoolbook
    sb = _schoolbook(n, m, p)
    U = np.array(sb["U"], dtype=float)
    V = np.array(sb["V"], dtype=float)
    W = np.array(sb["W"], dtype=float)
    current_rank = n * m * p
    
    # Try to reduce rank by one at a time
    # For each step, try to find a way to eliminate one term
    
    for target in range(current_rank - 1, target_rank - 1, -1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random attempts to reduce from current_rank to target
        success = False
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Create a perturbed version
            U_p = U[:, :target+1].copy()
            V_p = V[:, :target+1].copy()
            W_p = W[:, :target+1].copy()
            
            # Add a small random perturbation to the last column
            noise_scale = 0.1
            U_p[:, target] += rng.normal(0, noise_scale, n*m)
            V_p[:, target] += rng.normal(0, noise_scale, m*p)
            W_p[:, target] += rng.normal(0, noise_scale, p*n)
            
            # Check if this is close to valid
            # Compute the residual
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(target+1):
                            s += U_p[i*m+j, c] * V_p[j*p+k, c] * W_p[k*n+i, c]
                        residual += (s - 1.0) ** 2
            
            if residual < 1e-12:
                success = True
                U = U_p
                V = V_p
                W = W_p
                current_rank = target + 1
                break
        
        if not success:
            break
    
    if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    if not _verify_decomposition(best["U"], best["V"], best["W"], n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try to find lower-rank decompositions
    # For 3x3x4, try ranks 35, 34, 33, ...
    for target_rank in range(best_rank - 1, max(best_rank - 10, 20), -1):
        if time.time() > deadline - 10:
            break
        
        result = _random_search(n, m, p, target_rank, deadline, seed=target_rank)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                best = result
                best_rank = r
    
    return best
# EVOLVE-BLOCK-END
