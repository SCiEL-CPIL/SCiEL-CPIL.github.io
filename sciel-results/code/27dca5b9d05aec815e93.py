"""Program for 3x5x5 matrix multiplication tensor decomposition.

Strategy: Start with the schoolbook decomposition (rank 75) and attempt to find
a lower-rank decomposition using structured random search and a local search
approach that perturbs the schoolbook basis and checks for linear dependencies
that could reduce rank.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]
N, M, P = 3, 5, 5
DEADLINE = 120.0
SAFETY_MARGIN = 15.0

def _time_remaining():
    return DEADLINE - time.time() - SAFETY_MARGIN

def _schoolbook():
    """Return the standard rank-75 decomposition."""
    n, m, p = N, M, P
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(decomp, tolerance=1e-6):
    """Check if the decomposition exactly realizes the matrix multiplication tensor."""
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    n, m, p = N, M, P
    rank = U.shape[1]
    
    # Reshape to tensor dimensions
    Uc = U.reshape(n, m, rank)
    Vc = V.reshape(m, p, rank)
    Wc = W.reshape(p, n, rank)
    
    # Compute sum over rank dimension c: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    try:
        total = np.einsum('ijc,jkc,kic->ijk', Uc, Vc, Wc)
    except Exception:
        return False
        
    return np.allclose(total, 1.0, atol=tolerance)

def _verify_fast(U, V, W, n, m, p, tolerance=1e-6):
    """Fast verification using numpy arrays directly."""
    rank = U.shape[1]
    Uc = U.reshape(n, m, rank)
    Vc = V.reshape(m, p, rank)
    Wc = W.reshape(p, n, rank)
    try:
        total = np.einsum('ijc,jkc,kic->ijk', Uc, Vc, Wc)
    except Exception:
        return False
    return np.allclose(total, 1.0, atol=tolerance)

def _try_rank_reduction_spectral():
    """
    Attempt to reduce rank by finding linear dependencies among the rank-1 terms.
    This is a heuristic: we look for subsets of columns that can be replaced by
    a lower-dimensional combination.
    
    For now, this is a placeholder for a more sophisticated approach.
    """
    return None

def _local_search_from_schoolbook(max_iter=1000):
    """
    Start from schoolbook and try to perturb entries to find a valid lower-rank decomposition.
    This is unlikely to work directly, but we try a few structured perturbations.
    """
    if _time_remaining() < 10:
        return None
    
    base = _schoolbook()
    U = base["U"].copy()
    V = base["V"].copy()
    W = base["W"].copy()
    
    # Try to merge some terms by setting some entries to 0 and adjusting others
    # This is a very rough heuristic
    vals = [1, -1, 0.5, -0.5, 2, -2]
    
    for _ in range(max_iter):
        if _time_remaining() < 5:
            break
        
        # Pick a random rank to try (lower than 75)
        target_rank = random.randint(50, 74)
        
        # Create a random subset of the schoolbook terms
        cols = random.sample(range(75), target_rank)
        
        U_new = U[:, cols].copy()
        V_new = V[:, cols].copy()
        W_new = W[:, cols].copy()
        
        # Add some noise to see if we can "fix" it
        if random.random() < 0.5:
            # Perturb a few entries
            for _ in range(random.randint(1, 5)):
                r = random.randint(0, target_rank - 1)
                U_new[random.randint(0, N*M-1), r] = random.choice(vals)
                V_new[random.randint(0, M*P-1), r] = random.choice(vals)
                W_new[random.randint(0, P*N-1), r] = random.choice(vals)
        
        if _verify_fast(U_new, V_new, W_new, N, M, P):
            return {"U": U_new, "V": V_new, "W": W_new}
            
    return None

def _search_random_structured(rank_min, rank_max):
    """
    Structured random search: use the fact that the tensor is a sum of rank-1 terms
    with specific support patterns.
    """
    if _time_remaining() < 5:
        return None
    
    vals = [1, -1, 0.5, -0.5, 2, -2, 1.5, -1.5]
    
    while _time_remaining() > 10:
        target_rank = random.randint(rank_min, rank_max)
        
        U = np.zeros((N*M, target_rank))
        V = np.zeros((M*P, target_rank))
        W = np.zeros((P*N, target_rank))
        
        # Each term should have support on a "line" in each dimension
        # For term c, pick:
        # - a row i_c in [0, N) and a set of columns J_c in [0, M) for U
        # - a column j_c in [0, M) and a set of rows K_c in [0, P) for V
        # - a row k_c in [0, P) and a set of columns I_c in [0, N) for W
        # But this is just schoolbook. Let's allow more general patterns.
        
        for c in range(target_rank):
            # Random sparse pattern
            u_sparsity = random.randint(1, 4)
            v_sparsity = random.randint(1, 4)
            w_sparsity = random.randint(1, 4)
            
            u_rows = random.sample(range(N*M), u_sparsity)
            v_rows = random.sample(range(M*P), v_sparsity)
            w_rows = random.sample(range(P*N), w_sparsity)
            
            for r in u_rows:
                U[r, c] = random.choice(vals)
            for r in v_rows:
                V[r, c] = random.choice(vals)
            for r in w_rows:
                W[r, c] = random.choice(vals)
                
        if _verify_fast(U, V, W, N, M, P):
            return {"U": U, "V": V, "W": W}
            
    return None

def construct():
    # Start with the safe, known schoolbook decomposition
    base = _schoolbook()
    best_rank = N * M * P  # 75
    best_decomp = base
    
    # Check if we have time to attempt optimization
    if _time_remaining() > 20:
        # Try local search from schoolbook
        candidate = _local_search_from_schoolbook()
        if candidate is not None:
            return candidate
        
        # Try structured random search for lower ranks
        # Search from 74 down to 60
        for r in range(best_rank - 1, max(60, best_rank - 15), -1):
            if _time_remaining() < 15:
                break
            candidate = _search_random_structured(r, r)
            if candidate is not None:
                return candidate
                
    return best_decomp

if __name__ == "__main__":
    # Quick self-check
    result = construct()
    if _verify(result):
        print("Valid decomposition found with rank", len(result["U"][0]))
    else:
        print("ERROR: Invalid decomposition")
