"""Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition search."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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


def _build_tensor_matrix(n, m, p):
    """
    Build the multiplication tensor as a matrix for SNNMF.
    The tensor T[i,j,k] = 1 for all valid i,j,k.
    We flatten (i,j) into rows and k into columns, but that's not quite right.
    
    Better approach: The tensor is a 3D array T[i,j,k]. We can view it as:
    For each k, T[:,:,k] is an n x m matrix of all 1s.
    
    For SNNMF, we need to find U, V, W such that:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    This is a 3-way tensor decomposition. Let's use a different strategy:
    Enumerate all possible rank-1 components and select the minimal subset.
    """
    return None


def _generate_rank1_candidates(n, m, p, max_entries=50):
    """
    Generate candidate rank-1 components for the tensor decomposition.
    A rank-1 component is (u, v, w) where:
    - u is a vector in R^(n*m)
    - v is a vector in R^(m*p)  
    - w is a vector in R^(p*n)
    
    For the matmul tensor, natural candidates are based on the structure.
    We'll generate candidates from the schoolbook basis and its variations.
    """
    candidates = []
    
    # Schoolbook basis: one component per (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                candidates.append((u, v, w))
    
    # Add some structured candidates
    # For example, combine k dimensions
    for i in range(n):
        for j in range(m):
            for k1 in range(p):
                for k2 in range(k1 + 1, p):
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    u[i * m + j] = 1
                    v[j * p + k1] = 1
                    v[j * p + k2] = 1
                    w[k1 * n + i] = 0.5
                    w[k2 * n + i] = 0.5
                    candidates.append((u, v, w))
    
    return candidates


def _snnmf_search(n, m, p, deadline, max_rank=60):
    """
    Sparse Non-Negative Matrix Factorization approach.
    Use iterative hard thresholding to find minimal rank decomposition.
    """
    start_time = time.time()
    
    # Generate candidates
    candidates = _generate_rank1_candidates(n, m, p)
    n_cands = len(candidates)
    
    # We need to select a subset of candidates such that:
    # For all (i,j,k): sum_{r in selected} u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    
    # This is a 3D matching problem. Let's use a greedy approach with L1 regularization.
    
    # First, try to find a solution using only schoolbook components
    # Then try to replace groups with fewer components
    
    # Strategy: Start with schoolbook, try to merge components
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p
    
    # Try various rank targets from low to high
    for target_rank in range(32, min(best_rank, max_rank + 1)):
        if time.time() > deadline - 10:
            break
        
        # Use random restarts to find a decomposition of this rank
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(trial * 7919 + target_rank)
            
            # Random initialization with sparse structure
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Initialize with random sparse patterns
            for r in range(target_rank):
                # Each rank-1 component should have some structure
                # Choose which (i,j) pairs are active in U
                n_active_u = np.random.randint(1, 4)
                active_u = np.random.choice(n * m, n_active_u, replace=False)
                U[active_u, r] = np.random.choice([-1, 1], n_active_u)
                
                n_active_v = np.random.randint(1, 4)
                active_v = np.random.choice(m * p, n_active_v, replace=False)
                V[active_v, r] = np.random.choice([-1, 1], n_active_v)
                
                n_active_w = np.random.randint(1, 4)
                active_w = np.random.choice(p * n, n_active_w, replace=False)
                W[active_w, r] = np.random.choice([-1, 1], n_active_w)
            
            # Verify
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p):
                best_U, best_V, best_W = U_list, V_list, W_list
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    if best_U is not None:
        return {"U": best_U, "V": best_V, "W": best_W}, best_rank
    return None, best_rank


def _verify(U, V, W, n, m, p):
    """Check if the factorization exactly reproduces the matrix multiplication tensor."""
    rank = len(U[0])
    # Verify: for each (i,j,k), check that sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _try_random_search(n, m, p, max_rank, n_trials, deadline):
    """Try to find a low-rank decomposition using random initialization and local search."""
    best = None
    best_rank = n * m * p  # schoolbook rank
    
    start = time.time()
    for trial in range(n_trials):
        if time.time() > deadline - 5:
            break
        
        # Try ranks from low to high
        for rank in range(max_rank, best_rank):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small integers/half-integers
            np.random.seed(trial * 1000 + rank)
            U = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(n*m, rank))
            V = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(m*p, rank))
            W = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(p*n, rank))
            
            # Check
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
            elif rank < best_rank:
                # Keep as candidate for local refinement
                pass
    
    return best, best_rank


def _structured_decomposition(n, m, p):
    """
    Try structured decompositions based on known techniques.
    For 3x4x8, we exploit the block structure.
    """
    # The schoolbook rank is 96. Lower bound is 32.
    
    # Block structure: B is 4x8 = [B1 | B2] where B1, B2 are 4x4
    # C = A * B1 + A * B2
    # This is just schoolbook with rank 96.
    
    # Try to use the fact that we can share computation between B1 and B2.
    # If we could compute A*B1 and A*B2 with fewer than 12 operations each...
    
    # For now, return schoolbook. The SNNMF search will try to improve.
    return _schoolbook(n, m, p)


def _local_refine(U, V, W, n, m, p, max_iters, deadline):
    """
    Local search: try to reduce rank by zeroing out columns or 
    finding redundant columns that can be eliminated.
    """
    rank = len(U[0])
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    # Try to eliminate columns that are linearly dependent or redundant
    for _ in range(max_iters):
        if time.time() > deadline - 2:
            break
        
        improved = False
        for r in range(rank - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            
            # Try removing column r: check if the remaining columns still span the tensor
            # This is expensive, so only try for the last few columns
            if r < rank - 5:
                break
            
            # Check if column r is "redundant" by seeing if we can express it
            # in terms of others (approximately)
            # For now, just check if setting it to 0 still works
            U_save = [U[i][r] for i in range(n*m)]
            V_save = [V[i][r] for i in range(m*p)]
            W_save = [W[i][r] for i in range(p*n)]
            
            # Zero out column r
            for i in range(n*m):
                U[i][r] = 0
            for i in range(m*p):
                V[i][r] = 0
            for i in range(p*n):
                W[i][r] = 0
            
            if _verify(U, V, W, n, m, p):
                # Remove column r
                for i in range(n*m):
                    U[i].pop(r)
                for i in range(m*p):
                    V[i].pop(r)
                for i in range(p*n):
                    W[i].pop(r)
                rank -= 1
                improved = True
            else:
                # Restore
                for i in range(n*m):
                    U[i][r] = U_save[i]
                for i in range(m*p):
                    V[i][r] = V_save[i]
                for i in range(p*n):
                    W[i][r] = W_save[i]
            
            if improved:
                break
        
        if not improved:
            break
    
    return {"U": U, "V": V, "W": W}, rank


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Start with schoolbook
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    assert _verify(best_result["U"], best_result["V"], best_result["W"], n, m, p), "Schoolbook verification failed"
    
    # Try structured decompositions
    if time.time() < deadline - 10:
        structured = _structured_decomposition(n, m, p)
        if _verify(structured["U"], structured["V"], structured["W"], n, m, p):
            s_rank = len(structured["U"][0])
            if s_rank < best_rank:
                best_result = structured
                best_rank = s_rank
    
    # Try SNNMF-based search
    if time.time() < deadline - 15:
        snnfm_result, snnfm_rank = _snnmf_search(n, m, p, deadline, max_rank=60)
        if snnfm_result is not None:
            best_result = snnfm_result
            best_rank = snnfm_rank
    
    # Try random search with various rank targets
    if time.time() < deadline - 5:
        # Try ranks from 40 up to best_rank - 1
        max_search_rank = min(best_rank - 1, 60)
        result, found_rank = _try_random_search(n, m, p, max_search_rank, 200, deadline)
        if result is not None:
            best_result = result
            best_rank = found_rank
    
    # Local refinement on the best result found
    if time.time() < deadline - 2:
        refined, refined_rank = _local_refine(
            best_result["U"], best_result["V"], best_result["W"], 
            n, m, p, max_iters=50, deadline=deadline
        )
        if _verify(refined["U"], refined["V"], refined["W"], n, m, p):
            if refined_rank < best_rank:
                best_result = refined
                best_rank = refined_rank
    
    # Final verification
    if not _verify(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        # Fallback to schoolbook
        best_result = _schoolbook(n, m, p)
    
    return best_result
# EVOLVE-BLOCK-END
