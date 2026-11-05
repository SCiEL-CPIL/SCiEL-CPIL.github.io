"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]
DEADLINE = 120.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0  # 5s safety margin

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    if Ua.shape != (n*m, rank) or Va.shape != (m*p, rank) or Wa.shape != (p*n, rank):
        return False
    
    # Check all entries are integers or half-integers
    for arr in [Ua, Va, Wa]:
        if not np.allclose(2*arr, np.round(2*arr), atol=1e-9):
            return False
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 as list of (u_dict, v_dict, w_dict)."""
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    ]
    return terms

def _build_from_terms(terms, n, m, p):
    """Build U, V, W from a list of (u_dict, v_dict, w_dict) terms."""
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_d, v_d, w_d) in enumerate(terms):
        for (i, j), val in u_d.items():
            U[i*m+j][r] = val
        for (j, k), val in v_d.items():
            V[j*p+k][r] = val
        for (k, i), val in w_d.items():
            W[k*n+i][r] = val
    
    return U, V, W

def _strassen_block_decomposition(n, m, p):
    """
    Build a decomposition using Strassen on 2x2x2 blocks where possible.
    Returns (U, V, W) or None if not applicable.
    """
    U_data = []
    V_data = []
    W_data = []
    current_rank = [0]
    
    def add_strassen(i_off, j_off, k_off):
        terms = _strassen_2x2x2()
        for u_d, v_d, w_d in terms:
            r = current_rank[0]
            current_rank[0] += 1
            for (li, lj), val in u_d.items():
                U_data.append(((i_off + li) * m + (j_off + lj), r, val))
            for (lj, lk), val in v_d.items():
                V_data.append(((j_off + lj) * p + (k_off + lk), r, val))
            for (lk, li), val in w_d.items():
                W_data.append(((k_off + lk) * n + (i_off + li), r, val))
    
    def add_schoolbook(i_off, j_off, k_off, bn, bm, bp):
        for li in range(bn):
            for lj in range(bm):
                for lk in range(bp):
                    r = current_rank[0]
                    current_rank[0] += 1
                    U_data.append(((i_off + li) * m + (j_off + lj), r, 1))
                    V_data.append(((j_off + lj) * p + (k_off + lk), r, 1))
                    W_data.append(((k_off + lk) * n + (i_off + li), r, 1))
    
    def decompose(i_off, j_off, k_off, bn, bm, bp):
        if bn == 2 and bm == 2 and bp == 2:
            add_strassen(i_off, j_off, k_off)
        elif bn == 1:
            add_schoolbook(i_off, j_off, k_off, bn, bm, bp)
        else:
            # Try to split into blocks that can use Strassen
            if bn == 3 and bm == 5 and bp == 6:
                # Split: (2,5,4) + (2,5,2) + (1,5,4) + (1,5,2)
                decompose(i_off, j_off, k_off, 2, 5, 4)
                decompose(i_off, j_off, k_off+4, 2, 5, 2)
                decompose(i_off+2, j_off, k_off, 1, 5, 4)
                decompose(i_off+2, j_off, k_off+4, 1, 5, 2)
            elif bn == 2 and bm == 5 and bp == 4:
                # Split m: (2,2,4) + (2,3,4)
                decompose(i_off, j_off, k_off, 2, 2, 4)
                decompose(i_off, j_off+2, k_off, 2, 3, 4)
            elif bn == 2 and bm == 5 and bp == 2:
                # Split m: (2,2,2) + (2,3,2)
                decompose(i_off, j_off, k_off, 2, 2, 2)
                decompose(i_off, j_off+2, k_off, 2, 3, 2)
            elif bn == 2 and bm == 2 and bp == 4:
                # Split p: (2,2,2) + (2,2,2)
                decompose(i_off, j_off, k_off, 2, 2, 2)
                decompose(i_off, j_off, k_off+2, 2, 2, 2)
            elif bn == 2 and bm == 3 and bp == 4:
                # Split m: (2,1,4) + (2,2,4)
                decompose(i_off, j_off, k_off, 2, 1, 4)
                decompose(i_off, j_off+1, k_off, 2, 2, 4)
            elif bn == 2 and bm == 3 and bp == 2:
                # Split m: (2,1,2) + (2,2,2)
                decompose(i_off, j_off, k_off, 2, 1, 2)
                decompose(i_off, j_off+1, k_off, 2, 2, 2)
            elif bn == 1:
                add_schoolbook(i_off, j_off, k_off, bn, bm, bp)
            else:
                add_schoolbook(i_off, j_off, k_off, bn, bm, bp)
    
    decompose(0, 0, 0, n, m, p)
    
    rank = current_rank[0]
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for row, col, val in U_data:
        U[row][col] = val
    for row, col, val in V_data:
        V[row][col] = val
    for row, col, val in W_data:
        W[row][col] = val
    
    if _verify_fast(U, V, W, n, m, p):
        return U, V, W
    return None

def _random_initialization(n, m, p, rank, rng):
    """Initialize U, V, W with random small integer/half-integer entries."""
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Use small random values to encourage sparsity
    for arr in [U, V, W]:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                # 50% chance of 0, otherwise random from {-2, -1, -0.5, 0.5, 1, 2}
                if rng.random() < 0.5:
                    arr[i, j] = 0
                else:
                    arr[i, j] = rng.choice([-2, -1, -0.5, 0.5, 1, 2])
    
    return U, V, W

def _tensor_residual(U, V, W, n, m, p):
    """Compute the residual tensor T - 1."""
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return T - np.ones((n, m, p))

def _tensor_norm(U, V, W, n, m, p):
    """Compute the Frobenius norm of the residual tensor."""
    res = _tensor_residual(U, V, W, n, m, p)
    return np.linalg.norm(res)

def _project_to_half_integers(U, V, W, tol=1e-6):
    """Round entries to nearest half-integer."""
    U = np.round(2 * U) / 2
    V = np.round(2 * V) / 2
    W = np.round(2 * W) / 2
    return U, V, W

def _local_refinement(U, V, W, n, m, p, max_iter=50, tol=1e-8, rng=None):
    """
    Local refinement: iteratively adjust one matrix at a time to reduce residual.
    Uses least squares to solve for each matrix given the others.
    """
    if rng is None:
        rng = random.Random()
    
    best_norm = _tensor_norm(U, V, W, n, m, p)
    U_best = U.copy()
    V_best = V.copy()
    W_best = W.copy()
    
    for iteration in range(max_iter):
        if _elapsed() > DEADLINE - 5.0:
            break
        
        # Try updating U given V, W
        # For each column r, we want U[:, r] such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is complex; instead use a simpler approach:
        # Perturb one entry at a time and accept if it reduces the norm.
        
        improved = False
        
        # Try small perturbations on random entries
        for _ in range(100):
            if _elapsed() > DEADLINE - 5.0:
                break
            
            # Choose which matrix to perturb
            mat_idx = rng.randint(0, 2)
            if mat_idx == 0:
                mat = U
            elif mat_idx == 1:
                mat = V
            else:
                mat = W
            
            # Choose a random entry
            i = rng.randint(0, mat.shape[0] - 1)
            j = rng.randint(0, mat.shape[1] - 1)
            
            # Try changing to a nearby half-integer
            current_val = mat[i, j]
            neighbors = [current_val + delta for delta in [-1, -0.5, 0.5, 1]]
            neighbors = [max(-4, min(4, nb)) for nb in neighbors]
            
            for new_val in neighbors:
                if new_val == current_val:
                    continue
                
                # Save and modify
                old_val = mat[i, j]
                mat[i, j] = new_val
                
                new_norm = _tensor_norm(U, V, W, n, m, p)
                
                if new_norm < best_norm - 1e-10:
                    best_norm = new_norm
                    improved = True
                    # Keep the change
                    if mat_idx == 0:
                        U_best = U.copy()
                    elif mat_idx == 1:
                        V_best = V.copy()
                    else:
                        W_best = W.copy()
                    break
                else:
                    # Revert
                    mat[i, j] = old_val
        
        if not improved:
            break
    
    return U_best, V_best, W_best, best_norm

def _structured_random_search(n, m, p, max_rank=60, max_trials=100, seed=42):
    """
    Structured random search with local refinement.
    Tries various ranks from low to high, with random initialization and local refinement.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    # Start with the Strassen-based decomposition as a baseline
    strassen_result = _strassen_block_decomposition(n, m, p)
    if strassen_result:
        U_s, V_s, W_s = strassen_result
        base_rank = len(U_s[0])
        best_rank = base_rank
        best_U, best_V, best_W = U_s, V_s, W_s
    else:
        base_rank = n * m * p
        best_rank = base_rank
        best_U = np.zeros((n*m, base_rank))
        best_V = np.zeros((m*p, base_rank))
        best_W = np.zeros((p*n, base_rank))
        # Initialize with schoolbook
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i*m+j, col] = 1
                    best_V[j*p+k, col] = 1
                    best_W[k*n+i, col] = 1
                    col += 1
    
    # Try to find lower rank decompositions
    for rank in range(max(20, best_rank - 10), best_rank):
        if _elapsed() > DEADLINE - 10.0:
            break
        
        for trial in range(max_trials):
            if _elapsed() > DEADLINE - 5.0:
                break
            
            # Random initialization
            U, V, W = _random_initialization(n, m, p, rank, rng)
            
            # Local refinement
            U, V, W, norm = _local_refinement(U, V, W, n, m, p, max_iter=30, rng=rng)
            
            # Check if we got a valid decomposition
            if norm < 1e-6:
                # Round to half-integers
                U, V, W = _project_to_half_integers(U, V, W)
                
                # Verify
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                if _verify_fast(U_list, V_list, W_list, n, m, p):
                    best_rank = rank
                    best_U = U
                    best_V = V
                    best_W = W
                    # Continue searching for even lower rank
                    break
        
        # If we found a valid decomposition at this rank, continue to lower ranks
        # Otherwise, stop if we've been trying for too long
        if _elapsed() > DEADLINE - 15.0:
            break
    
    # Convert best result to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    if _verify_fast(U_list, V_list, W_list, n, m, p):
        return U_list, V_list, W_list
    
    # Fallback to Strassen or schoolbook
    if strassen_result:
        return strassen_result
    return _schoolbook(n, m, p)["U"], _schoolbook(n, m, p)["V"], _schoolbook(n, m, p)["W"]

def construct():
    n, m, p = DIMENSIONS
    
    # First try the structured Strassen-based decomposition
    strassen_result = _strassen_block_decomposition(n, m, p)
    if strassen_result:
        U_s, V_s, W_s = strassen_result
        base_rank = len(U_s[0])
    else:
        base_rank = n * m * p
    
    # If we have time, try random search for lower rank
    if _remaining() > 30.0:
        U_r, V_r, W_r = _structured_random_search(n, m, p, max_rank=base_rank, max_trials=50, seed=42)
        if U_r and V_r and W_r:
            rank_r = len(U_r[0])
            if rank_r < base_rank:
                return {"U": U_r, "V": V_r, "W": W_r}
    
    # Return the best Strassen-based result or schoolbook
    if strassen_result:
        return {"U": U_s, "V": V_s, "W": W_s}
    else:
        sb = _schoolbook(n, m, p)
        return sb

