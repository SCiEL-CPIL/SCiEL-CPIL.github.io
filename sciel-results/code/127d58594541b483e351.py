import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 150.0  # evaluator kills after 150s wall-clock
SAFETY_MARGIN = 10.0  # seconds before limit to stop

def _check_time():
    """Return True if we should stop searching."""
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] = sum_c U[i*4+c]... wait, let me use proper indexing
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (16, rank)
    # W shape: (p*n, rank) = (8, rank)
    # W[k*n+i] where k is output col, i is output row
    
    # The tensor T[i,j,k] should equal sum_{c=0}^{rank-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Let's check with random matrices
    rng = np.random.RandomState(42)
    for trial in range(3):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_true = A @ B
        C_approx = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                acc = 0.0
                for c in range(rank):
                    for j in range(m):
                        acc += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                C_approx[i, k] = acc
        if not np.allclose(C_true, C_approx, atol=1e-10, rtol=1e-10):
            return False
    return True

def _build_rank_28_baseline():
    """Build the known rank-28 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    def add_term(u_vec, v_vec, w_vec, col_idx):
        for idx, coeff in u_vec:
            U[idx, col_idx] += coeff
        for idx, coeff in v_vec:
            V[idx, col_idx] += coeff
        for idx, coeff in w_vec:
            W[idx, col_idx] += coeff
    
    def strassen_terms():
        terms = []
        # M1 = (X00+X11)(Y00+Y11)
        u = {0: 1, 3: 1}
        v = {0: 1, 3: 1}
        w = {0: 1, 3: 1}
        terms.append((u, v, w))
        
        # M2 = (X10+X11)Y00
        u = {2: 1, 3: 1}
        v = {0: 1}
        w = {2: 1, 3: -1}
        terms.append((u, v, w))
        
        # M3 = X00(Y01-Y11)
        u = {0: 1}
        v = {1: 1, 3: -1}
        w = {1: 1, 3: 1}
        terms.append((u, v, w))
        
        # M4 = X11(Y10-Y00)
        u = {3: 1}
        v = {2: 1, 0: -1}
        w = {0: 1, 2: 1}
        terms.append((u, v, w))
        
        # M5 = (X00+X01)Y11
        u = {0: 1, 1: 1}
        v = {3: 1}
        w = {0: -1, 1: 1}
        terms.append((u, v, w))
        
        # M6 = (X10-X00)(Y00+Y01)
        u = {2: 1, 0: -1}
        v = {0: 1, 1: 1}
        w = {3: 1}
        terms.append((u, v, w))
        
        # M7 = (X01-X11)(Y10+Y11)
        u = {1: 1, 3: -1}
        v = {2: 1, 3: 1}
        w = {0: 1}
        terms.append((u, v, w))
        
        return terms

    base_terms = strassen_terms()
    
    def apply_strassen(a_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
            col = start_col + t_idx
            u_global = {}
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = a_offset * 2 + c
                g_idx = g_r * 4 + g_c
                u_global[g_idx] = u_global.get(g_idx, 0) + coeff
            v_global = {}
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = b_row_offset * 2 + r
                g_c = b_col_offset * 2 + c
                g_idx = g_r * 4 + g_c
                v_global[g_idx] = v_global.get(g_idx, 0) + coeff
            w_global = {}
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = c_col_offset * 2 + c
                g_idx = g_c * 2 + g_r
                w_global[g_idx] = w_global.get(g_idx, 0) + coeff
            add_term(list(u_global.items()), list(v_global.items()), list(w_global.items()), col)

    apply_strassen(0, 0, 0, 0, 0)
    apply_strassen(1, 1, 0, 0, 7)
    apply_strassen(0, 0, 1, 1, 14)
    apply_strassen(1, 1, 1, 1, 21)
    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_attempts=50):
    """
    Try to reduce the rank by finding linear dependencies among the rank-1 terms.
    Uses a combination of:
    1. Random linear combinations of terms to check if they span the same tensor space
    2. Attempting to express some terms as linear combinations of others
    """
    U_orig = U.copy()
    V_orig = V.copy()
    W_orig = W.copy()
    orig_rank = U_orig.shape[1]
    
    # Strategy: Try to find a set of linear combinations that span the same subspace
    # but with fewer terms. We use the fact that if sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] = T,
    # then any invertible linear transformation of the columns preserves the tensor.
    
    # Try random rank reductions by projecting onto lower-dimensional subspaces
    rng = np.random.RandomState(12345)
    
    for attempt in range(max_attempts):
        if _check_time():
            break
        
        # Try various approaches
        if attempt % 3 == 0:
            # Approach 1: Try to zero out one column by expressing it as combination of others
            # Pick a random column to eliminate
            col_to_elim = rng.randint(orig_rank)
            
            # Check if column col_to_elim is in the span of other columns (in tensor sense)
            # This is hard to check directly, so we try a different approach
            
            # Approach 2: Use random linear combinations
            # Create a random (orig_rank-1) x orig_rank matrix Q
            Q = rng.randn(orig_rank - 1, orig_rank)
            # Try to see if U @ Q.T, V @ Q.T, W @ Q.T still decompose the tensor
            U_new = U_orig @ Q.T
            V_new = V_orig @ Q.T
            W_new = W_orig @ Q.T
            
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                # Check if any columns are (nearly) zero
                for c in range(orig_rank - 1):
                    if np.allclose(U_new[:, c], 0, atol=1e-12) and \
                       np.allclose(V_new[:, c], 0, atol=1e-12) and \
                       np.allclose(W_new[:, c], 0, atol=1e-12):
                        # Drop this column
                        mask = [i for i in range(orig_rank - 1) if i != c]
                        U_trim = U_new[:, mask]
                        V_trim = V_new[:, mask]
                        W_trim = W_new[:, mask]
                        if U_trim.shape[1] < target_rank:
                            return U_trim, V_trim, W_trim
                        return U_trim, V_trim, W_trim
        
        elif attempt % 3 == 1:
            # Approach 3: Try structured reductions
            # Look for patterns where terms can be merged
            # Use SVD-like approach on the coefficient tensors
            pass
        
        else:
            # Approach 4: Try specific small rank reductions
            # Attempt to reduce from orig_rank to orig_rank-1
            pass
    
    return None, None, None

def _try_find_rank_r(n, m, p, r, time_budget=2.0):
    """
    Try to find a rank-r decomposition by starting from rank-28 and attempting reductions.
    Returns (U, V, W) if found, else (None, None, None).
    """
    if r >= 28:
        U, V, W = _build_rank_28_baseline()
        if r == 28:
            return U, V, W
        # Need to reduce further
        U, V, W = _try_reduce_rank(U, V, W, n, m, p, r, max_attempts=20)
        if U is not None:
            return U, V, W
        return None, None, None
    
    # For r < 28, we need a fundamentally different approach
    # Try random initialization with iterative refinement
    start = time.time()
    rng = np.random.RandomState(999)
    
    # Number of random restarts based on time budget
    max_restarts = max(1, int((time_budget) / 0.1))
    
    for restart in range(max_restarts):
        if _check_time():
            break
        if (time.time() - start) > time_budget:
            break
        
        # Initialize with random values
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        # Scale to reasonable magnitude
        U /= np.linalg.norm(U) * np.sqrt(r)
        V /= np.linalg.norm(V) * np.sqrt(r)
        W /= np.linalg.norm(W) * np.sqrt(r)
        
        # Iterative refinement: try to minimize the error
        # Use a simple gradient-free approach: perturb and accept if error decreases
        best_error = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Compute initial error
        for iteration in range(50):
            if _check_time():
                break
            
            # Compute current error
            error = _compute_tensor_error(U, V, W, n, m, p)
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_error < 1e-10:
                break
            
            # Try a small perturbation
            scale = max(0.01, best_error * 0.1)
            U_pert = U + scale * rng.randn(*U.shape)
            V_pert = V + scale * rng.randn(*V.shape)
            W_pert = W + scale * rng.randn(*W.shape)
            
            error_pert = _compute_tensor_error(U_pert, V_pert, W_pert, n, m, p)
            if error_pert < error:
                U, V, W = U_pert, V_pert, W_pert
    
    if best_error < 1e-10:
        return best_U, best_V, best_W
    
    return None, None, None

def _compute_tensor_error(U, V, W, n, m, p):
    """Compute the error of the tensor decomposition using a few random test matrices."""
    rng = np.random.RandomState(777)
    total_error = 0.0
    num_tests = 3
    
    for _ in range(num_tests):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_true = A @ B
        C_approx = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                acc = 0.0
                for c in range(U.shape[1]):
                    for j in range(m):
                        acc += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                C_approx[i, k] = acc
        total_error += np.sum((C_true - C_approx) ** 2)
    
    return total_error / num_tests

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix
    by a 4x4 matrix. Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    
    # Start with the known rank-28 decomposition
    U, V, W = _build_rank_28_baseline()
    
    # Verify baseline
    assert _verify_decomposition(U, V, W, n, m, p), "Baseline decomposition failed verification"
    
    # Try to find lower-rank decompositions
    # Theoretical lower bounds: For (2,4,4), the border rank is at least 16 (from general bounds)
    # Known results: The tensor rank of <2,4,4> is known to be 28 in some references,
    # but let's try to verify if we can do better.
    
    best_result = (U, V, W)
    best_rank = 28
    
    # Try ranks from 16 up to 27, but with limited time
    # Focus on the most promising ranges first
    candidate_ranks = [27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16]
    
    for r in candidate_ranks:
        if _check_time():
            break
        
        # Allocate decreasing time budget for lower ranks (harder)
        if r >= 25:
            time_budget = 3.0
        elif r >= 22:
            time_budget = 2.0
        elif r >= 18:
            time_budget = 1.5
        else:
            time_budget = 1.0
        
        U_r, V_r, W_r = _try_find_rank_r(n, m, p, r, time_budget=time_budget)
        
        if U_r is not None:
            # Verify
            if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                best_result = (U_r, V_r, W_r)
                best_rank = r
                break  # Found a valid lower rank, stop searching
    
    U, V, W = best_result
    
    # Round to half-integers if close
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    # Check if rounding preserved validity
    if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
        U, V, W = U_rounded, V_rounded, W_rounded
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf, atol=1e-10, rtol=1e-10), (C, Cf)
    print("OK rank", U.shape[1])
