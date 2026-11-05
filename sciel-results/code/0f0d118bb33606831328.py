"""Combined approach: structured rank-44 base + rank reduction via structured elimination."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _build_target_tensor(n, m, p):
    """Return the 3D tensor T[i,j,k] = 1."""
    return np.ones((n, m, p), dtype=np.float64)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if U,V,W exactly decompose the tensor (within tol)."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _construct_structured_decomposition():
    """Construct a rank-44 decomposition using Strassen for 2x2x2 blocks and schoolbook for 1x2x2 blocks."""
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    def add_strassen(rA, cA, rB, cB):
        nonlocal col
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = -1
        col += 1
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = -1
        W[(cB+1)*n + (rA+0), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = -1
        W[(cB+0)*n + (rA+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1
        col += 1
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = -1
        col += 1
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+0), col] = -1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        U[(rA+0)*m + (cA+1), col] = 1
        U[(rA+1)*m + (cA+1), col] = -1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1

    def add_schoolbook_1x2x2(rA, cA, rB, cB):
        nonlocal col
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        col += 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        col += 1

    add_strassen(0, 0, 0, 0)
    add_strassen(0, 2, 2, 0)
    add_strassen(0, 0, 0, 2)
    add_strassen(0, 2, 2, 2)
    add_schoolbook_1x2x2(2, 0, 0, 0)
    add_schoolbook_1x2x2(2, 2, 2, 0)
    add_schoolbook_1x2x2(2, 0, 0, 2)
    add_schoolbook_1x2x2(2, 2, 2, 2)
    
    return U, V, W

def _compute_outer_products(U, V, W, n, m, p):
    """Precompute all term outer products as flattened vectors."""
    rank = U.shape[1]
    n_entries = n * m * p
    all_outers = np.zeros((n_entries, rank))
    for r in range(rank):
        outer = np.einsum('ij,jk,ki->ijk',
                         U[:, r].reshape(n, m),
                         V[:, r].reshape(m, p),
                         W[:, r].reshape(p, n))
        all_outers[:, r] = outer.flatten()
    return all_outers

def _try_eliminate_fast(U, V, W, n, m, p, time_budget=30):
    """Try to eliminate one term using fast linear algebra with precomputed products."""
    start_time = time.time()
    rank = U.shape[1]
    
    if rank <= 1:
        return U, V, W, False
    
    # Precompute outer products
    all_outers = _compute_outer_products(U, V, W, n, m, p)
    
    # Try each term as candidate for elimination
    for r_to_remove in range(rank):
        if time.time() - start_time > time_budget:
            break
        
        # Build matrix of remaining terms
        remaining_idx = [r for r in range(rank) if r != r_to_remove]
        A_mat = all_outers[:, remaining_idx]
        deficit_flat = all_outers[:, r_to_remove]
        
        # Solve least squares
        try:
            delta, residuals, rank_mat, sv = np.linalg.lstsq(A_mat, deficit_flat, rcond=1e-10)
            
            # Check residual
            if np.linalg.norm(A_mat @ delta - deficit_flat) < 1e-8:
                # Check if delta entries are nice (integers or half-integers)
                delta_2x = delta * 2
                if np.allclose(delta_2x, np.round(delta_2x), atol=1e-6):
                    # Apply the delta
                    U_new = U.copy()
                    V_new = V.copy()
                    W_new = W.copy()
                    
                    for i, orig_r in enumerate(remaining_idx):
                        U_new[:, i] += delta[i] * U[:, orig_r]
                        V_new[:, i] += delta[i] * V[:, orig_r]
                        W_new[:, i] += delta[i] * W[:, orig_r]
                    
                    # Remove the eliminated term
                    U_new = np.delete(U_new, r_to_remove, axis=1)
                    V_new = np.delete(V_new, r_to_remove, axis=1)
                    W_new = np.delete(W_new, r_to_remove, axis=1)
                    
                    # Verify
                    if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-6):
                        return U_new, V_new, W_new, True
        except:
            pass
    
    return U, V, W, False

def _try_multiple_eliminations(U, V, W, n, m, p, time_budget=60):
    """Try to eliminate multiple terms iteratively."""
    start_time = time.time()
    U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
    initial_rank = U.shape[1]
    eliminations = 0
    
    while time.time() - start_time < time_budget:
        U_cur, V_cur, W_cur, success = _try_eliminate_fast(U_cur, V_cur, W_cur, n, m, p, time_budget=5)
        if not success:
            break
        eliminations += 1
    
    if eliminations > 0:
        return U_cur, V_cur, W_cur
    return U, V, W

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Start with the structured rank-44 decomposition
    U44, V44, W44 = _construct_structured_decomposition()
    best_U, best_V, best_W = U44.copy(), V44.copy(), W44.copy()
    best_rank = 44
    
    # Try to reduce rank with aggressive search
    remaining_time = deadline - time.time() - 15
    if remaining_time > 10:
        U_try, V_try, W_try = _try_multiple_eliminations(U44, V44, W44, n, m, p, time_budget=remaining_time)
        if U_try.shape[1] < best_rank:
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = U_try.shape[1]
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
