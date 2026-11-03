"""Matrix multiplication 3x4 by 4x6: Structured Strassen decomposition with Greedy Integer Span Reduction."""
import time
import numpy as np
import random

n, m, p = 3, 4, 6

_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _time_ok():
    return _time_left() > 5.0

def _build_base():
    """Build the rank-66 base decomposition. Returns (U, V, W) as numpy int arrays."""
    rank = 66
    U = np.zeros((n * m, rank), dtype=int)   # 12 x 66
    V = np.zeros((m * p, rank), dtype=int)   # 24 x 66
    W = np.zeros((p * n, rank), dtype=int)   # 18 x 66

    col = 0

    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j, col] = 1
        V[j * p + k, col] = 1
        W[k * n + i, col] = 1
        col += 1

    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        a = i_base * m + j_base
        b = i_base * m + (j_base + 1)
        c = (i_base + 1) * m + j_base
        d = (i_base + 1) * m + (j_base + 1)
        e = j_base * p + k_base
        f = j_base * p + (k_base + 1)
        g = (j_base + 1) * p + k_base
        h = (j_base + 1) * p + (k_base + 1)
        P = k_base * n + i_base
        Q = (k_base + 1) * n + i_base
        R = k_base * n + (i_base + 1)
        S = (k_base + 1) * n + (i_base + 1)

        # P1 = (a+d)(e+h)
        U[a, col] = 1; U[d, col] = 1
        V[e, col] = 1; V[h, col] = 1
        W[P, col] = 1; W[S, col] = 1
        col += 1
        # P2 = (c+d)e
        U[c, col] = 1; U[d, col] = 1
        V[e, col] = 1
        W[R, col] = 1; W[S, col] = -1
        col += 1
        # P3 = a(f-h)
        U[a, col] = 1
        V[f, col] = 1; V[h, col] = -1
        W[Q, col] = 1; W[S, col] = 1
        col += 1
        # P4 = d(g-e)
        U[d, col] = 1
        V[g, col] = 1; V[e, col] = -1
        W[P, col] = 1; W[R, col] = 1
        col += 1
        # P5 = (a+b)h
        U[a, col] = 1; U[b, col] = 1
        V[h, col] = 1
        W[P, col] = -1; W[Q, col] = 1
        col += 1
        # P6 = (c-a)(e+f)
        U[c, col] = 1; U[a, col] = -1
        V[e, col] = 1; V[f, col] = 1
        W[S, col] = 1
        col += 1
        # P7 = (b-d)(g+h)
        U[b, col] = 1; U[d, col] = -1
        V[g, col] = 1; V[h, col] = 1
        W[P, col] = 1
        col += 1

    for j in range(m):
        for k in range(p):
            add_schoolbook(0, j, k)

    for k_block in range(p // 2):
        for j_block in range(m // 2):
            add_strassen(1, 2 * j_block, 2 * k_block)

    return U, V, W

def _compute_contributions(U, V, W):
    """
    Compute the vectorized tensor contribution for each column r.
    Returns a matrix of shape (n*m*p, rank) where each column r is 
    the vector v_r such that sum_r v_r = target_tensor.
    """
    R = U.shape[1]
    contributions = np.zeros((n * m * p, R), dtype=int)
    
    # Precompute indices for speed
    for r in range(R):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        
        # Efficiently compute the tensor product for this column
        # We need to fill contributions[flat_idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Iterate over non-zero elements of U, V, W to save time? 
        # Given small dimensions, direct iteration is fine.
        
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val == 0: continue
                for k in range(p):
                    v_val = v_col[j * p + k]
                    if v_val == 0: continue
                    w_val = w_col[k * n + i]
                    if w_val == 0: continue
                    val = u_val * v_val * w_val
                    flat_idx = (i * m + j) * p + k  # Wait, tensor is T[i,j,k]. 
                    # The target tensor is 1 for all i,j,k.
                    # The index in the flattened vector should correspond to (i,j,k).
                    # Let's use a consistent flattening: i*m*p + j*p + k
                    contributions[i * m * p + j * p + k, r] = val
    return contributions

def _is_redundant(contrib_matrix, active_cols, idx):
    """
    Check if column idx is an integer linear combination of the other active columns.
    contrib_matrix: (72, R)
    active_cols: list of indices
    idx: index to check
    """
    target = contrib_matrix[:, idx]
    if not np.any(target):
        return True, None # Zero column is redundant
    
    # Build matrix of other active columns
    other_cols = [c for c in active_cols if c != idx]
    if not other_cols:
        return False, None
        
    A = contrib_matrix[:, other_cols].astype(float)
    b = target.astype(float)
    
    try:
        # Solve A x = b
        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return False, None
        
    # Check if solution is integer
    rounded_sol = np.round(sol)
    if not np.allclose(sol, rounded_sol, atol=1e-6):
        return False, None
        
    # Verify exactly
    if np.allclose(A @ rounded_sol, b, atol=1e-6):
        return True, rounded_sol.astype(int)
        
    return False, None

def _verify(U, V, W):
    """Check exactness and return (ok, effective_rank)."""
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    
    # Reconstruct tensor
    recon = np.zeros((n, m, p), dtype=int)
    for r in range(U.shape[1]):
        if not (np.any(U[:, r]) and np.any(V[:, r]) and np.any(W[:, r])):
            continue
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val == 0: continue
                for k in range(p):
                    v_val = v_col[j * p + k]
                    if v_val == 0: continue
                    w_val = w_col[k * n + i]
                    if w_val == 0: continue
                    recon[i, j, k] += u_val * v_val * w_val
    
    target = np.ones((n, m, p), dtype=int)
    ok = np.array_equal(recon, target)
    
    # Count effective rank
    rank = 0
    for r in range(U.shape[1]):
        if np.any(U[:, r]) and np.any(V[:, r]) and np.any(W[:, r]):
            rank += 1
            
    return ok, rank

def construct():
    base_U, base_V, base_W = _build_base()
    
    # Verify base
    ok, r = _verify(base_U, base_V, base_W)
    if not ok:
        return {"U": base_U.tolist(), "V": base_V.tolist(), "W": base_W.tolist()}
        
    best_U, best_V, best_W = base_U.copy(), base_V.copy(), base_W.copy()
    best_rank = r
    
    # Precompute contributions for the base
    contribs = _compute_contributions(base_U, base_V, base_W)
    R = base_U.shape[1]
    
    # Active columns mask
    active = list(range(R))
    
    # Greedy Reduction with Randomized Restarts
    rng = random.Random(42)
    
    # We will run multiple passes. In each pass, we try to remove columns.
    # To escape local minima, we shuffle the order of checks.
    
    for restart in range(100): # Limit restarts to avoid infinite loop if time allows
        if not _time_ok():
            break
            
        # Shuffle active columns for this pass
        current_active = active.copy()
        rng.shuffle(current_active)
        
        changed = False
        for idx in current_active:
            if idx not in active:
                continue
                
            is_red, _ = _is_redundant(contribs, active, idx)
            if is_red:
                # Remove column
                active.remove(idx)
                changed = True
                # Note: We don't update contribs, we just mark it inactive.
                # The check for other columns uses the current 'active' list.
                
        if changed:
            # Update best if we found a smaller rank
            current_rank = len(active)
            if current_rank < best_rank:
                best_rank = current_rank
                # Create new U, V, W with zeros for removed columns
                best_U = base_U.copy()
                best_V = base_V.copy()
                best_W = base_W.copy()
                for r in range(R):
                    if r not in active:
                        best_U[:, r] = 0
                        best_V[:, r] = 0
                        best_W[:, r] = 0
                        
    # Final verification of the best found
    ok, final_rank = _verify(best_U, best_V, best_W)
    if not ok:
        # Fallback to base if something went wrong
        return {"U": base_U.tolist(), "V": base_V.tolist(), "W": base_W.tolist()}
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
