"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    result = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return np.allclose(result, T, atol=tol)

def _make_rank17():
    """Return the known rank-17 decomposition."""
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    terms = [
        ([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]),
        ([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]),
        ([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]),
        ([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]),
        ([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]),
        ([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        ([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]),
        ([0], [1], [2], [1], [4], [1]),
        ([1], [1], [5], [1], [4], [1]),
        ([3], [1], [2], [1], [5], [1]),
        ([4], [1], [5], [1], [5], [1]),
        ([2], [1], [6], [1], [0], [1]),
        ([2], [1], [7], [1], [2], [1]),
        ([2], [1], [8], [1], [4], [1]),
        ([5], [1], [6], [1], [1], [1]),
        ([5], [1], [7], [1], [3], [1]),
        ([5], [1], [8], [1], [5], [1]),
    ]
    
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for a, val in zip(u_idx, u_val):
            U[a, r] = val
        for b, val in zip(v_idx, v_val):
            V[b, r] = val
        for c, val in zip(w_idx, w_val):
            W[c, r] = val
    return U, V, W

def _try_block_rank16(deadline):
    """
    Try to find a rank-16 decomposition by starting from rank-17 and
    removing one term while adjusting others to compensate.
    
    Uses a more systematic local search with gradient-free optimization
    on the entries of U, V, W after dropping one column.
    """
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    rank = 16
    
    # Precompute the list of non-zero tensor entries for faster error computation
    # T has 18 ones (one for each (i,j,k) triple)
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                nz_indices.append((a, b, c))
    
    def compute_error(U, V, W):
        """Compute squared error only at the 18 non-zero positions."""
        err = 0.0
        for a, b, c in nz_indices:
            val = 0.0
            for r in range(rank):
                val += U[a, r] * V[b, r] * W[c, r]
            err += (val - 1.0) ** 2
        # Also check that zero positions remain zero (sample a few)
        # For speed, only check a subset of zero positions
        for _ in range(20):
            a = random.randint(0, n_a-1)
            b = random.randint(0, n_b-1)
            c = random.randint(0, n_c-1)
            # Check if this is a non-zero position
            is_nz = False
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if a == i*m+j and b == j*p+k and c == k*n+i:
                            is_nz = True
            if not is_nz:
                val = 0.0
                for r in range(rank):
                    val += U[a, r] * V[b, r] * W[c, r]
                err += val ** 2
        return err
    
    # Strategy: Try each possible column to drop, then do local search
    for drop in range(17):
        if time.time() > deadline - 10:
            break
        
        U, V, W = _make_rank17()
        U = np.delete(U, drop, axis=1)
        V = np.delete(V, drop, axis=1)
        W = np.delete(W, drop, axis=1)
        
        # Initialize with the dropped column's contribution distributed
        # Start with small random perturbations
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = compute_error(best_U, best_V, best_W)
        
        # Local search: try modifying random entries
        for it in range(300):
            if time.time() > deadline - 5:
                break
            
            # Pick which matrix and which entry to modify
            which = random.randint(0, 2)
            if which == 0:
                a = random.randint(0, n_a-1)
                r = random.randint(0, rank-1)
                old = U[a, r]
            elif which == 1:
                b = random.randint(0, n_b-1)
                r = random.randint(0, rank-1)
                old = V[b, r]
            else:
                c = random.randint(0, n_c-1)
                r = random.randint(0, rank-1)
                old = W[c, r]
            
            # Try a range of candidate values including continuous values
            step = 0.25
            candidates = [old + delta for delta in [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]]
            # Also try integer/half-integer values
            candidates.extend([-1, -0.5, 0, 0.5, 1])
            
            best_val = old
            best_local_err = best_err
            
            for new_val in candidates:
                if which == 0:
                    U[a, r] = new_val
                elif which == 1:
                    V[b, r] = new_val
                else:
                    W[c, r] = new_val
                
                err_new = compute_error(U, V, W)
                
                if err_new < best_local_err - 1e-12:
                    best_local_err = err_new
                    best_val = new_val
                
                # Restore
                if which == 0:
                    U[a, r] = old
                elif which == 1:
                    V[b, r] = old
                else:
                    W[c, r] = old
            
            # Apply best
            if which == 0:
                U[a, r] = best_val
            elif which == 1:
                V[b, r] = best_val
            else:
                W[c, r] = best_val
            
            best_err = best_local_err
            
            if best_err < 1e-10:
                break
        
        if best_err < 1e-10:
            return U, V, W
    
    return None

def _try_structural_rank16(deadline):
    """
    Placeholder for structural search.
    Returns None immediately to prioritize other paths and avoid timeout.
    """
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    
    # First, try to find rank 16 using block-structure-aware search
    T = _build_tensor(n, m, p)
    
    # Try block-structure search for rank 16
    result_16 = _try_block_rank16(deadline)
    if result_16 is not None:
        U, V, W = result_16
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 17
    U, V, W = _make_rank17()
    if _verify_decomposition(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final fallback: schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((n_a, rank_fb))
    V_fb = np.zeros((n_b, rank_fb))
    W_fb = np.zeros((n_c, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
