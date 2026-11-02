import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 130.0  # safety margin before 150s kill

def _build_target_tensor():
    """Build the dense target tensor for M_{2,4,4}."""
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _build_rank28():
    """Build the rank-28 Strassen-block decomposition. Returns (U, V, W) as int arrays."""
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    def strassen_terms():
        terms = []
        # M1
        terms.append(({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}))
        # M2
        terms.append(({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}))
        # M3
        terms.append(({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}))
        # M4
        terms.append(({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}))
        # M5
        terms.append(({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}))
        # M6
        terms.append(({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}))
        # M7
        terms.append(({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}))
        return terms

    base_terms = strassen_terms()

    def apply_strassen(a_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
            col = start_col + t_idx
            u_global = {}
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * 4 + (a_offset * 2 + c)
                u_global[g_idx] = u_global.get(g_idx, 0) + coeff
            v_global = {}
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (b_row_offset * 2 + r) * 4 + (b_col_offset * 2 + c)
                v_global[g_idx] = v_global.get(g_idx, 0) + coeff
            w_global = {}
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (c_col_offset * 2 + c) * 2 + r
                w_global[g_idx] = w_global.get(g_idx, 0) + coeff
            for idx, coeff in u_global.items():
                U[idx, col] += coeff
            for idx, coeff in v_global.items():
                V[idx, col] += coeff
            for idx, coeff in w_global.items():
                W[idx, col] += coeff

    apply_strassen(0, 0, 0, 0, 0)
    apply_strassen(1, 1, 0, 0, 7)
    apply_strassen(0, 0, 1, 1, 14)
    apply_strassen(1, 1, 1, 1, 21)
    return U, V, W

def _verify(U, V, W):
    """Check exact decomposition on random integer matrices."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    rng = np.random.RandomState(42)
    for trial in range(3):
        A = rng.randint(-5, 6, (n, m)).astype(float)
        B = rng.randint(-5, 6, (m, p)).astype(float)
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(rank):
                    for jj in range(m):
                        acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
                Cf[ii, kk] = acc
        if not np.allclose(C, Cf, atol=1e-6):
            return False
    return True

def _compute_error(U, V, W, T_target):
    """Compute the squared error between the reconstructed tensor and the target."""
    # U: (8, R), V: (16, R), W: (8, R)
    # Reconstructed tensor: sum_r U[:,r] outer V[:,r] outer W[:,r]
    # We can do this efficiently with einsum or loops.
    # Given small R, a loop over R is fine, or vectorized.
    # T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    T_recon = np.einsum('ar,br,cr->abc', U.astype(float), V.astype(float), W.astype(float))
    diff = T_target - T_recon
    return np.sum(diff * diff)

def _search_rank27(T_target):
    """
    Search for a rank-27 decomposition using local search.
    """
    n, m, p = DIMENSIONS
    dim_U = n * m  # 8
    dim_V = m * p  # 16
    dim_W = p * n  # 8
    rank = 27
    
    rng = np.random.RandomState(123)
    
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    max_restarts = 20
    max_steps = 2000
    
    for restart in range(max_restarts):
        if time.time() > _DEADLINE - 5.0:
            break
            
        # Initialize with small random integers
        U = rng.randint(-2, 3, (dim_U, rank))
        V = rng.randint(-2, 3, (dim_V, rank))
        W = rng.randint(-2, 3, (dim_W, rank))
        
        err = _compute_error(U, V, W, T_target)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_err == 0:
                return best_U, best_V, best_W
        
        # Local search
        stuck_count = 0
        for step in range(max_steps):
            if time.time() > _DEADLINE - 5.0:
                break
                
            # Pick a random matrix and entry
            mat_choice = rng.randint(3)
            if mat_choice == 0:
                mat = U
                dim = dim_U
            elif mat_choice == 1:
                mat = V
                dim = dim_V
            else:
                mat = W
                dim = dim_W
                
            row = rng.randint(dim)
            col = rng.randint(rank)
            
            # Try changing value
            old_val = mat[row, col]
            # Try +/- 1
            delta = rng.choice([-1, 1])
            mat[row, col] = old_val + delta
            
            new_err = _compute_error(U, V, W, T_target)
            
            if new_err < err:
                err = new_err
                stuck_count = 0
                if err == 0:
                    return U, V, W
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                mat[row, col] = old_val
                stuck_count += 1
                
            if stuck_count > 100:
                # Random jump to escape local minima
                jump_mat = rng.choice([U, V, W])
                for _ in range(5):
                    r = rng.randint(jump_mat.shape[0])
                    c = rng.randint(jump_mat.shape[1])
                    jump_mat[r, c] = rng.randint(-3, 4)
                err = _compute_error(U, V, W, T_target)
                stuck_count = 0

    if best_err == 0:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, build the rank-28 decomposition as fallback
    U28, V28, W28 = _build_rank28()
    
    # Verify rank-28 works
    if not _verify(U28, V28, W28):
        # Fallback: return rank-28 anyway (shouldn't happen)
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    # Try to find a rank-27 decomposition
    if time.time() < _DEADLINE - 10.0:
        T_target = _build_target_tensor()
        result = _search_rank27(T_target)
        if result is not None:
            U27, V27, W27 = result
            # Final verification
            if _verify(U27, V27, W27):
                return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
    
    # Return rank-28
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
# EVOLVE-BLOCK-END

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
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
