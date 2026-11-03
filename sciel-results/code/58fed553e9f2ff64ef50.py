"""Fresh program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

n, m, p = DIMENSIONS
NM, MP, PN = n * m, m * p, p * n

def build_target():
    """Build the target multiplication tensor T of shape (NM, MP, PN)."""
    T = np.zeros((NM, MP, PN), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def compute_error(U, V, W, T):
    """Compute Frobenius norm of error between target and reconstructed tensor."""
    # T_rec[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T_rec = np.einsum('ar,br,cr->abc', U, V, W)
    return np.linalg.norm(T - T_rec)

def try_find_decomposition(R, T, time_budget, rng):
    """Try to find a rank-R decomposition via local search."""
    best_U, best_V, best_W = None, None, None
    best_err = np.inf
    
    # Multiple random restarts
    n_restarts = 0
    while time.time() < time_budget:
        n_restarts += 1
        # Initialize with small random integers
        U = rng.integers(-1, 2, size=(NM, R))
        V = rng.integers(-1, 2, size=(MP, R))
        W = rng.integers(-1, 2, size=(PN, R))
        
        err = compute_error(U, V, W, T)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-9:
                break
        
        # Local search: coordinate descent
        for iteration in range(200):
            if time.time() > time_budget:
                break
            improved = False
            # Try adjusting each entry of U, V, W
            for mat_name, mat in [('U', U), ('V', V), ('W', W)]:
                for idx in range(mat.size):
                    row, col = np.unravel_index(idx, mat.shape)
                    old_val = mat[row, col]
                    for new_val in [old_val - 1, old_val + 1]:
                        if new_val == old_val:
                            continue
                        mat[row, col] = new_val
                        new_err = compute_error(U, V, W, T)
                        if new_err < err:
                            err = new_err
                            improved = True
                            if err < 1e-9:
                                break
                        else:
                            mat[row, col] = old_val
                    if err < 1e-9:
                        break
                if err < 1e-9:
                    break
            if err < 1e-9:
                break
            if not improved and iteration > 50:
                break
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_err < 1e-9:
                break
    
    if best_err < 1e-9:
        # Convert to integers (they should already be close to integers)
        U_int = np.round(best_U).astype(int)
        V_int = np.round(best_V).astype(int)
        W_int = np.round(best_W).astype(int)
        # Verify
        T_rec = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
        if np.allclose(T_rec, T, atol=1e-9):
            return U_int.tolist(), V_int.tolist(), W_int.tolist()
    
    return None

def construct():
    start_time = time.time()
    time_limit = 120  # Keep under 150s
    
    T = build_target()
    
    # Try ranks from 18 upward
    for R in [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 5:
            break
        rng = np.random.default_rng(R * 1000 + 42)
        result = try_find_decomposition(R, T, start_time + remaining - 2, rng)
        if result is not None:
            U_list, V_list, W_list = result
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook rank 36
    rank = n * m * p
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
