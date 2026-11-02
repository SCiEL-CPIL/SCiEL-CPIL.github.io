import numpy as np
import time
import random

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140.0 # Safety margin
    
    # Target: Find rank 27 decomposition. Fallback to 28.
    
    # 1. Prepare the rank 28 decomposition (Strassen)
    # We reuse the logic from the previous successful run to ensure correctness.
    
    strassen = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    U28 = np.zeros((n * m, 28), dtype=int)
    V28 = np.zeros((m * p, 28), dtype=int)
    W28 = np.zeros((p * n, 28), dtype=int)
    
    u_map_L = [0, 1, 4, 5]
    u_map_R = [2, 3, 6, 7]
    v_map_TL = [0, 1, 4, 5]
    v_map_TR = [2, 3, 6, 7]
    v_map_BL = [8, 9, 12, 13]
    v_map_BR = [10, 11, 14, 15]
    w_map_L = [0, 2, 1, 3]
    w_map_R = [4, 6, 5, 7]
    
    col = 0
    def apply_block(u_map, v_map, w_map):
        nonlocal col
        for u_t, v_t, w_t in strassen:
            for idx, val in u_t:
                U28[u_map[idx], col] += val
            for idx, val in v_t:
                V28[v_map[idx], col] += val
            for idx, val in w_t:
                W28[w_map[idx], col] += val
            col += 1
            
    apply_block(u_map_L, v_map_TL, w_map_L)
    apply_block(u_map_R, v_map_BL, w_map_L)
    apply_block(u_map_L, v_map_TR, w_map_R)
    apply_block(u_map_R, v_map_BR, w_map_R)

    # Verify 28 is correct (sanity check)
    def check_decomp(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # C_ik = sum_r W_{k*n+i, r} (sum_j U_{i*m+j, r} A_ij) (sum_j V_{j*p+k, r} B_jk)
        # We need sum_r W_{k*n+i, r} U_{i*m+j1, r} V_{j2*p+k, r} = delta(j1, j2)
        # For all i, k, j1, j2
        r = U.shape[1]
        # Flatten tensors for faster check
        # Eq: Sum_r W_{k, i, r} U_{i, j1, r} V_{j2, k, r} = delta(j1, j2)
        # Indices: i in [0, n), k in [0, p), j1 in [0, m), j2 in [0, m)
        # W index: k * n + i
        # U index: i * m + j1
        # V index: j2 * p + k
        
        for i in range(n):
            for k in range(p):
                w_idx = k * n + i
                for j1 in range(m):
                    u_idx = i * m + j1
                    for j2 in range(m):
                        v_idx = j2 * p + k
                        # Sum over r
                        acc = np.dot(W[w_idx, :], U[u_idx, :] * V[v_idx, :])
                        expected = 1 if j1 == j2 else 0
                        if not np.isclose(acc, expected, atol=1e-9):
                            return False
        return True

    if not check_decomp(U28, V28, W28):
        # Should not happen
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}

    # 2. Attempt to find rank 27 decomposition
    # Strategy: Random search with local refinement.
    # We will try to optimize a random initialization of U, V, W with rank 27.
    # Objective: Minimize || T_decomp - T_target ||^2
    
    def generate_random_init(rank, seed=42):
        np.random.seed(seed)
        # Initialize with small integers
        U_r = np.random.randint(-2, 3, size=(n * m, rank))
        V_r = np.random.randint(-2, 3, size=(m * p, rank))
        W_r = np.random.randint(-2, 3, size=(p * n, rank))
        return U_r, V_r, W_r

    def error_metric(U, V, W):
        # Compute the error of the decomposition
        r = U.shape[1]
        err = 0.0
        for i in range(n):
            for k in range(p):
                w_idx = k * n + i
                for j1 in range(m):
                    u_idx = i * m + j1
                    for j2 in range(m):
                        v_idx = j2 * p + k
                        # acc = sum_r W[w_idx, r] * U[u_idx, r] * V[v_idx, r]
                        # Using dot product for speed
                        acc = np.dot(W[w_idx, :], U[u_idx, :] * V[v_idx, :])
                        expected = 1 if j1 == j2 else 0
                        err += (acc - expected) ** 2
        return err

    best_rank = 28
    best_U, best_V, best_W = U28, V28, W28

    # Try a few random restarts for rank 27
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit - 20:
            break
            
        np.random.seed(1000 + restart)
        U_r, V_r, W_r = generate_random_init(27, seed=1000+restart)
        
        # Local search: Gradient descent with small steps
        # Simplified: Random perturbation
        current_err = error_metric(U_r, V_r, W_r)
        best_err = current_err
        best_U_r, best_V_r, best_W_r = U_r.copy(), V_r.copy(), W_r.copy()
        
        # Perform a short random walk
        steps = 100
        for step in range(steps):
            if time.time() - start_time > time_limit - 20:
                break
            
            # Perturb one random element
            mat_choice = np.random.choice(3)
            if mat_choice == 0:
                idx = np.random.choice(U_r.size)
                old_val = U_r.flat[idx]
                U_r.flat[idx] = old_val + np.random.choice([-1, 1])
            elif mat_choice == 1:
                idx = np.random.choice(V_r.size)
                old_val = V_r.flat[idx]
                V_r.flat[idx] = old_val + np.random.choice([-1, 1])
            else:
                idx = np.random.choice(W_r.size)
                old_val = W_r.flat[idx]
                W_r.flat[idx] = old_val + np.random.choice([-1, 1])
                
            new_err = error_metric(U_r, V_r, W_r)
            if new_err < best_err:
                best_err = new_err
                best_U_r, best_V_r, best_W_r = U_r.copy(), V_r.copy(), W_r.copy()
                if best_err < 1e-6:
                    # Found a very good candidate, round and verify
                    U_int = np.round(best_U_r).astype(int)
                    V_int = np.round(best_V_r).astype(int)
                    W_int = np.round(best_W_r).astype(int)
                    if check_decomp(U_int, V_int, W_int):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
            else:
                # Revert if worse (or accept with some probability, but let's be greedy for now)
                if mat_choice == 0:
                    U_r.flat[idx] = old_val
                elif mat_choice == 1:
                    V_r.flat[idx] = old_val
                else:
                    W_r.flat[idx] = old_val
                    
        # If we got close but not exact, try rounding the best found in this restart
        if best_err < 1.0:
            U_int = np.round(best_U_r).astype(int)
            V_int = np.round(best_V_r).astype(int)
            W_int = np.round(best_W_r).astype(int)
            if check_decomp(U_int, V_int, W_int):
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}

    # If rank 27 not found, return rank 28
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    # Test with random matrices to ensure correctness
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
            
    if not np.allclose(C, Cf):
        print("Verification Failed")
        print(C)
        print(Cf)
    else:
        print("Verification Passed")
    print("Rank:", U.shape[1])
