import numpy as np
import time

def construct():
    start_time = time.time()
    max_time = 145  # Safety margin before 150s kill

    n, m, p = 2, 3, 4

    # Build the proven rank-22 construction
    def build_rank_22():
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val

        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        # A indices: 00:0, 01:1, 10:3, 11:4
        # B indices: 00:0, 01:1, 10:4, 11:5
        # C indices (W): 00:0, 01:2, 10:1, 11:3

        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        # B indices: 02:2, 03:3, 12:6, 13:7
        # C indices (W): 02:4, 03:6, 12:5, 13:7

        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

        # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
        # A indices: 02:2, 12:5
        # B indices: 20:8, 21:9, 22:10, 23:11
        # C indices: 00:0, 01:2, 02:4, 03:6 for i=0
        #            10:1, 11:3, 12:5, 13:7 for i=1

        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)

        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return U, V, W, rank

    U22, V22, W22, _ = build_rank_22()

    # Convert to numpy arrays
    U22_np = np.array(U22, dtype=float)
    V22_np = np.array(V22, dtype=float)
    W22_np = np.array(W22, dtype=float)

    def verify(U, V, W, rank):
        """Check if the decomposition is exact using vectorized numpy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Build tensor: T[i,j,k,t] = U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
        # We need sum_t T[i,j,k,t] = 1 for all i,j,k
        
        # Efficient check: for each t, compute the outer product structure
        for t in range(rank):
            u_t = U_arr[:, t]  # shape (n*m,)
            v_t = V_arr[:, t]  # shape (m*p,)
            w_t = W_arr[:, t]  # shape (p*n,)
            
            # Reshape u_t to (n, m), v_t to (m, p), w_t to (p, n)
            u_t_2d = u_t.reshape(n, m)
            v_t_2d = v_t.reshape(m, p)
            w_t_2d = w_t.reshape(p, n)
            
            # The contribution to C[i,k] from term t is:
            # sum_j u_t_2d[i,j] * v_t_2d[j,k] * w_t_2d[k,i]
            # This is a cyclic sum: sum_j (u_t_2d[i,j] * v_t_2d[j,k]) * w_t_2d[k,i]
            # = sum_j (u_t_2d @ v_t_2d)[i,k] * w_t_2d[k,i]
            # = trace-like: sum_{i,k} (u_t_2d @ v_t_2d)[i,k] * w_t_2d[k,i]
            
            # Actually let's just check each (i,j,k) triple
            # But that's slow. Let's use the fact that we need:
            # sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
            
            # Vectorized: for fixed t, compute matrix M[i,k] = sum_j U[i*m+j,t]*V[j*p+k,t]
            # Then check sum_t M_t[i,k] * W[k*n+i,t] = 1 for all i,k
            
            # M_t[i,k] = sum_j u_t_2d[i,j] * v_t_2d[j,k] = (u_t_2d @ v_t_2d)[i,k]
            M_t = u_t_2d @ v_t_2d  # shape (n, p)
            
            # Now check: for each i,k, M_t[i,k] * w_t_2d[k,i] should sum to 1 over t
            # But we're checking per t, so we need to accumulate
            # Let's just do the full check at the end instead
        
        # Full vectorized check
        total = np.zeros((n, p))
        for t in range(rank):
            u_t = U_arr[:, t].reshape(n, m)
            v_t = V_arr[:, t].reshape(m, p)
            w_t = W_arr[:, t].reshape(p, n)
            M_t = u_t @ v_t  # (n, p)
            # Element (i,k) gets contribution M_t[i,k] * w_t[k,i]
            # w_t[k,i] = W_arr[k*n+i, t]
            for i in range(n):
                for k in range(p):
                    total[i, k] += M_t[i, k] * w_t[k, i]
        
        return np.allclose(total, 1.0, atol=1e-9)

    def snap_to_half_integer(arr, tol=1e-6):
        """Snap values to nearest half-integer if close."""
        snapped = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                nearest_int = round(val)
                if abs(val - nearest_int) < tol:
                    snapped[i, j] = nearest_int
                else:
                    nearest_half = round(val * 2) / 2.0
                    if abs(val - nearest_half) < tol:
                        snapped[i, j] = nearest_half
                    else:
                        snapped[i, j] = val
        return snapped

    def try_als(rank, max_iter=300, tol=1e-12, seed=42, init_U=None, init_V=None, init_W=None):
        """Alternating least squares to find rank decomposition."""
        rng = np.random.RandomState(seed)
        
        if init_U is not None:
            U = init_U.copy()
            V = init_V.copy()
            W = init_W.copy()
        else:
            U = rng.randn(n * m, rank) * 0.1
            V = rng.randn(m * p, rank) * 0.1
            W = rng.randn(p * n, rank) * 0.1

        for iteration in range(max_iter):
            if time.time() - start_time > max_time - 5:
                return None
            
            # Solve for W: fix U, V
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W_new[w, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[w, :] = rng.randn(rank) * 0.1
            
            # Solve for U: fix V, W
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U_new[u, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[u, :] = rng.randn(rank) * 0.1
            
            # Solve for V: fix U, W
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_new[v, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[v, :] = rng.randn(rank) * 0.1
            
            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new
            
            if diff < tol:
                U_list = [[round(float(x), 10) for x in row] for row in U]
                V_list = [[round(float(x), 10) for x in row] for row in V]
                W_list = [[round(float(x), 10) for x in row] for row in W]
                if verify(U_list, V_list, W_list, rank):
                    U_snap = snap_to_half_integer(U)
                    V_snap = snap_to_half_integer(V)
                    W_snap = snap_to_half_integer(W)
                    U_list = [[round(float(x), 10) for x in row] for row in U_snap]
                    V_list = [[round(float(x), 10) for x in row] for row in V_snap]
                    W_list = [[round(float(x), 10) for x in row] for row in W_snap]
                    if verify(U_list, V_list, W_list, rank):
                        return U_list, V_list, W_list
                    else:
                        return U_list, V_list, W_list
        return None

    # Try to find lower rank decompositions using ALS with warm start from rank 22
    for rank in range(19, 22):
        if time.time() - start_time > max_time - 10:
            break
        
        best_result = None
        
        # Try warm start from rank 22 solution
        init_U = U22_np[:, :rank]
        init_V = V22_np[:, :rank]
        init_W = W22_np[:, :rank]
        
        result = try_als(rank, max_iter=500, seed=0, init_U=init_U, init_V=init_V, init_W=init_W)
        if result is not None:
            U_list, V_list, W_list = result
            if verify(U_list, V_list, W_list, rank):
                best_result = (U_list, V_list, W_list, rank)
        
        # If warm start failed, try a few random starts
        if best_result is None:
            for seed in range(20):
                if time.time() - start_time > max_time - 5:
                    break
                result = try_als(rank, max_iter=200, seed=seed)
                if result is not None:
                    U_list, V_list, W_list = result
                    if verify(U_list, V_list, W_list, rank):
                        best_result = (U_list, V_list, W_list, rank)
                        break
        
        if best_result is not None:
            U_list, V_list, W_list, r = best_result
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return the proven rank-22 construction
    return {"U": U22, "V": V22, "W": W22}
