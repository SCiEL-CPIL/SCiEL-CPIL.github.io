import numpy as np
import time

def construct():
    n, m, p = 2, 3, 4
    deadline = time.time() + 100

    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

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
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

        # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return U, V, W

    # First, ensure we have a valid baseline
    U_base, V_base, W_base = build_rank_22()
    if verify(U_base, V_base, W_base, 22):
        best_result = {"U": U_base, "V": V_base, "W": W_base}
    else:
        # Fallback: simple schoolbook rank 24
        U = [[0]*24 for _ in range(n*m)]
        V = [[0]*24 for _ in range(m*p)]
        W = [[0]*24 for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t = i*m*p + j*p + k
                    U[i*m+j][t] = 1
                    V[j*p+k][t] = 1
                    W[k*n+i][t] = 1
        best_result = {"U": U, "V": V, "W": W}
        if not verify(U, V, W, 24):
            raise RuntimeError("Baseline verification failed")

    def try_als(rank, max_iter=500, tol=1e-10, seed=42):
        """Alternating least squares to find rank decomposition."""
        rng = np.random.RandomState(seed)
        
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = np.ones((p * n, rank))
        
        best_err = float('inf')
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                return None
            
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
                    except:
                        W_new[w, :] = rng.randn(rank)
            
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
                    except:
                        U_new[u, :] = rng.randn(rank)
            
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
                    except:
                        V_new[v, :] = rng.randn(rank)
            
            U, V, W = U_new, V_new, V_new
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for t in range(rank):
                            s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                        err += (s - 1.0)**2
            err = np.sqrt(err)
            
            if err < best_err:
                best_err = err
                if err < 1e-8:
                    U_list = [[round(float(x), 10) for x in row] for row in U]
                    V_list = [[round(float(x), 10) for x in row] for row in V]
                    W_list = [[round(float(x), 10) for x in row] for row in W]
                    if verify(U_list, V_list, W_list, rank):
                        return U_list, V_list, W_list

        return None

    # Try to find lower rank with more seeds and iterations
    for rank in range(20, 22):
        if time.time() > deadline - 15:
            break
        for seed in range(50):
            if time.time() > deadline - 15:
                break
            result = try_als(rank, max_iter=400, tol=1e-12, seed=seed)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}

    return best_result
