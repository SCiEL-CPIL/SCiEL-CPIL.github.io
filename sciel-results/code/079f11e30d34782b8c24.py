import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook
    rank_school = n * m * p
    
    # Build the target tensor non-zeros
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i)
    # u in [0, n*m), v in [0, m*p), w in [0, p*n)
    non_zeros = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                non_zeros.append((u, v, w))
    
    K = len(non_zeros)
    U_dims = n * m
    V_dims = m * p
    W_dims = p * n
    
    def verify_decomp(U, V, W):
        """
        Checks if U, V, W exactly decompose the tensor.
        Returns True if valid.
        """
        r = len(U[0])
        # Check all non-zero entries
        for u, v, w in non_zeros:
            # Sum over c: U[u, c] * V[v, c] * W[w, c]
            prod = 0.0
            for c in range(r):
                prod += U[u][c] * V[v][c] * W[w][c]
            if abs(prod - 1.0) > 1e-9:
                return False
        
        # Check all zero entries
        nz_set = set(non_zeros)
        for u in range(U_dims):
            for v in range(V_dims):
                for w in range(W_dims):
                    if (u, v, w) not in nz_set:
                        prod = 0.0
                        for c in range(r):
                            prod += U[u][c] * V[v][c] * W[w][c]
                        if abs(prod) > 1e-9:
                            return False
        return True

    def round_to_half(X):
        # Round to nearest 0.5
        return np.round(X * 2) / 2

    def al_solve(U, V, W, rank, iters=50, tol=1e-12):
        """
        Alternating Least Squares for CP decomposition.
        Minimizes sum (T - sum U_c V_c W_c)^2
        """
        # Precompute indices grouped by w
        indices_by_w = {i: [] for i in range(W_dims)}
        indices_by_u = {i: [] for i in range(U_dims)}
        indices_by_v = {i: [] for i in range(V_dims)}
        
        for idx, (u, v, w) in enumerate(non_zeros):
            indices_by_w[w].append(idx)
            indices_by_u[u].append(idx)
            indices_by_v[v].append(idx)
            
        for it in range(iters):
            # Update W
            for w in range(W_dims):
                if not indices_by_w[w]:
                    continue
                idxs = indices_by_w[w]
                A = np.empty((len(idxs), rank))
                for r_idx, idx in enumerate(idxs):
                    u, v, _ = non_zeros[idx]
                    A[r_idx, :] = U[u, :] * V[v, :]
                b = np.ones(len(idxs))
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[w, :] = sol
                
            # Update U
            for u in range(U_dims):
                if not indices_by_u[u]:
                    continue
                idxs = indices_by_u[u]
                A = np.empty((len(idxs), rank))
                for r_idx, idx in enumerate(idxs):
                    _, v, w = non_zeros[idx]
                    A[r_idx, :] = V[v, :] * W[w, :]
                b = np.ones(len(idxs))
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[u, :] = sol
                
            # Update V
            for v in range(V_dims):
                if not indices_by_v[v]:
                    continue
                idxs = indices_by_v[v]
                A = np.empty((len(idxs), rank))
                for r_idx, idx in enumerate(idxs):
                    u, _, w = non_zeros[idx]
                    A[r_idx, :] = U[u, :] * W[w, :]
                b = np.ones(len(idxs))
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[v, :] = sol
            
            # Check convergence
            if it > 0:
                err = 0.0
                for idx in range(K):
                    u, v, w = non_zeros[idx]
                    val = np.sum(U[u, :] * V[v, :] * W[w, :])
                    err += (val - 1.0)**2
                if err < tol:
                    break
                    
        return U, V, W

    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    # Try ranks from low to high
    # Start from 30, go up to schoolbook-1
    min_rank_to_try = 30
    max_rank_to_try = rank_school - 1
    
    for r in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline:
            break
            
        # Try a few restarts for this rank
        found = False
        restarts = 3 if r < 40 else 1
        
        for _ in range(restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(r * 100 + _)
            U = np.random.randn(U_dims, r)
            V = np.random.randn(V_dims, r)
            W = np.random.randn(W_dims, r)
            
            U, V, W = al_solve(U, V, W, r, iters=30)
            
            # Check error before rounding
            err = 0.0
            for idx in range(K):
                u, v, w = non_zeros[idx]
                val = np.sum(U[u, :] * V[v, :] * W[w, :])
                err += (val - 1.0)**2
            
            if err < 1e-4:
                Ur = round_to_half(U)
                Vr = round_to_half(V)
                Wr = round_to_half(W)
                
                # Convert to lists for verification
                U_list = Ur.tolist()
                V_list = Vr.tolist()
                W_list = Wr.tolist()
                
                if verify_decomp(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
                
        # If not found, continue to next rank

    # Fallback to schoolbook
    rank = rank_school
    U = np.zeros((U_dims, rank))
    V = np.zeros((V_dims, rank))
    W = np.zeros((W_dims, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u, col] = 1
                V[v, col] = 1
                W[w, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
