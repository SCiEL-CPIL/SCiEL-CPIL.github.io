import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    # Precompute index mappings for ALS
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    
    # For updating U:
    # For each u (i, j), we have equations for k=0..p-1
    # B_u[k, r] = V[j*p+k, r] * W[k*n+i, r]
    U_indices = []
    for i in range(n):
        for j in range(m):
            u = i * m + j
            eqs = []
            for k in range(p):
                v = j * p + k
                w = k * n + i
                eqs.append((v, w))
            U_indices.append((i, j, u, eqs))
            
    # For updating V:
    # For each v (j, k), we have equations for i=0..n-1
    # C_v[i, r] = U[i*m+j, r] * W[k*n+i, r]
    V_indices = []
    for j in range(m):
        for k in range(p):
            v = j * p + k
            eqs = []
            for i in range(n):
                u = i * m + j
                w = k * n + i
                eqs.append((u, w))
            V_indices.append((j, k, v, eqs))
            
    # For updating W:
    # For each w (k, i), we have equations for j=0..m-1
    # D_w[j, r] = U[i*m+j, r] * V[j*p+k, r]
    W_indices = []
    for k in range(p):
        for i in range(n):
            w = k * n + i
            eqs = []
            for j in range(m):
                u = i * m + j
                v = j * p + k
                eqs.append((u, v))
            W_indices.append((k, i, w, eqs))

    def verify_solution(U, V, W):
        # Check if U, V, W are half-integers
        # Using exact integer check on 2*U
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        
        if not np.allclose(U2, np.round(U2), atol=1e-9): return False
        if not np.allclose(V2, np.round(V2), atol=1e-9): return False
        if not np.allclose(W2, np.round(W2), atol=1e-9): return False
        
        # Check exact decomposition
        # Sum_r U[u,r] V[v,r] W[w,r] == 1
        # We only need to check the 18 non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[u, r] * V[v, r] * W[w, r]
                    if not np.isclose(s, 1.0, atol=1e-9):
                        return False
        return True

    def discrete_search(U, V, W, R):
        """
        Attempts to fix a near-solution by flipping half-integer values.
        This is a heuristic local search.
        """
        U_int = np.round(U * 2).astype(int)
        V_int = np.round(V * 2).astype(int)
        W_int = np.round(W * 2).astype(int)
        
        def calc_error(Ui, Vi, Wi):
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = i * m + j
                        v = j * p + k
                        w = k * n + i
                        s = 0.0
                        for r in range(R):
                            s += (Ui[u, r] * 0.5) * (Vi[v, r] * 0.5) * (Wi[w, r] * 0.5)
                        err += (s - 1.0) ** 2
            return err
        
        current_err = calc_error(U_int, V_int, W_int)
        
        # Try flipping single elements
        # This is expensive, so limit the search depth or randomize
        # We only do this if current_err is very small
        if current_err > 1e-6:
            return None
            
        # Randomly try flipping a few elements
        for _ in range(20):
            # Flip a random element in U
            u_idx = np.random.randint(n * m)
            r_idx = np.random.randint(R)
            U_int[u_idx, r_idx] += 1 # flip half-integer step
            new_err = calc_error(U_int, V_int, W_int)
            if new_err < current_err:
                current_err = new_err
            else:
                U_int[u_idx, r_idx] -= 1 # revert
                
            # Flip a random element in V
            v_idx = np.random.randint(m * p)
            r_idx = np.random.randint(R)
            V_int[v_idx, r_idx] += 1
            new_err = calc_error(U_int, V_int, W_int)
            if new_err < current_err:
                current_err = new_err
            else:
                V_int[v_idx, r_idx] -= 1
                
            # Flip a random element in W
            w_idx = np.random.randint(p * n)
            r_idx = np.random.randint(R)
            W_int[w_idx, r_idx] += 1
            new_err = calc_error(U_int, V_int, W_int)
            if new_err < current_err:
                current_err = new_err
            else:
                W_int[w_idx, r_idx] -= 1
                
        if current_err < 1e-9:
            return (U_int * 0.5, V_int * 0.5, W_int * 0.5)
        return None

    def als_decompose(R, max_iters=100, tol=1e-12):
        # Random initialization
        U = np.random.randint(-2, 3, (n * m, R)) * 0.5
        V = np.random.randint(-2, 3, (m * p, R)) * 0.5
        W = np.random.randint(-2, 3, (p * n, R)) * 0.5
        
        prev_err = np.inf
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            # Update U
            for i, j, u, eqs in U_indices:
                B = np.zeros((len(eqs), R))
                for idx, (v, w) in enumerate(eqs):
                    B[idx, :] = V[v, :] * W[w, :]
                # Solve B @ U_u = 1
                # B is 3xR, U_u is Rx1
                # Use lstsq
                U_u, _, _, _ = np.linalg.lstsq(B, np.ones(len(eqs)), rcond=None)
                U[u, :] = U_u
                
            # Update V
            for j, k, v, eqs in V_indices:
                C = np.zeros((len(eqs), R))
                for idx, (u, w) in enumerate(eqs):
                    C[idx, :] = U[u, :] * W[w, :]
                V_v, _, _, _ = np.linalg.lstsq(C, np.ones(len(eqs)), rcond=None)
                V[v, :] = V_v
                
            # Update W
            for k, i, w, eqs in W_indices:
                D = np.zeros((len(eqs), R))
                for idx, (u, v) in enumerate(eqs):
                    D[idx, :] = U[u, :] * V[v, :]
                W_w, _, _, _ = np.linalg.lstsq(D, np.ones(len(eqs)), rcond=None)
                W[w, :] = W_w
            
            # Calculate error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = i * m + j
                        v = j * p + k
                        w = k * n + i
                        s = 0.0
                        for r in range(R):
                            s += U[u, r] * V[v, r] * W[w, r]
                        err += (s - 1.0) ** 2
                        
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
            # Early exit if error is effectively zero
            if err < 1e-10:
                break
        
        U, V, W = best_U, best_V, best_W
        
        # Round to half-integers and verify
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        if verify_solution(U_round, V_round, W_round):
            return U_round, V_round, W_round
            
        # Try discrete search if close
        if best_err < 1e-4:
            res = discrete_search(U, V, W, R)
            if res:
                return res

        return None

    start_time = time.time()
    deadline = 148 # Safety margin for 150s limit

    # Try rank 16 first
    # If we find a solution, return it
    # We will run multiple restarts
    
    # Rank 16
    max_restarts_16 = 500
    for restart in range(max_restarts_16):
        if time.time() > deadline:
            break
        res = als_decompose(16)
        if res is not None:
            U, V, W = res
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Rank 17
    max_restarts_17 = 200
    for restart in range(max_restarts_17):
        if time.time() > deadline:
            break
        res = als_decompose(17)
        if res is not None:
            U, V, W = res
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
