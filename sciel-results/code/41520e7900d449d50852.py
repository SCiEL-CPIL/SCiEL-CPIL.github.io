import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=3
    # U: (n*m, rank) = (6, rank), index i*m+j
    # V: (m*p, rank) = (9, rank), index j*p+k
    # W: (p*n, rank) = (6, rank), index k*n+i
    
    def verify_solution(U, V, W):
        """Check if U, V, W give exact decomposition with half-integer entries."""
        if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
            return False
        R = U.shape[1]
        # Check half-integer entries
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-8):
            return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-8):
            return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-8):
            return False
        # Check exact decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    s = 0.0
                    for r in range(R):
                        s += U[u, r] * V[v, r] * W[w, r]
                    if not np.isclose(s, 1.0, atol=1e-8):
                        return False
        return True

    def als_decompose(U_init, V_init, W_init, R, max_iters=500, tol=1e-12):
        """ALS for tensor decomposition."""
        U = U_init.astype(float).copy()
        V = V_init.astype(float).copy()
        W = W_init.astype(float).copy()
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        prev_err = np.inf
        
        for it in range(max_iters):
            # Update U: for each u-index, solve linear system
            for u in range(n * m):
                i = u // m
                j = u % m
                # Equations: for each k, sum_r U[u,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                # This is: sum_r U[u,r] * (V[j*p+k,r]*W[k*n+i,r]) = 1 for each k
                A_mat = np.zeros((p, R))
                for k in range(p):
                    v = j * p + k
                    w = k * n + i
                    A_mat[k, :] = V[v, :] * W[w, :]
                b = np.ones(p)
                try:
                    U[u, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                except:
                    pass
            
            # Update V: for each v-index
            for v in range(m * p):
                j = v // p
                k = v % p
                # Equations: for each i, sum_r U[i*m+j,r]*V[v,r]*W[k*n+i,r] = 1
                A_mat = np.zeros((n, R))
                for i in range(n):
                    u = i * m + j
                    w = k * n + i
                    A_mat[i, :] = U[u, :] * W[w, :]
                b = np.ones(n)
                try:
                    V[v, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                except:
                    pass
            
            # Update W: for each w-index
            for w in range(p * n):
                k = w // n
                i = w % n
                # Equations: for each j, sum_r U[i*m+j,r]*V[j*p+k,r]*W[w,r] = 1
                A_mat = np.zeros((m, R))
                for j in range(m):
                    u = i * m + j
                    v = j * p + k
                    A_mat[j, :] = U[u, :] * V[v, :]
                b = np.ones(m)
                try:
                    W[w, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                except:
                    pass
            
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
        
        U, V, W = best_U, best_V, best_W
        
        # Round to half-integers and verify
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        if verify_solution(U_round, V_round, W_round):
            return U_round, V_round, W_round
            
        # Try with different rounding
        for offset in [0, 0.5, -0.5]:
            U_round = np.round((U + offset) * 2) / 2 - offset
            V_round = np.round((V + offset) * 2) / 2 - offset
            W_round = np.round((W + offset) * 2) / 2 - offset
            if verify_solution(U_round, V_round, W_round):
                return U_round, V_round, W_round
        
        return None

    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[u, r] * V[v, r] * W[w, r]
                    err += (s - 1.0) ** 2
        return err

    start_time = time.time()
    deadline = 140.0
    
    # Schoolbook solution (rank 18)
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
    
    best_result = (U_school, V_school, W_school)
    best_rank = rank_school
    
    # Try to find lower rank solutions
    for R in range(9, 18):
        if time.time() > deadline - 20:
            break
        
        found = False
        for restart in range(30):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(restart * 7919 + R * 104729)
            
            # Multiple initialization strategies
            if restart % 3 == 0:
                # Random small
                U_init = np.random.randn(n * m, R) * 0.3
                V_init = np.random.randn(m * p, R) * 0.3
                W_init = np.random.randn(p * n, R) * 0.3
            elif restart % 3 == 1:
                # Random with some structure
                U_init = np.random.randn(n * m, R) * 0.5
                V_init = np.random.randn(m * p, R) * 0.5
                W_init = np.random.randn(p * n, R) * 0.5
            else:
                # Sparse-like initialization
                U_init = np.zeros((n * m, R))
                V_init = np.zeros((m * p, R))
                W_init = np.zeros((p * n, R))
                for r in range(R):
                    # Randomly assign 1s
                    for _ in range(2):
                        U_init[np.random.randint(n * m), r] = np.random.choice([-1, 1])
                    for _ in range(2):
                        V_init[np.random.randint(m * p), r] = np.random.choice([-1, 1])
                    for _ in range(2):
                        W_init[np.random.randint(p * n), r] = np.random.choice([-1, 1])
            
            res = als_decompose(U_init, V_init, W_init, R, max_iters=300)
            if res is not None:
                U_sol, V_sol, W_sol = res
                best_result = (U_sol, V_sol, W_sol)
                best_rank = R
                found = True
                break
        
        if found:
            # Try to go even lower
            continue
    
    # If we found something better than schoolbook, return it
    if best_rank < rank_school:
        U, V, W = best_result
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
