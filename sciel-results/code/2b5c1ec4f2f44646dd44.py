import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 105
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school
    
    def check_decomposition(U, V, W, tol=1e-6):
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        for ip in range(n):
            for j in range(m):
                for jp in range(m):
                    for kp in range(p):
                        s = np.zeros((p, n))
                        for r in range(rank):
                            s += W3[:, :, r] * U3[ip, j, r] * V3[jp, kp, r]
                        expected = np.zeros((p, n))
                        expected[kp, ip] = 1.0
                        if np.max(np.abs(s - expected)) > tol:
                            return False
        return True
    
    eq_indices = []
    for ip in range(n):
        for j in range(m):
            for jp in range(m):
                for kp in range(p):
                    for i in range(n):
                        for k in range(p):
                            eq_indices.append((ip, j, jp, kp, i, k))
    num_eq = len(eq_indices)
    
    def solve_W(U, V, rank):
        A_mat = np.zeros((num_eq, p*n*rank))
        b_vec = np.zeros(num_eq)
        for idx, (ip, j, jp, kp, i, k) in enumerate(eq_indices):
            for r in range(rank):
                A_mat[idx, (k*n+i)*rank + r] = U[ip*m+j, r] * V[jp*p+kp, r]
            b_vec[idx] = 1.0 if (ip==i and j==jp and kp==k) else 0.0
        W_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        return W_flat.reshape(p*n, rank)
    
    def solve_V(U, W, rank):
        A_mat = np.zeros((num_eq, m*p*rank))
        b_vec = np.zeros(num_eq)
        for idx, (ip, j, jp, kp, i, k) in enumerate(eq_indices):
            for r in range(rank):
                A_mat[idx, (jp*p+kp)*rank + r] = U[ip*m+j, r] * W[k*n+i, r]
            b_vec[idx] = 1.0 if (ip==i and j==jp and kp==k) else 0.0
        V_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        return V_flat.reshape(m*p, rank)
    
    def solve_U(V, W, rank):
        A_mat = np.zeros((num_eq, n*m*rank))
        b_vec = np.zeros(num_eq)
        for idx, (ip, j, jp, kp, i, k) in enumerate(eq_indices):
            for r in range(rank):
                A_mat[idx, (ip*m+j)*rank + r] = V[jp*p+kp, r] * W[k*n+i, r]
            b_vec[idx] = 1.0 if (ip==i and j==jp and kp==k) else 0.0
        U_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        return U_flat.reshape(n*m, rank)
    
    def compute_error(U, V, W, rank):
        err = 0.0
        W3 = W.reshape(p, n, rank)
        for ip in range(n):
            for j in range(m):
                for jp in range(m):
                    for kp in range(p):
                        s = np.zeros((p, n))
                        for r in range(rank):
                            s += W3[:, :, r] * U[ip*m+j, r] * V[jp*p+kp, r]
                        expected = np.zeros((p, n))
                        expected[kp, ip] = 1.0
                        err += np.sum((s - expected)**2)
        return err
    
    def alternating_minimization(rank, max_iter=80, seed=None):
        if seed is not None:
            np.random.seed(seed)
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        prev_err = float('inf')
        for it in range(max_iter):
            W = solve_W(U, V, rank)
            V = solve_V(U, W, rank)
            U = solve_U(V, W, rank)
            
            err = compute_error(U, V, W, rank)
            if abs(prev_err - err) < 1e-12 or err < 1e-10:
                break
            prev_err = err
        
        return U, V, W, err
    
    for rank in range(15, 30):
        if time.time() > deadline - 10:
            break
        
        found = False
        num_trials = 12
        for trial in range(num_trials):
            if time.time() > deadline - 8:
                break
            
            U, V, W, err = alternating_minimization(rank, max_iter=60, seed=trial*1000+42)
            
            if err < 1e-5:
                if check_decomposition(U, V, W, tol=1e-4):
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    return best
