import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    def check_solution(U, V, W, R):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err += (val - 1.0)**2
        return err

    def solve_for_U(V, W, R):
        U = np.zeros((n*m, R))
        V_reshaped = V.reshape(m, p, R)
        W_reshaped = W.reshape(p, n, R)
        for j in range(m):
            for i in range(n):
                A = V_reshaped[j] * W_reshaped[:, i]
                b = np.ones(p)
                try:
                    U[i*m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return U

    def solve_for_V(U, W, R):
        V = np.zeros((m*p, R))
        U_reshaped = U.reshape(n, m, R)
        W_reshaped = W.reshape(p, n, R)
        for j in range(m):
            for k in range(p):
                A = U_reshaped[:, j] * W_reshaped[k]
                b = np.ones(n)
                try:
                    V[j*p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return V

    def solve_for_W(U, V, R):
        W = np.zeros((p*n, R))
        U_reshaped = U.reshape(n, m, R)
        V_reshaped = V.reshape(m, p, R)
        for k in range(p):
            for i in range(n):
                A = U_reshaped[i] * V_reshaped[:, k]
                b = np.ones(m)
                try:
                    W[k*n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return W

    def is_exact_solution(U, V, W, R):
        # Check if the decomposition is exact
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True

    def round_to_half_int(M):
        return np.round(M * 2) / 2.0

    best_rank = n*m*p
    best_U, best_V, best_W = None, None, None

    # Try ranks from 22 to 29
    for R in range(22, 30):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(200): # Increased trials due to faster solvers
            if time.time() > deadline:
                break
            
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Iterate
            for it in range(50):
                if time.time() > deadline:
                    break
                
                U = solve_for_U(V, W, R)
                V = solve_for_V(U, W, R)
                W = solve_for_W(U, V, R)
                
                # Check for exact solution after each iteration? 
                # It's expensive, but n*m*p=30 checks.
                # Let's check every 5 iterations or at the end.
                # Checking at the end is safer for time, but might miss early convergence.
                # Let's check every 10 iterations.
                if it % 10 == 0:
                    U_rounded = round_to_half_int(U)
                    V_rounded = round_to_half_int(V)
                    W_rounded = round_to_half_int(W)
                    if is_exact_solution(U_rounded, V_rounded, W_rounded, R):
                        best_rank = R
                        best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                        break
            if best_rank == R:
                break
            
            # Check one more time at the end of the trial
            U_rounded = round_to_half_int(U)
            V_rounded = round_to_half_int(V)
            W_rounded = round_to_half_int(W)
            if is_exact_solution(U_rounded, V_rounded, W_rounded, R):
                best_rank = R
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                break
        if best_rank == R:
            break
    
    if best_U is None:
        # Fall back to schoolbook
        R = n*m*p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
