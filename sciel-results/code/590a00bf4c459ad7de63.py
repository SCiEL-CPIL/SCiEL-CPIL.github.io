import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
                
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_rank = rank_schoolbook
    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook
    
    # Target ranks to try, from lower to higher
    # We try a few ranks just below schoolbook
    target_ranks = [71, 70, 69]
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W exactly decompose the tensor using vectorized ops."""
        r = U.shape[1]
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # We need U[i*m+j, :] . (V[j*p+k, :] * W[k*n+i, :]) == 1
        # Let's construct the full tensor check
        # This is expensive, but vectorized.
        # Alternative: check all (i,j,k)
        for i in range(n):
            for j in range(m):
                # Vector over k
                # U_row = U[i*m+j]
                # V_rows = V[j*p : (j+1)*p] # (p, r)
                # W_cols = W.reshape(p, n)[:, i] # (p, r) -> actually W[k*n+i]
                # W_subset = W[k*n+i] for k in 0..p-1
                # We can extract W_subset = W.reshape(p, n)[:, i].T ?
                # W is (p*n, r). Index k*n+i.
                # W_subset = W.reshape(p, n, r)[:, i, :] # (p, r)
                # Prod = V_rows * W_subset # (p, r)
                # Dot = U_row @ Prod.T # (p,)
                U_row = U[i * m + j]
                V_subset = V[j * p : (j + 1) * p] # (p, r)
                W_subset = W.reshape(p, n, r)[:, i, :] # (p, r)
                Prod = V_subset * W_subset # (p, r)
                Dots = U_row @ Prod.T # (p,)
                if not np.allclose(Dots, np.ones(p), atol=tol):
                    return False
        return True

    def solve_for_U(V, W, r):
        U = np.zeros((n * m, r))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k] = V[j * p + k] * W[k * n + i]
                try:
                    U[i * m + j] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                except:
                    U[i * m + j] = np.random.randn(r)
        return U

    def solve_for_V(U, W, r):
        V = np.zeros((m * p, r))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i] = U[i * m + j] * W[k * n + i]
                try:
                    V[j * p + k] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                except:
                    V[j * p + k] = np.random.randn(r)
        return V

    def solve_for_W(U, V, r):
        W = np.zeros((p * n, r))
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j] = U[i * m + j] * V[j * p + k]
                try:
                    W[k * n + i] = np.linalg.lstsq(A.T, np.ones(m), rcond=None)[0]
                except:
                    W[k * n + i] = np.random.randn(r)
        return W

    np.random.seed(42)
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
            
        found = False
        # Fewer attempts to stay within time
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            for it in range(20):
                if time.time() > deadline:
                    break
                    
                U = solve_for_U(V, W, target_rank)
                V = solve_for_V(U, W, target_rank)
                W = solve_for_W(U, V, target_rank)
                
                if check_decomposition(U, V, W, tol=1e-6):
                    # Refine
                    for refine_it in range(10):
                        U = solve_for_U(V, W, target_rank)
                        V = solve_for_V(U, W, target_rank)
                        W = solve_for_W(U, V, target_rank)
                        if check_decomposition(U, V, W, tol=1e-10):
                            break
                    
                    if check_decomposition(U, V, W, tol=1e-9):
                        best_rank = target_rank
                        best_U = U
                        best_V = V
                        best_W = W
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
