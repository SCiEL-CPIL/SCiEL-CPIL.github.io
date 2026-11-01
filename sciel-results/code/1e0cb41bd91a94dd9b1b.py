import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    def verify_full(U, V, W, rank):
        """Verify the full decomposition."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True
    
    # Try rank 44: 4 blocks of rank 11 for 2x3x2
    # Use alternating minimization for 2x3x2
    def solve_2x3x2(r, time_budget):
        n2, m2, p2 = 2, 3, 2
        nm, mp, pn = 6, 6, 4
        
        best_err = float('inf')
        best = None
        
        for seed in range(200):
            if time.time() > time_budget:
                break
            
            np.random.seed(seed * 104729 + r * 31)
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5
            
            for it in range(5000):
                if time.time() > time_budget:
                    break
                
                # Alternating updates
                for i in range(n2):
                    for j in range(m2):
                        A = np.zeros((p2, r))
                        b = np.zeros(p2)
                        for k in range(p2):
                            A[k,:] = V[j*p2+k,:] * W[k*n2+i,:]
                            b[k] = 1.0
                        try:
                            U[i*m2+j,:] = np.linalg.solve(A.T @ A + 1e-10*np.eye(r), A.T @ b)
                        except:
                            pass
                
                for j in range(m2):
                    for k in range(p2):
                        A = np.zeros((n2, r))
                        b = np.zeros(n2)
                        for i in range(n2):
                            A[i,:] = U[i*m2+j,:] * W[k*n2+i,:]
                            b[i] = 1.0
                        try:
                            V[j*p2+k,:] = np.linalg.solve(A.T @ A + 1e-10*np.eye(r), A.T @ b)
                        except:
                            pass
                
                for k in range(p2):
                    for i in range(n2):
                        A = np.zeros((m2, r))
                        b = np.zeros(m2)
                        for j in range(m2):
                            A[j,:] = U[i*m2+j,:] * V[j*p2+k,:]
                            b[j] = 1.0
                        try:
                            W[k*n2+i,:] = np.linalg.solve(A.T @ A + 1e-10*np.eye(r), A.T @ b)
                        except:
                            pass
                
                err = 0.0
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            val = np.dot(U[i*m2+j,:], V[j*p2+k,:] * W[k*n2+i,:])
                            err += (val - 1.0)**2
                
                if err < best_err:
                    best_err = err
                    best = (U.copy(), V.copy(), W.copy())
                
                if err < 1e-14:
                    break
            
            if best_err < 1e-12:
                break
        
        if best_err < 1e-10:
            return best
        return None
    
    # Try to find rank-11 decomposition for 2x3x2
    result = solve_2x3x2(11, deadline - 3)
    
    if result is not None:
        U2, V2, W2 = result
        r = 11
        total_rank = r * 4
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        for block in range(4):
            for c in range(r):
                col = block * r + c
                for i in range(2):
                    for j in range(3):
                        U[i * m + j, col] = U2[i * 3 + j, c]
                for j in range(3):
                    for k in range(2):
                        V[j * p + (block * 2 + k), col] = V2[j * 2 + k, c]
                for k in range(2):
                    for i in range(2):
                        W[(block * 2 + k) * n + i, col] = W2[k * 2 + i, c]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
