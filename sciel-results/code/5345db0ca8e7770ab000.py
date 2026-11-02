import numpy as np
import time

def construct():
    n, m, p = 2, 3, 4
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Tensor T[i, j, k] = 1 for all i,j,k
    T = np.ones((n, m, p))
    
    def verify(U, V, W):
        """Verify decomposition"""
        rank = U.shape[1]
        T_comp = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_comp[i, j, k] = np.dot(
                        U[i*m+j, :], 
                        V[j*p+k, :] * W[k*n+i, :]
                    )
        return np.allclose(T_comp, T, atol=1e-9)
    
    def round_to_half(x):
        return np.round(x * 2) / 2
    
    def try_rank(rank, max_restarts=30):
        """Try to find rank-r decomposition using ALS"""
        nm, mp, pn = n*m, m*p, p*n
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization with small integers
            np.random.seed(42 + restart * 7)
            U = np.random.randint(-1, 2, (nm, rank)).astype(float)
            V = np.random.randint(-1, 2, (mp, rank)).astype(float)
            W = np.random.randint(-1, 2, (pn, rank)).astype(float)
            
            # Ensure at least some non-zero entries in each row
            for _ in range(20):
                U[np.random.randint(nm), np.random.randint(rank)] = np.random.choice([-1.0, 1.0])
                V[np.random.randint(mp), np.random.randint(rank)] = np.random.choice([-1.0, 1.0])
                W[np.random.randint(pn), np.random.randint(rank)] = np.random.choice([-1.0, 1.0])
            
            # ALS iterations
            for it in range(300):
                if time.time() > deadline:
                    break
                
                # Update U: for each (i,j), solve least squares
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, rank))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(M, T[i,j,:], rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, rank))
                        for i in range(n):
                            M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(M, T[:,j,k], rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, rank))
                        for j in range(m):
                            M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(M, T[i,j,k], rcond=None)[0]
                        except:
                            pass
            
            # Round to half-integers
            U_r = round_to_half(U)
            V_r = round_to_half(V)
            W_r = round_to_half(W)
            
            if verify(U_r, V_r, W_r):
                return U_r, V_r, W_r
        
        return None
    
    # Try ranks from 12 upward
    for rank in [12, 13, 14, 15, 16, 18, 20, 24]:
        if time.time() > deadline:
            break
        
        result = try_rank(rank, max_restarts=25 if rank <= 14 else 15)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
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
