"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 130
    n, m, p = 2, 4, 6
    
    # Build the target tensor: T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def als_decompose(T, rank, max_iter=100, seed=42):
        np.random.seed(seed)
        U = np.random.randn(T.shape[0], rank) * 0.1
        V = np.random.randn(T.shape[1], rank) * 0.1
        W = np.random.randn(T.shape[2], rank) * 0.1
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U
            for r in range(rank):
                T_comp = np.zeros_like(T)
                for r2 in range(rank):
                    T_comp += np.einsum('i,j,k->ijk', U[:,r2], V[:,r2], W[:,r2])
                R = T - T_comp + np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
                vw = np.outer(V[:,r], W[:,r]).reshape(-1)
                R_reshaped = R.reshape(T.shape[0], -1)
                U[:,r] = R_reshaped @ vw / (vw @ vw)
            
            # Update V
            for r in range(rank):
                T_comp = np.zeros_like(T)
                for r2 in range(rank):
                    T_comp += np.einsum('i,j,k->ijk', U[:,r2], V[:,r2], W[:,r2])
                R = T - T_comp + np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
                uw = np.outer(U[:,r], W[:,r]).reshape(-1)
                R_reshaped = R.reshape(T.shape[1], -1)
                V[:,r] = R_reshaped @ uw / (uw @ uw)
            
            # Update W
            for r in range(rank):
                T_comp = np.zeros_like(T)
                for r2 in range(rank):
                    T_comp += np.einsum('i,j,k->ijk', U[:,r2], V[:,r2], W[:,r2])
                R = T - T_comp + np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
                uv = np.outer(U[:,r], V[:,r]).reshape(-1)
                R_reshaped = R.reshape(T.shape[2], -1)
                W[:,r] = R_reshaped @ uv / (uv @ uv)
            
            # Check convergence
            T_comp = np.zeros_like(T)
            for r in range(rank):
                T_comp += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
            loss = np.sum((T_comp - T)**2)
            if loss < 1e-10:
                break
        
        return U, V, W, loss
    
    # Try different ranks from high to low
    best_result = None
    for rank in [28, 24, 20, 16, 12, 8]:
        if time.time() > deadline:
            break
        
        for trial in range(3):
            if time.time() > deadline:
                break
            
            U, V, W, loss = als_decompose(T, rank, max_iter=50, seed=42+trial)
            
            if loss < 1e-8:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify exactness
                T_comp = np.zeros_like(T)
                for r in range(rank):
                    T_comp += np.einsum('i,j,k->ijk', U_r[:,r], V_r[:,r], W_r[:,r])
                
                if np.allclose(T_comp, T, atol=1e-9):
                    if best_result is None or rank < best_result[0]:
                        best_result = (rank, U_r, V_r, W_r)
                    break
    
    if best_result is not None:
        rank, U, V, W = best_result
        U_out = [[float(x) for x in row] for row in U]
        V_out = [[float(x) for x in row] for row in V]
        W_out = [[float(x) for x in row] for row in W]
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
