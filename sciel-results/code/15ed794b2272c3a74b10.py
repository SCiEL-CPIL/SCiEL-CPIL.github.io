"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    T_recon = np.zeros_like(T)
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    return np.allclose(T_recon, T, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100
    
    # Try to find a rank-12 decomposition
    # Use a simple iterative approach
    for rank in [12, 13, 14]:
        if time.time() > deadline:
            break
        
        for seed in range(100):
            if time.time() > deadline:
                break
            
            np.random.seed(seed)
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            converged = False
            for it in range(1000):
                if time.time() > deadline:
                    break
                
                # ALS updates
                # Update U
                for a in range(n*m):
                    # T[a] ≈ U[a] @ (V ⊙ W)
                    VW = np.outer(V.T, W.T).reshape(rank, -1)
                    Ta = np.zeros((m*p*p*n,))
                    # Fill Ta from T[a]
                    Ta = np.zeros((m*p, p*n))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if i*m+j == a:
                                    Ta[j*p+k, k*n+i] = 1.0
                    Ta_flat = Ta.reshape(-1)
                    try:
                        U[a] = np.linalg.solve(VW @ VW.T + 1e-8*np.eye(rank), VW @ Ta_flat)
                    except:
                        pass
                
                # Update V
                for b in range(m*p):
                    UW = np.outer(U.T, W.T).reshape(rank, -1)
                    Tb = np.zeros((n*m, p*n))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if j*p+k == b:
                                    Tb[i*m+j, k*n+i] = 1.0
                    Tb_flat = Tb.reshape(-1)
                    try:
                        V[b] = np.linalg.solve(UW @ UW.T + 1e-8*np.eye(rank), UW @ Tb_flat)
                    except:
                        pass
                
                # Update W
                for c in range(p*n):
                    UV = np.outer(U.T, V.T).reshape(rank, -1)
                    Tc = np.zeros((n*m, m*p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if k*n+i == c:
                                    Tc[i*m+j, j*p+k] = 1.0
                    Tc_flat = Tc.reshape(-1)
                    try:
                        W[c] = np.linalg.solve(UV @ UV.T + 1e-8*np.eye(rank), UV @ Tc_flat)
                    except:
                        pass
                
                # Check error
                T_recon = np.zeros((n*m, m*p, p*n))
                for r in range(rank):
                    T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                err = np.linalg.norm(T_recon - T) if False else 0  # skip for speed
                
                if time.time() > deadline:
                    break
            
            # Final check
            if time.time() > deadline:
                break
            
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                # Try to round
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
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
# EVOLVE-BLOCK-END
