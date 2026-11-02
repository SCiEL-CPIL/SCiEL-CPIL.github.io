import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Target tensor T as a vector
    # Index mapping:
    # U: i*m + j
    # V: j*p + k
    # W: k*n + i
    # T[a, b, d] where a in [0, nm), b in [0, mp), d in [0, pn)
    # a = i*m + j
    # b = j'*p + k'
    # d = k''*n + i''
    # T[a,b,d] = 1 if j==j' and i==i'' and k==k'' else 0
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                d = k*n + i
                T[a, b, d] = 1.0
    T_flat = T.flatten()
    
    def unpack(x):
        U = x[:nm*14].reshape(nm, 14)
        V = x[nm*14:nm*14+mp*14].reshape(mp, 14)
        W = x[nm*14+mp*14:].reshape(pn, 14)
        return U, V, W

    def residual(x):
        U, V, W = unpack(x)
        # Compute tensor product sum
        # Reshape U, V, W to (14, nm, 1) etc?
        # sum_c U[:, c] V[:, c] W[:, c]
        # Use einsum or broadcasting
        # T_comp[a, b, d] = sum_c U[a, c] V[b, c] W[d, c]
        T_comp = np.einsum('ac,bc,dc->abd', U, V, W)
        return (T_comp.flatten() - T_flat).ravel()

    # Initial guess
    # Try random initialization
    best_rank = 18
    best_result = None
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    # Try rank 14
    # We can try multiple random restarts
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Random initialization
        x0 = np.random.randn(nm*14 + mp*14 + pn*14) * 0.5
        
        try:
            res = least_squares(residual, x0, method='lm', max_nfev=1000)
            if res.success and np.linalg.norm(res.fun) < 1e-6:
                U, V, W = unpack(res.x)
                # Round to nearest half-integer
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify exactness
                T_comp = np.einsum('ac,bc,dc->abd', U_int, V_int, W_int)
                if np.allclose(T_comp, T, atol=1e-9):
                    # Convert to lists
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
        except Exception:
            pass

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
