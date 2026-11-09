import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 10
    mp = m * p  # 25
    pn = p * n  # 10
    
    # Target tensor
    T = np.ones((n, m, p), dtype=float)
    
    start_time = time.time()
    deadline = 110.0
    
    # Try ranks from 49 down to 25
    for rank in range(49, 24, -1):
        if time.time() - start_time > deadline:
            break
        
        np.random.seed(12345 + rank)
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        best_res = float('inf')
        
        for it in range(200):
            if time.time() - start_time > deadline:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    V_s = V[j*p:(j+1)*p, :]  # (p, rank)
                    W_s = np.array([W[k*n+i, :] for k in range(p)])  # (p, rank)
                    M = V_s * W_s  # (p, rank)
                    u, _, _, _ = np.linalg.lstsq(M, T[i, j, :], rcond=None)
                    U[i*m+j, :] = u
            
            # Update V
            for j in range(m):
                for k in range(p):
                    N = np.zeros((n, rank))
                    for i in range(n):
                        N[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    v, _, _, _ = np.linalg.lstsq(N, T[:, j, k], rcond=None)
                    V[j*p+k, :] = v
            
            # Update W
            for k in range(p):
                for i in range(n):
                    P = np.zeros((m, rank))
                    for j in range(m):
                        P[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    w, _, _, _ = np.linalg.lstsq(P, T[i, :, k], rcond=None)
                    W[k*n+i, :] = w
            
            # Residual
            res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        res += (val - 1.0)**2
            
            if res < best_res:
                best_res = res
            
            if best_res < 1e-12:
                break
        
        if best_res < 1e-8:
            # Try rational recovery
            def to_rational(val, tol=1e-7):
                if abs(val) < tol:
                    return 0
                f = Fraction(val).limit_denominator(100000)
                return f
            
            Ur = [[to_rational(x) for x in row] for row in U]
            Vr = [[to_rational(x) for x in row] for row in V]
            Wr = [[to_rational(x) for x in row] for row in W]
            
            # Verify
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = sum(Ur[i*m+j][c] * Vr[j*p+k][c] * Wr[k*n+i][c] for c in range(rank))
                        if abs(float(s) - 1.0) > 1e-9:
                            ok = False
                            break
                if not ok:
                    break
            
            if ok:
                return {"U": Ur, "V": Vr, "W": Wr}
    
    # Schoolbook fallback
    rank = 50
    U = [[0]*rank for _ in range(nm)]
    V = [[0]*rank for _ in range(mp)]
    W = [[0]*rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
