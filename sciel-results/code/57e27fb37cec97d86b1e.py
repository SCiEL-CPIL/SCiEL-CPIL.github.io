import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _verify_exact(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    
    # Try to find rank 41 decomposition (one better than schoolbook)
    # Use a targeted search
    
    target_rank = 41
    # Initialize with schoolbook and try to merge two columns
    # This is complex, so let's use a different strategy
    
    # Strategy: Randomized local search with integer values
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        U = [[np.random.randint(-3, 4) for _ in range(target_rank)] for _ in range(n*m)]
        V = [[np.random.randint(-3, 4) for _ in range(target_rank)] for _ in range(m*p)]
        W = [[np.random.randint(-3, 4) for _ in range(target_rank)] for _ in range(p*n)]
        
        for it in range(500):
            if time.time() > deadline:
                break
            
            # Compute all errors
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(target_rank))
                        errors.append((val - 1, i, j, k))
            
            max_err = max(abs(e[0]) for e in errors)
            if max_err < 1e-10:
                U_int = [[int(x) for x in row] for row in U]
                V_int = [[int(x) for x in row] for row in V]
                W_int = [[int(x) for x in row] for row in W]
                if _verify_exact(U_int, V_int, W_int, n, m, p):
                    return {"U": U_int, "V": V_int, "W": W_int}
            
            # Fix worst error
            worst = max(errors, key=lambda x: abs(x[0]))
            err, i, j, k = worst
            
            # Try adjusting W
            for c in range(target_rank):
                a = U[i*m+j][c] * V[j*p+k][c]
                if a == 0:
                    continue
                old = W[k*n+i][c]
                S = sum(U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc] for cc in range(target_rank))
                new_val = (1 - S + a * old) / a
                nv = int(round(new_val))
                if abs(new_val - nv) < 1e-9 and abs(nv) <= 15:
                    W[k*n+i][c] = nv
                    # Quick check
                    new_err = sum(U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc] for cc in range(target_rank)) - 1
                    if abs(new_err) < abs(err) * 0.5:
                        pass
                    else:
                        W[k*n+i][c] = old
                        break
            
            # Try adjusting U
            for c in range(target_rank):
                a = V[j*p+k][c] * W[k*n+i][c]
                if a == 0:
                    continue
                old = U[i*m+j][c]
                S = sum(U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc] for cc in range(target_rank))
                new_val = (1 - S + a * old) / a
                nv = int(round(new_val))
                if abs(new_val - nv) < 1e-9 and abs(nv) <= 15:
                    U[i*m+j][c] = nv
                    new_err = sum(U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc] for cc in range(target_rank)) - 1
                    if abs(new_err) < abs(err) * 0.5:
                        pass
                    else:
                        U[i*m+j][c] = old
                        break
            
            # Try adjusting V
            for c in range(target_rank):
                a = U[i*m+j][c] * W[k*n+i][c]
                if a == 0:
                    continue
                old = V[j*p+k][c]
                S = sum(U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc] for cc in range(target_rank))
                new_val = (1 - S + a * old) / a
                nv = int(round(new_val))
                if abs(new_val - nv) < 1e-9 and abs(nv) <= 15:
                    V[j*p+k][c] = nv
                    new_err = sum(U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc] for cc in range(target_rank)) - 1
                    if abs(new_err) < abs(err) * 0.5:
                        pass
                    else:
                        V[j*p+k][c] = old
                        break
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
