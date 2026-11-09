"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                if abs(val - 1) > tol:
                    return False
    
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i1, j1 = divmod(a, m)
                j2, k1 = divmod(b, p)
                k2, i2 = divmod(c, n)
                if i1 == i2 and j1 == j2 and k1 == k2:
                    continue
                val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                if abs(val) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS
    
    # Precompute RHS
    rhs_U = np.zeros((n*m, m*p*p*n))
    for a in range(n*m):
        i1, j1 = divmod(a, m)
        for b in range(m*p):
            j2, k1 = divmod(b, p)
            for c in range(p*n):
                k2, i2 = divmod(c, n)
                if i1 == i2 and j1 == j2 and k1 == k2:
                    rhs_U[a, b*(p*n)+c] = 1.0
    
    rhs_V = np.zeros((m*p, n*m*p*n))
    for b in range(m*p):
        j1, k1 = divmod(b, p)
        for a in range(n*m):
            i1, j2 = divmod(a, m)
            for c in range(p*n):
                k2, i2 = divmod(c, n)
                if i1 == i2 and j2 == j1 and k1 == k2:
                    rhs_V[b, a*(p*n)+c] = 1.0
    
    rhs_W = np.zeros((p*n, n*m*m*p))
    for c in range(p*n):
        k1, i1 = divmod(c, n)
        for a in range(n*m):
            i2, j1 = divmod(a, m)
            for b in range(m*p):
                j2, k2 = divmod(b, p)
                if i1 == i2 and j1 == j2 and k1 == k2:
                    rhs_W[c, a*(m*p)+b] = 1.0
    
    for target_rank in range(23, 54):
        if time.time() > deadline - 5:
            break
        remaining = deadline - time.time()
        if remaining < 3:
            break
        
        rng = np.random.RandomState(target_rank * 31 + 17)
        best_loss = float('inf')
        best_U = best_V = best_W = None
        
        n_attempts = min(60, max(5, int(remaining * 2)))
        
        for attempt in range(n_attempts):
            if time.time() - start_time > deadline - 2:
                break
            
            U = rng.randn(n*m, target_rank) * 0.5
            V = rng.randn(m*p, target_rank) * 0.5
            W = rng.randn(p*n, target_rank) * 0.5
            
            for it in range(100):
                if time.time() - start_time > deadline - 2:
                    break
                
                for a in range(n*m):
                    A_sys = np.outer(V.ravel(), W.ravel()).reshape(m*p*p*n, target_rank)
                    U[a, :], _, _, _ = np.linalg.lstsq(A_sys, rhs_U[a], rcond=None)
                
                for b in range(m*p):
                    A_sys = np.outer(U.ravel(), W.ravel()).reshape(n*m*p*n, target_rank)
                    V[b, :], _, _, _ = np.linalg.lstsq(A_sys, rhs_V[b], rcond=None)
                
                for c in range(p*n):
                    A_sys = np.outer(U.ravel(), V.ravel()).reshape(n*m*m*p, target_rank)
                    W[c, :], _, _, _ = np.linalg.lstsq(A_sys, rhs_W[c], rcond=None)
                
                loss = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m+j
                            b = j*p+k
                            c = k*n+i
                            val = np.dot(U[a,:], V[b,:]*W[c,:])
                            loss += (val-1)**2
                rng2 = np.random.RandomState(it*7+attempt)
                for _ in range(1500):
                    a = rng2.randint(0, n*m)
                    b = rng2.randint(0, m*p)
                    c = rng2.randint(0, p*n)
                    i1,j1 = divmod(a,m)
                    j2,k1 = divmod(b,p)
                    k2,i2 = divmod(c,n)
                    if i1==i2 and j1==j2 and k1==k2:
                        continue
                    val = np.dot(U[a,:], V[b,:]*W[c,:])
                    loss += val**2
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if best_loss < 1e-12:
                    break
            
            if best_loss < 1e-8:
                U_r = np.round(best_U * 2) / 2
                V_r = np.round(best_V * 2) / 2
                W_r = np.round(best_W * 2) / 2
                if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return _schoolbook()
# EVOLVE-BLOCK-END
