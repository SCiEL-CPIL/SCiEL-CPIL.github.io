import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130

    nm = n * m
    mp = m * p
    pn = p * n

    nz = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nz.append((a, b, c))
    nz_set = set(nz)

    def verify(U, V, W, tol=1e-8):
        r = U.shape[1]
        for (a, b, c) in nz:
            if abs(np.dot(U[a], V[b] * W[c]) - 1.0) > tol:
                return False
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) not in nz_set:
                        if abs(np.dot(U[a], V[b] * W[c])) > tol:
                            return False
        return True

    def als_rank(r, max_iter=150, n_init=8, tol=1e-12):
        best_err = np.inf
        best_result = None
        for init in range(n_init):
            if time.time() > deadline:
                break
            np.random.seed(42 + init * 1000)
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            prev_err = np.inf
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                # Precompute V*W outer products
                # For U update: M[a] is implicit. For each a, solve:
                # sum over all (b,c): (V[b]*W[c]) * U[a] = rhs[a,b,c]
                # This is: M @ U[a,:] = rhs_a where M[b*pn+c,:] = V[b,:]*W[c,:]
                M = np.zeros((mp * pn, r))
                for b in range(mp):
                    for c in range(pn):
                        M[b * pn + c, :] = V[b, :] * W[c, :]
                
                for a in range(nm):
                    rhs = np.zeros(mp * pn)
                    for (aa, b, c) in nz:
                        if aa == a:
                            rhs[b * pn + c] = 1.0
                    U[a, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                
                Mv = np.zeros((nm * pn, r))
                for a in range(nm):
                    for c in range(pn):
                        Mv[a * pn + c, :] = U[a, :] * W[c, :]
                for b in range(mp):
                    rhs = np.zeros(nm * pn)
                    for (a, bb, c) in nz:
                        if bb == b:
                            rhs[a * pn + c] = 1.0
                    V[b, :] = np.linalg.lstsq(Mv, rhs, rcond=None)[0]
                
                Mw = np.zeros((nm * mp, r))
                for a in range(nm):
                    for b in range(mp):
                        Mw[a * mp + b, :] = U[a, :] * V[b, :]
                for c in range(pn):
                    rhs = np.zeros(nm * mp)
                    for (a, b, cc) in nz:
                        if cc == c:
                            rhs[a * mp + b] = 1.0
                    W[c, :] = np.linalg.lstsq(Mw, rhs, rcond=None)[0]
                
                # Compute error on nonzero entries
                err = 0.0
                for (a, b, c) in nz:
                    val = np.dot(U[a], V[b] * W[c])
                    err += (val - 1.0) ** 2
                
                if abs(prev_err - err) < tol:
                    break
                prev_err = err
            
            # Check convergence
            err_final = 0.0
            for (a, b, c) in nz:
                val = np.dot(U[a], V[b] * W[c])
                err_final += (val - 1.0) ** 2
            
            if err_final < 1e-6:
                if verify(U, V, W, tol=1e-8):
                    return U, V, W
                if err_final < best_err:
                    best_err = err_final
                    best_result = (U.copy(), V.copy(), W.copy())
        
        if best_result is not None and best_err < 1e-6:
            U, V, W = best_result
            if verify(U, V, W, tol=1e-8):
                return U, V, W
        return None

    for r in range(21, 64):
        if time.time() > deadline:
            break
        result = als_rank(r, max_iter=100, n_init=5)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

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
