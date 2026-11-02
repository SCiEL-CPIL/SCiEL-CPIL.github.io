"""Rewritten program: search for a low-rank exact tensor decomposition of 2x3x3 matmul."""
import time
import random
import itertools
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize
import itertools

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for l in range(p):
                T[i, j, l] = 1.0
    return T

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    T = _build_tensor()
    R = len(U[0]) if isinstance(U, list) and U else U.shape[1]
    for i in range(n):
        for j in range(m):
            for l in range(p):
                s = 0.0
                for c in range(R):
                    s += U[i*m+j][c] * V[j*p+l][c] * W[l*n+i][c]
                if abs(s - T[i, j, l]) > tol:
                    return False
    return True

def _flatten(U, V, W):
    return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

def _unflatten(x, R):
    n, m, p = DIMENSIONS
    U = x[:n*m*R].reshape(n*m, R)
    V = x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
    W = x[n*m*R + m*p*R:].reshape(p*n, R)
    return U, V, W

def _objective(x, R):
    n, m, p = DIMENSIONS
    U, V, W = _unflatten(x, R)
    err = 0.0
    for i in range(n):
        for j in range(m):
            for l in range(p):
                s = 0.0
                for c in range(R):
                    s += U[i*m+j][c] * V[j*p+l][c] * W[l*n+i][c]
                err += (s - 1.0)**2
    return err

def _try_direct_rank(R, max_iters=2000, seed=0):
    """Try to find a rank-R decomposition via random init + coordinate descent + L-BFGS polish."""
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    best_err = float('inf')
    best = None
    
    for it in range(max_iters):
        # Random initialization with structure
        if it % 3 == 0:
            U = np.array([[rng.gauss(0,1) for _ in range(R)] for _ in range(n*m)], dtype=float)
            V = np.array([[rng.gauss(0,1) for _ in range(R)] for _ in range(m*p)], dtype=float)
            W = np.array([[rng.gauss(0,1) for _ in range(R)] for _ in range(p*n)], dtype=float)
        elif it % 3 == 1:
            U = np.zeros((n*m, R), dtype=float)
            V = np.zeros((m*p, R), dtype=float)
            W = np.zeros((p*n, R), dtype=float)
            for c in range(min(R, n*m)):
                U[c, c] = 1.0
            for c in range(min(R, m*p)):
                V[c, c] = 1.0
            for c in range(min(R, p*n)):
                W[c, c] = 1.0
            U += rng.gauss(0, 0.1) * np.ones_like(U)
            V += rng.gauss(0, 0.1) * np.ones_like(V)
            W += rng.gauss(0, 0.1) * np.ones_like(W)
        else:
            U = np.zeros((n*m, R), dtype=float)
            V = np.zeros((m*p, R), dtype=float)
            W = np.zeros((p*n, R), dtype=float)
            for c in range(R):
                idx_u = rng.randint(0, n*m-1)
                idx_v = rng.randint(0, m*p-1)
                idx_w = rng.randint(0, p*n-1)
                U[idx_u, c] = rng.choice([-1, 1])
                V[idx_v, c] = rng.choice([-1, 1])
                W[idx_w, c] = rng.choice([-1, 1])
        
        # Coordinate descent
        cd_steps = 80 if R <= 14 else 60
        for _ in range(cd_steps):
            for i in range(n):
                for j in range(m):
                    idx = i*m+j
                    M = np.zeros((p, R))
                    b = np.zeros(p)
                    for l in range(p):
                        for c in range(R):
                            M[l, c] = V[j*p+l][c] * W[l*n+i][c]
                        b[l] = 1.0
                    try:
                        sol = np.linalg.lstsq(M, b, rcond=None)[0]
                        U[idx, :] = sol
                    except:
                        pass
            for j in range(m):
                for l in range(p):
                    idx = j*p+l
                    M = np.zeros((n, R))
                    b = np.zeros(n)
                    for i in range(n):
                        for c in range(R):
                            M[i, c] = U[i*m+j][c] * W[l*n+i][c]
                        b[i] = 1.0
                    try:
                        sol = np.linalg.lstsq(M, b, rcond=None)[0]
                        V[idx, :] = sol
                    except:
                        pass
            for l in range(p):
                for i in range(n):
                    idx = l*n+i
                    M = np.zeros((m, R))
                    b = np.zeros(m)
                    for j in range(m):
                        for c in range(R):
                            M[j, c] = U[i*m+j][c] * V[j*p+l][c]
                        b[j] = 1.0
                    try:
                        sol = np.linalg.lstsq(M, b, rcond=None)[0]
                        W[idx, :] = sol
                    except:
                        pass
        
        # Evaluate error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for l in range(p):
                    s = 0.0
                    for c in range(R):
                        s += U[i*m+j][c]*V[j*p+l][c]*W[l*n+i][c]
                    err += (s - 1.0)**2
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        if best_err < 1e-10:
            break
    
    if best is None:
        return None
    if best_err < 1e-6:
        U, V, W = best
        
        # L-BFGS polish
        x0 = _flatten(U, V, W)
        try:
            res = minimize(_objective, x0, args=(R,), method='L-BFGS-B', 
                          options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-15})
            if res.fun < best_err:
                U, V, W = _unflatten(res.x, R)
                best_err = res.fun
        except:
            pass
        
        # High-precision polish: run additional coordinate descent
        for _ in range(50):
            for i in range(n):
                for j in range(m):
                    idx = i*m+j
                    M = np.zeros((p, R))
                    b = np.zeros(p)
                    for l in range(p):
                        for c in range(R):
                            M[l, c] = V[j*p+l][c] * W[l*n+i][c]
                        b[l] = 1.0
                    try:
                        sol = np.linalg.lstsq(M, b, rcond=None)[0]
                        U[idx, :] = sol
                    except:
                        pass
            for j in range(m):
                for l in range(p):
                    idx = j*p+l
                    M = np.zeros((n, R))
                    b = np.zeros(n)
                    for i in range(n):
                        for c in range(R):
                            M[i, c] = U[i*m+j][c] * W[l*n+i][c]
                        b[i] = 1.0
                    try:
                        sol = np.linalg.lstsq(M, b, rcond=None)[0]
                        V[idx, :] = sol
                    except:
                        pass
            for l in range(p):
                for i in range(n):
                    idx = l*n+i
                    M = np.zeros((m, R))
                    b = np.zeros(m)
                    for j in range(m):
                        for c in range(R):
                            M[j, c] = U[i*m+j][c] * V[j*p+l][c]
                        b[j] = 1.0
                    try:
                        sol = np.linalg.lstsq(M, b, rcond=None)[0]
                        W[idx, :] = sol
                    except:
                        pass
        
        # Convert to rationals with wider search
        U_r = [[_to_frac(x) for x in row] for row in U]
        V_r = [[_to_frac(x) for x in row] for row in V]
        W_r = [[_to_frac(x) for x in row] for row in W]
        if _verify_exact(U_r, V_r, W_r):
            return (U_r, V_r, W_r)
        
        U_r2 = [[_to_frac2(x) for x in row] for row in U]
        V_r2 = [[_to_frac2(x) for x in row] for row in V]
        W_r2 = [[_to_frac2(x) for x in row] for row in W]
        if _verify_exact(U_r2, V_r2, W_r2):
            return (U_r2, V_r2, W_r2)
    return None

def _to_frac(x):
    r = round(x * 2) / 2.0
    if abs(x - r) < 1e-6:
        return r
    for den in [1, 2, 3, 4, 5, 6]:
        f = round(x * den) / den
        if abs(x - f) < 1e-6:
            return f
    return x

def _to_frac2(x):
    for den in [1, 2, 3, 4, 5, 6, 8, 12]:
        f = round(x * den) / den
        if abs(x - f) < 1e-5:
            return f
    f = round(x * 10) / 10.0
    if abs(x - f) < 1e-4:
        return f
    return x

def _verify_exact(U, V, W):
    n, m, p = DIMENSIONS
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for l in range(p):
                s = Fraction(0, 1)
                for c in range(R):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+l][c]) * Fraction(W[l*n+i][c])
                if s != 1:
                    return False
    return True

def _try_gf2_search(R, max_seeds=100):
    """Search for rank-R decomposition over GF(2) using structured random init."""
    n, m, p = DIMENSIONS
    best_err = float('inf')
    best = None
    
    for seed in range(max_seeds):
        rng = random.Random(seed)
        U = np.zeros((n*m, R), dtype=int)
        V = np.zeros((m*p, R), dtype=int)
        W = np.zeros((p*n, R), dtype=int)
        
        for c in range(R):
            # Random sparse structure over GF(2)
            density = 0.3
            for i in range(n*m):
                if rng.random() < density:
                    U[i, c] = rng.randint(0, 1)
            for j in range(m*p):
                if rng.random() < density:
                    V[j, c] = rng.randint(0, 1)
            for l in range(p*n):
                if rng.random() < density:
                    W[l, c] = rng.randint(0, 1)
        
        # Evaluate error over GF(2)
        err = 0
        for i in range(n):
            for j in range(m):
                for l in range(p):
                    s = 0
                    for c in range(R):
                        s ^= U[i*m+j][c] & V[j*p+l][c] & W[l*n+i][c]
                    if s != 1:
                        err += 1
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        if best_err == 0:
            break
    
    if best is None or best_err > 0:
        return None
    
    # Convert GF(2) solution to integers (0, 1)
    U, V, W = best
    U_r = [[int(x) for x in row] for row in U]
    V_r = [[int(x) for x in row] for row in V]
    W_r = [[int(x) for x in row] for row in W]
    
    # Verify over integers (should work since GF(2) solution implies integer solution with 0,1)
    if _verify_exact(U_r, V_r, W_r):
        return (U_r, V_r, W_r)
    return None

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

def construct():
    deadline = time.time() + 115
    # Try ranks from 12 up to 18
    for R in range(12, 19):
        if time.time() > deadline - 15:
            break
        # More seeds for lower ranks, fewer for higher
        max_seeds = 30 if R <= 14 else 15
        for seed in range(max_seeds):
            if time.time() > deadline - 10:
                break
            max_iters = 200 if R <= 14 else 120
            res = _try_direct_rank(R, max_iters=max_iters, seed=seed)
            if res is not None:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
        
        # Try GF(2) search for this rank
        if time.time() < deadline - 20:
            res = _try_gf2_search(R, max_seeds=50)
            if res is not None:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
