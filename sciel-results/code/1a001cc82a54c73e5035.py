"""Search for a low-rank exact tensor decomposition of 2x3x3 matmul."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p), dtype=float)
    return T

def _verify_exact(U, V, W):
    n, m, p = DIMENSIONS
    R = len(U[0]) if U else 0
    if R == 0:
        return False
    for i in range(n):
        for j in range(m):
            for l in range(p):
                s = Fraction(0, 1)
                for c in range(R):
                    u_val = U[i*m+j][c]
                    v_val = V[j*p+l][c]
                    w_val = W[l*n+i][c]
                    if isinstance(u_val, (int, float)):
                        u_f = Fraction(u_val).limit_denominator(1000000)
                    else:
                        u_f = u_val
                    if isinstance(v_val, (int, float)):
                        v_f = Fraction(v_val).limit_denominator(1000000)
                    else:
                        v_f = v_val
                    if isinstance(w_val, (int, float)):
                        w_f = Fraction(w_val).limit_denominator(1000000)
                    else:
                        w_f = w_val
                    s += u_f * v_f * w_f
                if s != 1:
                    return False
    return True

def _compute_error(U, V, W):
    n, m, p = DIMENSIONS
    R = U.shape[1]
    err = 0.0
    for i in range(n):
        for j in range(m):
            for l in range(p):
                s = 0.0
                for c in range(R):
                    s += U[i*m+j, c] * V[j*p+l, c] * W[l*n+i, c]
                err += (s - 1.0)**2
    return err

def _coordinate_descent(U, V, W, R, n, m, p, steps=100):
    """Run coordinate descent to refine the decomposition."""
    for _ in range(steps):
        # Update U
        for i in range(n):
            for j in range(m):
                idx = i*m + j
                M = np.zeros((p, R))
                b = np.zeros(p)
                for l in range(p):
                    for c in range(R):
                        M[l, c] = V[j*p+l, c] * W[l*n+i, c]
                    b[l] = 1.0
                try:
                    sol = np.linalg.lstsq(M, b, rcond=None)[0]
                    U[idx, :] = sol
                except:
                    pass
        # Update V (FIXED: l ranges over p)
        for j in range(m):
            for l in range(p):
                idx = j*p + l
                M = np.zeros((n, R))
                b = np.zeros(n)
                for i in range(n):
                    for c in range(R):
                        M[i, c] = U[i*m+j, c] * W[l*n+i, c]
                    b[i] = 1.0
                try:
                    sol = np.linalg.lstsq(M, b, rcond=None)[0]
                    V[idx, :] = sol
                except:
                    pass
        # Update W
        for l in range(p):
            for i in range(n):
                idx = l*n + i
                M = np.zeros((m, R))
                b = np.zeros(m)
                for j in range(m):
                    for c in range(R):
                        M[j, c] = U[i*m+j, c] * V[j*p+l, c]
                    b[j] = 1.0
                try:
                    sol = np.linalg.lstsq(M, b, rcond=None)[0]
                    W[idx, :] = sol
                except:
                    pass
    return U, V, W

def _try_structured_init(R, seed, n, m, p):
    """Try different structured initializations."""
    rng = random.Random(seed)
    inits = []
    
    # Init 1: Pure random
    U = np.array([[rng.gauss(0,1) for _ in range(R)] for _ in range(n*m)])
    V = np.array([[rng.gauss(0,1) for _ in range(R)] for _ in range(m*p)])
    W = np.array([[rng.gauss(0,1) for _ in range(R)] for _ in range(p*n)])
    inits.append((U, V, W))
    
    # Init 2: Identity-like with perturbation
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    for c in range(min(R, n*m)):
        U[c, c] = 1.0
    for c in range(min(R, m*p)):
        V[c, c] = 1.0
    for c in range(min(R, p*n)):
        W[c, c] = 1.0
    U += rng.gauss(0, 0.5)
    V += rng.gauss(0, 0.5)
    W += rng.gauss(0, 0.5)
    inits.append((U, V, W))
    
    # Init 3: Sparse random
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    for c in range(R):
        idx_u = rng.randint(0, n*m-1)
        idx_v = rng.randint(0, m*p-1)
        idx_w = rng.randint(0, p*n-1)
        U[idx_u, c] = rng.choice([-1, 1])
        V[idx_v, c] = rng.choice([-1, 1])
        W[idx_w, c] = rng.choice([-1, 1])
    inits.append((U, V, W))
    
    # Init 4: Block structure - exploit that n=2
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    # First half of columns for i=0, second half for i=1
    half = R // 2
    for c in range(half):
        U[c, c] = 1.0  # i=0, j=c
        V[c, c] = 1.0  # j=c, l=0
        W[c, c] = 1.0  # l=0, i=0
    for c in range(half, R):
        c2 = c - half
        U[m + c2, c] = 1.0  # i=1, j=c2
        V[c2, c] = 1.0  # j=c2, l=0
        W[c2 + p, c] = 1.0  # l=0, i=1
    U += rng.gauss(0, 0.3)
    V += rng.gauss(0, 0.3)
    W += rng.gauss(0, 0.3)
    inits.append((U, V, W))
    
    return inits

def _snap_to_rationals(U, V, W, max_den=24):
    """Snap values to simple fractions."""
    def snap(x):
        # Try integer first
        r = round(x)
        if abs(x - r) < 1e-8:
            return int(r)
        # Try half-integers
        r = round(x * 2) / 2.0
        if abs(x - r) < 1e-8:
            return r
        # Try other small denominators
        for den in range(2, max_den + 1):
            num = round(x * den)
            f = num / den
            if abs(x - f) < 1e-8:
                return f
        # Fall back to nearest 0.01
        f = round(x * 100) / 100.0
        if abs(x - f) < 1e-6:
            return f
        return x
    
    U_r = [[snap(x) for x in row] for row in U]
    V_r = [[snap(x) for x in row] for row in V]
    W_r = [[snap(x) for x in row] for row in W]
    return U_r, V_r, W_r

def _try_direct_rank(R, max_iters=50, seed=0):
    """Try to find a rank-R decomposition."""
    n, m, p = DIMENSIONS
    best_err = float('inf')
    best = None
    
    for it in range(max_iters):
        current_seed = seed + it
        inits = _try_structured_init(R, current_seed, n, m, p)
        
        for U_init, V_init, W_init in inits:
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
            
            # Run coordinate descent
            U, V, W = _coordinate_descent(U, V, W, R, n, m, p, steps=80)
            
            # Compute error
            err = _compute_error(U, V, W)
            
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            
            if best_err < 1e-10:
                break
        if best_err < 1e-10:
            break
    
    if best is None:
        return None
    
    if best_err < 1e-6:
        U, V, W = best
        # High-precision polish
        U, V, W = _coordinate_descent(U, V, W, R, n, m, p, steps=50)
        err = _compute_error(U, V, W)
        
        if err < 1e-8:
            # Try to snap to rationals
            U_r, V_r, W_r = _snap_to_rationals(U, V, W)
            if _verify_exact(U_r, V_r, W_r):
                return (U_r, V_r, W_r)
            
            # Try with wider denominator
            U_r2, V_r2, W_r2 = _snap_to_rationals(U, V, W, max_den=48)
            if _verify_exact(U_r2, V_r2, W_r2):
                return (U_r2, V_r2, W_r2)
    
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
    deadline = time.time() + 145
    n, m, p = DIMENSIONS
    
    # Try ranks from 12 to 17 (skip 18 which is schoolbook)
    for R in range(12, 18):
        if time.time() > deadline - 20:
            break
        
        # More attempts for lower ranks
        max_seeds = 40 if R <= 14 else 25
        for seed in range(max_seeds):
            if time.time() > deadline - 15:
                break
            res = _try_direct_rank(R, max_iters=30, seed=seed)
            if res is not None:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
