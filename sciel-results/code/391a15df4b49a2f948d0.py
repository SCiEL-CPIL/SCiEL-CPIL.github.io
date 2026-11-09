import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 3, 4]
T0 = time.time()
DEADLINE = T0 + 110

def _build_T():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a*m + b
                j = b*p + c
                k = c*n + a
                T[i, j, k] = 1.0
    return T

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _als(T, R, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    n, m, p = T.shape
    U = rng.normal(size=(n, R))
    V = rng.normal(size=(m, R))
    W = rng.normal(size=(p, R))
    
    for it in range(iters):
        # Update U
        # For each r, U[:, r] = (T * (V_r odot W_r)) / ||V_r odot W_r||^2
        # T is (n, m, p). V_r is (m,). W_r is (p,).
        # V_r odot W_r is (m, p).
        # T * (V_r odot W_r) sums over j, k.
        # Result is (n,).
        
        for r in range(R):
            v_r = V[:, r]
            w_r = W[:, r]
            prod = np.outer(v_r, w_r) # (m, p)
            numer = np.tensordot(T, prod, axes=([1, 2], [0, 1])) # (n,)
            denom = np.sum(prod**2)
            if denom > 1e-12:
                U[:, r] = numer / denom
            else:
                U[:, r] = 0

        # Update V
        for r in range(R):
            u_r = U[:, r]
            w_r = W[:, r]
            prod = np.outer(u_r, w_r) # (n, p)
            numer = np.tensordot(T, prod, axes=([0, 2], [0, 1])) # (m,)
            denom = np.sum(prod**2)
            if denom > 1e-12:
                V[:, r] = numer / denom
            else:
                V[:, r] = 0

        # Update W
        for r in range(R):
            u_r = U[:, r]
            v_r = V[:, r]
            prod = np.outer(u_r, v_r) # (n, m)
            numer = np.tensordot(T, prod, axes=([0, 1], [0, 1])) # (p,)
            denom = np.sum(prod**2)
            if denom > 1e-12:
                W[:, r] = numer / denom
            else:
                W[:, r] = 0

    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    res = np.linalg.norm(T - recon)
    return U, V, W, res

def _exact_verify(U, V, W, T_int):
    n, m, p = T_int.shape
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(R):
                    s += Fraction(U[i][r]) * Fraction(V[j][r]) * Fraction(W[k][r])
                if s != T_int[i, j, k]:
                    return False
    return True

def _to_frac_list(arr, denom):
    out = []
    for row in arr:
        r = []
        for x in row:
            # Round to nearest multiple of 1/denom
            r.append(round(x * denom) / denom)
        out.append(r)
    return out

def construct():
    T = _build_T()
    T_int = T.astype(int)
    
    # Search from lower bound to schoolbook
    # Lower bound for 3x3x3 is 19? 
    # Let's try 19 to 35.
    for R in range(19, 36):
        if time.time() > DEADLINE:
            break
        found = None
        for seed in range(0, 20): # Fewer seeds per rank to save time
            if time.time() > DEADLINE:
                break
            U, V, W, res = _als(T, R, iters=50, seed=seed)
            if res < 1e-6:
                # Try to exact-ify
                for denom in [1, 2, 4, 8]:
                    Uf = _to_frac_list(U, denom)
                    Vf = _to_frac_list(V, denom)
                    Wf = _to_frac_list(W, denom)
                    if _exact_verify(Uf, Vf, Wf, T_int):
                        found = (Uf, Vf, Wf)
                        break
                if found:
                    break
        if found:
            Uf, Vf, Wf = found
            return {"U": Uf, "V": Vf, "W": Wf}
            
    return _schoolbook()
