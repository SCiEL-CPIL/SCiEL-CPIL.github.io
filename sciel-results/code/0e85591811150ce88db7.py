import time
import math
import numpy as np

DIMENSIONS = [6, 6, 6]

START_TIME = time.time()
TIME_LIMIT = 120.0  # safety margin before 150s kill


def _check_time():
    return (time.time() - START_TIME) > (TIME_LIMIT - 10.0)


def _extended_gcd(a, b):
    """Returns (g, x, y) such that a*x + b*y = g = gcd(a, b)"""
    if a == 0:
        return b, 0, 1
    else:
        g, x, y = _extended_gcd(b % a, a)
        return g, y - (b // a) * x, x


def _find_integer_solution(g):
    """
    Find an integer vector w such that w . g = 1.
    Assumes gcd(g) = 1.
    """
    # g is a 1D numpy array of integers
    g = np.array(g, dtype=int)
    if np.any(g == 0) and np.all(g == 0):
        raise ValueError("Zero vector")
    
    # Find index of non-zero element
    non_zero_indices = np.nonzero(g)[0]
    if len(non_zero_indices) == 0:
        raise ValueError("Zero vector")
        
    # If only one non-zero, it must be 1 or -1
    if len(non_zero_indices) == 1:
        idx = non_zero_indices[0]
        if g[idx] == 1:
            w = np.zeros_like(g)
            w[idx] = 1
            return w
        elif g[idx] == -1:
            w = np.zeros_like(g)
            w[idx] = -1
            return w
        else:
            raise ValueError("Cannot solve for single element not +/- 1")

    # Iteratively combine
    # Start with first two non-zero elements
    idx1 = non_zero_indices[0]
    idx2 = non_zero_indices[1]
    
    # w1, w2 such that g[idx1]*w1 + g[idx2]*w2 = gcd(g[idx1], g[idx2])
    g1 = g[idx1]
    g2 = g[idx2]
    d, x1, x2 = _extended_gcd(g1, g2)
    
    # Let's maintain a vector of coefficients for the original g.
    # coeff[i] is the coefficient of g[i] in the current linear combination equal to current_gcd.
    coeff = np.zeros_like(g)
    coeff[idx1] = x1
    coeff[idx2] = x2
    current_gcd = d
    
    # Iterate through remaining non-zero elements
    for idx in non_zero_indices[2:]:
        gi = g[idx]
        # We have current_gcd = sum coeff[i] * g[i]
        # We want to combine with gi.
        # Find y1, y2 such that current_gcd * y1 + gi * y2 = new_gcd
        new_gcd, y1, y2 = _extended_gcd(current_gcd, gi)
        
        # Update coeff
        coeff = coeff * y1
        coeff[idx] = y2
        current_gcd = new_gcd
        
    # Now we have current_gcd = sum coeff[i] * g[i]
    # We want 1 = sum w[i] * g[i]
    # So w[i] = coeff[i] / current_gcd
    # Since current_gcd must be 1 (or -1) for a solution to exist with integer w
    if current_gcd == 1:
        return coeff
    elif current_gcd == -1:
        return -coeff
    else:
        # This should not happen if gcd(g)=1
        raise ValueError("GCD is not 1, no integer solution exists")


def _kron_decompose(Ua, Va, Wa, Ub, Vb, Wb, n1, m1, p1, n2, m2, p2):
    """Kronecker product of two tensor decompositions (vectorized)."""
    r1 = Ua.shape[1]
    r2 = Ub.shape[1]

    n, m, p = n1 * n2, m1 * m2, p1 * p2
    r = r1 * r2

    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))

    for r1i in range(r1):
        for r2i in range(r2):
            r_idx = r1i * r2 + r2i
            # U: reshape Ua[:, r1i] to (n1, m1), Ub[:, r2i] to (n2, m2)
            Ua_sub = Ua[:, r1i].reshape(n1, m1)
            Ub_sub = Ub[:, r2i].reshape(n2, m2)
            U_kron = np.kron(Ua_sub, Ub_sub)
            U[:, r_idx] = U_kron.reshape(-1)

            # V: reshape Va[:, r1i] to (m1, p1), Vb[:, r2i] to (m2, p2)
            Va_sub = Va[:, r1i].reshape(m1, p1)
            Vb_sub = Vb[:, r2i].reshape(m2, p2)
            V_kron = np.kron(Va_sub, Vb_sub)
            V[:, r_idx] = V_kron.reshape(-1)

            # W: reshape Wa[:, r1i] to (p1, n1), Wb[:, r2i] to (p2, n2)
            Wa_sub = Wa[:, r1i].reshape(p1, n1)
            Wb_sub = Wb[:, r2i].reshape(p2, n2)
            W_kron = np.kron(Wa_sub, Wb_sub)
            W[:, r_idx] = W_kron.reshape(-1)

    return U, V, W


def _strassen_2x2():
    n, m, p = 2, 2, 2
    U = np.zeros((n * m, 7))
    V = np.zeros((m * p, 7))
    W = np.zeros((p * n, 7))

    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1

    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    W[2, 2] = 1; W[2, 4] = 1
    W[1, 1] = 1; W[1, 3] = 1
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    return U, V, W


def _laderman_3x3():
    n, m, p = 3, 3, 3
    r = 23
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))

    # P0 = (a+d)(w+z)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # P1 = (a+b)(w+x)
    U[0, 1] = 1; U[1, 1] = 1
    V[0, 1] = 1; V[1, 1] = 1
    # P2 = (b-d)(x-z)
    U[1, 2] = 1; U[3, 2] = -1
    V[1, 2] = 1; V[3, 2] = -1
    # P3 = (c+f)(y+v)
    U[2, 3] = 1; U[5, 3] = 1
    V[2, 3] = 1; V[5, 3] = 1
    # P4 = (f-i)(v-q)
    U[5, 4] = 1; U[8, 4] = -1
    V[5, 4] = 1; V[7, 4] = -1
    # P5 = (i+g)(r+p)
    U[8, 5] = 1; U[6, 5] = 1
    V[8, 5] = 1; V[6, 5] = 1
    # P6 = (d+e)(z+u)
    U[3, 6] = 1; U[4, 6] = 1
    V[3, 6] = 1; V[4, 6] = 1
    # P7 = (e-g)(u-p)
    U[4, 7] = 1; U[6, 7] = -1
    V[4, 7] = 1; V[6, 7] = -1
    # P8 = (b+e)(x+u)
    U[1, 8] = 1; U[4, 8] = 1
    V[1, 8] = 1; V[4, 8] = 1
    # P9 = (a+e)(w+u)
    U[0, 9] = 1; U[4, 9] = 1
    V[0, 9] = 1; V[4, 9] = 1
    # P10 = (c+i)(y+r)
    U[2, 10] = 1; U[8, 10] = 1
    V[2, 10] = 1; V[8, 10] = 1
    # P11 = (d+g)(z+p)
    U[3, 11] = 1; U[6, 11] = 1
    V[3, 11] = 1; V[6, 11] = 1
    # P12 = (a+c)(w+y)
    U[0, 12] = 1; U[2, 12] = 1
    V[0, 12] = 1; V[2, 12] = 1
    # P13 = (d+f)(z+v)
    U[3, 13] = 1; U[5, 13] = 1
    V[3, 13] = 1; V[5, 13] = 1
    # P14 = (g+i)(p+r)
    U[6, 14] = 1; U[8, 14] = 1
    V[6, 14] = 1; V[8, 14] = 1
    # P15 = (b+f+h)(x+v+q)
    U[1, 15] = 1; U[5, 15] = 1; U[7, 15] = 1
    V[1, 15] = 1; V[5, 15] = 1; V[7, 15] = 1
    # P16 = (a+d+e)(w+z+u)
    U[0, 16] = 1; U[3, 16] = 1; U[4, 16] = 1
    V[0, 16] = 1; V[3, 16] = 1; V[4, 16] = 1
    # P17 = (b+e+h)(x+u+q)
    U[1, 17] = 1; U[4, 17] = 1; U[7, 17] = 1
    V[1, 17] = 1; V[4, 17] = 1; V[7, 17] = 1
    # P18 = (c+f+i)(y+v+r)
    U[2, 18] = 1; U[5, 18] = 1; U[8, 18] = 1
    V[2, 18] = 1; V[5, 18] = 1; V[8, 18] = 1
    # P19 = (a+d+g)(w+z+p)
    U[0, 19] = 1; U[3, 19] = 1; U[6, 19] = 1
    V[0, 19] = 1; V[3, 19] = 1; V[6, 19] = 1
    # P20 = (a+b+c)(w+x+y)
    U[0, 20] = 1; U[1, 20] = 1; U[2, 20] = 1
    V[0, 20] = 1; V[1, 20] = 1; V[2, 20] = 1
    # P21 = (d+e+f)(z+u+v)
    U[3, 21] = 1; U[4, 21] = 1; U[5, 21] = 1
    V[3, 21] = 1; V[4, 21] = 1; V[5, 21] = 1
    # P22 = (g+h+i)(p+q+r)
    U[6, 22] = 1; U[7, 22] = 1; U[8, 22] = 1
    V[6, 22] = 1; V[7, 22] = 1; V[8, 22] = 1

    # Solve for W using exact integer arithmetic
    # For each (i,k), we need W[k*n+i, :] such that sum_r W[k*n+i,r] * G[r,i,k] = 1
    # where G[r,i,k] = sum_j U[i*m+j,r] * V[j*p+k,r]
    for i in range(n):
        for k in range(p):
            g_vec = np.zeros(r, dtype=int)
            for rr in range(r):
                val = 0
                for j in range(m):
                    val += int(U[i * m + j, rr]) * int(V[j * p + k, rr])
                g_vec[rr] = val
            
            # Try to find integer solution
            try:
                w_row = _find_integer_solution(g_vec)
                W[k * n + i, :] = w_row
            except Exception:
                # If GCD is not 1, try using fractions
                # We need sum_r W[r] * g_vec[r] = 1
                # Use extended GCD approach with rational numbers
                from fractions import Fraction
                g_frac = [Fraction(int(x)) for x in g_vec]
                # Find x such that sum x[r]*g[r] = 1
                # This is equivalent to finding integer solution to the same problem
                # but allowing half-integers
                # Try: find integer solution to 2*sum x[r]*g[r] = 2
                # i.e., sum x[r]*(2*g[r]) = 2
                # This always has a solution if gcd(2*g) divides 2
                doubled = [int(x) * 2 for x in g_vec]
                try:
                    w_row2 = _find_integer_solution(doubled)
                    # w_row2 . doubled = 1, so w_row2 . g_vec = 1/2
                    # We need w . g_vec = 1, so w = 2 * w_row2
                    W[k * n + i, :] = 2 * w_row2
                except Exception:
                    # Last resort: use least squares with rounding
                    denom = float(np.dot(g_vec, g_vec))
                    if denom > 1e-9:
                        W[k * n + i, :] = np.rint(g_vec / denom)
                    else:
                        W[k * n + i, :] = 0
    return U, V, W


def _verify(U, V, W, n, m, p, seed=0):
    """Vectorized verification using matrix operations."""
    rng = np.random.default_rng(seed)
    A = rng.integers(-3, 4, size=(n, m)).astype(float)
    B = rng.integers(-3, 4, size=(m, p)).astype(float)
    C_true = A @ B
    U = np.asarray(U, dtype=float); V = np.asarray(V, dtype=float); W = np.asarray(W, dtype=float)
    r = U.shape[1]
    
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    # For each rank component rr: G[i,k,rr] = sum_j U[i,j,rr] * V[j,k,rr]
    # This is a batched contraction
    G = np.einsum('ijr,jkr->ikr', U_3d, V_3d)
    
    # C_rec[i,k] = sum_rr W[k,i,rr] * G[i,k,rr]
    C_rec = np.einsum('kir,ikr->ik', W_3d, G)
    
    return np.allclose(C_rec, C_true, atol=1e-9)


def construct():
    n, m, p = DIMENSIONS
    if _check_time():
        # Fallback: schoolbook 6x6
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try the improved Kronecker composition: Strassen(2x2, r=7) x Laderman(3x3, r=23)
    try:
        U2, V2, W2 = _strassen_2x2()
        U3, V3, W3 = _laderman_3x3()
        U, V, W = _kron_decompose(U2, V2, W2, U3, V3, W3,
                                  2, 2, 2, 3, 3, 3)
        # Verify exactness on multiple random instances for robustness
        ok = True
        for seed in [1, 7, 42, 123, 999, 555]:
            if not _verify(U, V, W, n, m, p, seed=seed):
                ok = False
                break
        if ok:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass

    # Fallback: Strassen(2x2) x schoolbook(3x3), rank 189 (known-good)
    U2, V2, W2 = _strassen_2x2()
    n3, m3, p3 = 3, 3, 3
    r3 = n3 * m3 * p3
    U3 = np.zeros((n3 * m3, r3))
    V3 = np.zeros((m3 * p3, r3))
    W3 = np.zeros((p3 * n3, r3))
    col = 0
    for i in range(n3):
        for j in range(m3):
            for k in range(p3):
                U3[i * m3 + j, col] = 1
                V3[j * p3 + k, col] = 1
                W3[k * n3 + i, col] = 1
                col += 1
    U, V, W = _kron_decompose(U2, V2, W2, U3, V3, W3, 2, 2, 2, 3, 3, 3)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
