"""Tensor decomposition for 2x2 x 2x9 matrix multiplication."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIMEOUT = 130  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START_TIME


def _build_tensor(n, m, p):
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _validate(U, V, W, n, m, p, atol=1e-9):
    """Check if U,V,W exactly decompose the multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = _build_tensor(n, m, p)
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T, T_recon, atol=atol)


def _strassen_block(k0, m, p, n):
    """
    Generate U, V, W terms for a 2x2x2 Strassen block covering columns k0, k0+1.
    Returns lists of (u_vec, v_vec, w_vec).
    u_vec: length n*m (4)
    v_vec: length m*p (18)
    w_vec: length p*n (18)
    """
    # Strassen terms for 2x2x2:
    # P0=(A00+A11)(B00+B11)
    # P1=(A10+A11)B00
    # P2=A00(B01-B11)
    # P3=A11(B10-B00)
    # P4=(A00+A01)B11
    # P5=(A10-A00)(B00+B01)
    # P6=(A01-A11)(B10+B11)
    # C00=P0-P1-P4+P6, C01=P4+P6, C10=P1+P5, C11=P0-P2+P3-P5
    
    # Local indices for U (A): 0=A00, 1=A01, 2=A10, 3=A11
    # Local indices for V (B): 0=B0,k0, 1=B0,k0+1, 2=B1,k0, 3=B1,k0+1
    # Local indices for W (C): 0=Ck0,0, 1=Ck0,1, 2=Ck0+1,0, 3=Ck0+1,1
    
    terms = [
        # (U coeffs, V coeffs, W coeffs)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, -1), (1, 1), (2, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, -1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, -1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (3, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(0, -1), (2, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (1, 1)]),
    ]
    
    out_u = []
    out_v = []
    out_w = []
    
    for uc, vc, wc in terms:
        u = np.zeros(n * m)
        for idx, c in uc:
            u[idx] += c
            
        v = np.zeros(m * p)
        # Map local V indices to global
        # Local 0 -> Global k0
        # Local 1 -> Global k0+1
        # Local 2 -> Global 9+k0
        # Local 3 -> Global 9+k0+1
        v_map = {
            0: k0,
            1: k0 + 1,
            2: m * p // 2 + k0, # 9 + k0
            3: m * p // 2 + k0 + 1 # 9 + k0 + 1
        }
        for idx, c in vc:
            v[v_map[idx]] += c
            
        w = np.zeros(p * n)
        # Map local W indices to global
        # Local 0 -> Global 2*k0
        # Local 1 -> Global 2*k0+1
        # Local 2 -> Global 2*k0+2
        # Local 3 -> Global 2*k0+3
        w_map = {
            0: 2 * k0,
            1: 2 * k0 + 1,
            2: 2 * k0 + 2,
            3: 2 * k0 + 3
        }
        for idx, c in wc:
            w[w_map[idx]] += c
            
        out_u.append(u)
        out_v.append(v)
        out_w.append(w)
        
    return out_u, out_v, out_w


def _schoolbook_block(k, m, p, n):
    """
    Generate U, V, W terms for a single column k using schoolbook (rank 4).
    """
    out_u = []
    out_v = []
    out_w = []
    
    # Term 1: A00 * B0k -> C0k
    u = np.zeros(n * m); u[0] = 1
    v = np.zeros(m * p); v[k] = 1
    w = np.zeros(p * n); w[2 * k] = 1
    out_u.append(u); out_v.append(v); out_w.append(w)
    
    # Term 2: A01 * B1k -> C0k
    u = np.zeros(n * m); u[1] = 1
    v = np.zeros(m * p); v[m * p // 2 + k] = 1 # 9+k
    w = np.zeros(p * n); w[2 * k] = 1
    out_u.append(u); out_v.append(v); out_w.append(w)
    
    # Term 3: A10 * B0k -> C1k
    u = np.zeros(n * m); u[2] = 1
    v = np.zeros(m * p); v[k] = 1
    w = np.zeros(p * n); w[2 * k + 1] = 1
    out_u.append(u); out_v.append(v); out_w.append(w)
    
    # Term 4: A11 * B1k -> C1k
    u = np.zeros(n * m); u[3] = 1
    v = np.zeros(m * p); v[m * p // 2 + k] = 1
    w = np.zeros(p * n); w[2 * k + 1] = 1
    out_u.append(u); out_v.append(v); out_w.append(w)
    
    return out_u, out_v, out_w


def construct():
    n, m, p = DIMENSIONS

    U_terms = []
    V_terms = []
    W_terms = []

    # Use Strassen for pairs of columns: (0,1), (2,3), (4,5), (6,7)
    for k0 in [0, 2, 4, 6]:
        u, v, w = _strassen_block(k0, m, p, n)
        U_terms.extend(u)
        V_terms.extend(v)
        W_terms.extend(w)

    # Use Schoolbook for the last column: 8
    u, v, w = _schoolbook_block(8, m, p, n)
    U_terms.extend(u)
    V_terms.extend(v)
    W_terms.extend(w)

    if not U_terms:
        # Fallback if something went wrong
        return _fallback()

    U = np.array(U_terms).T
    V = np.array(V_terms).T
    W = np.array(W_terms).T

    if _validate(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # If validation fails, return schoolbook (rank 36) to be safe
        return _fallback()


def _fallback():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
