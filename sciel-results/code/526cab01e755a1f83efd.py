"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank, deadline=None):
    """Check that the decomposition exactly reproduces C = A*B using vectorized NumPy."""
    if deadline is not None and time.time() > deadline:
        return False
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # Reconstruct full tensor: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # T = einsum('ar,br,cr->abc', Ua, Va, Wa)
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa, optimize=True)
    
    # Build expected tensor
    Texp = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                Texp[a, b, c] = 1.0
    
    # Check exact match
    if not np.allclose(T, Texp, atol=1e-10, rtol=0):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _rank55(n, m, p):
    """Rank-55 decomposition for 2x3 x 3x10 using Strassen per k-pair.
    Per pair: 7 Strassen terms for the 2x2 sub-block + 4 direct B2 terms = 11.
    """
    rank = 55
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    col = 0
    for t in range(5):
        k0 = 2*t
        k1 = 2*t + 1

        # Strassen for [[A00,A01],[A10,A11]] x [[B0k0,B0k1],[B1k0,B1k1]]
        # a=A00, b=A01, c=A10, d=A11, e=B0k0, f=B0k1, g=B1k0, h=B1k1
        # M1=(a+d)(e+h)
        U[0][col] += 1; U[4][col] += 1
        V[k0][col] += 1; V[10+k0][col] += 1
        W[2*k0][col] += 1; W[2*k1][col] += 1
        col += 1
        # M2=(c+d)e
        U[3][col] += 1; U[4][col] += 1
        V[k0][col] += 1
        W[2*k0+1][col] += 1
        col += 1
        # M3=a(f-h)
        U[0][col] += 1
        V[k1][col] += 1; V[10+k1][col] -= 1
        W[2*k0][col] += 1; W[2*k1][col] += 1
        col += 1
        # M4=d(g-e)
        U[4][col] += 1
        V[10+k0][col] += 1; V[k0][col] -= 1
        W[2*k0][col] += 1; W[2*k1][col] += 1
        col += 1
        # M5=(a+b)h
        U[0][col] += 1; U[1][col] += 1
        V[10+k1][col] += 1
        W[2*k0+1][col] += 1; W[2*k1+1][col] += 1
        col += 1
        # M6=(c-a)(e+f)
        U[3][col] += 1; U[0][col] -= 1
        V[k0][col] += 1; V[k1][col] += 1
        W[2*k0+1][col] += 1
        col += 1
        # M7=(b-d)(g+h)
        U[1][col] += 1; U[4][col] -= 1
        V[10+k0][col] += 1; V[10+k1][col] += 1
        W[2*k0][col] += 1
        col += 1

        # Direct B2 terms
        U[2][col] += 1; V[20+k0][col] += 1; W[2*k0][col] += 1
        col += 1
        U[2][col] += 1; V[20+k1][col] += 1; W[2*k1][col] += 1
        col += 1
        U[5][col] += 1; V[20+k0][col] += 1; W[2*k0+1][col] += 1
        col += 1
        U[5][col] += 1; V[20+k1][col] += 1; W[2*k1+1][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # Try rank-55 decomposition with verification
    try:
        res55 = _rank55(n, m, p)
        if time.time() < deadline - 5:
            if _verify(res55["U"], res55["V"], res55["W"], n, m, p, 55, deadline=deadline):
                return res55
    except Exception:
        pass

    # Fallback: schoolbook rank-60
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
