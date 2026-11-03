"""Tensor decomposition for 4x4 x 4x6 matrix multiplication (rank 77)."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """Return the 7 Strassen terms for 2x2x2 as (U_dict, V_dict, W_dict).
    U: (i,j) for A 2x2; V: (j,k) for B 2x2; W: (k,i) for C 2x2 (C = A@B)."""
    terms = []
    # m1 = (a+d)(e+g) -> C00 += m1, C11 += m1
    terms.append((
        {(0,0):1,(1,1):1},
        {(0,0):1,(1,1):1},
        {(0,0):1,(1,1):1},
    ))
    # m2 = (c+d)e -> C10 += m2, C11 -= m2
    terms.append((
        {(1,0):1,(1,1):1},
        {(0,0):1},
        {(1,0):1,(1,1):-1},
    ))
    # m3 = a(f-g) -> C01 += m3, C00 -= m3
    terms.append((
        {(0,0):1},
        {(0,1):1,(1,1):-1},
        {(0,1):1,(0,0):-1},
    ))
    # m4 = d(g-f) -> C10 += m4, C11 -= m4
    terms.append((
        {(1,1):1},
        {(1,0):1,(0,0):-1},
        {(1,0):1,(1,1):-1},
    ))
    # m5 = (a+b)h -> C00 += m5, C10 -= m5
    terms.append((
        {(0,0):1,(0,1):1},
        {(1,1):1},
        {(0,0):1,(1,0):-1},
    ))
    # m6 = (c-a)(e+f) -> C01 += m6, C11 += m6
    terms.append((
        {(1,0):1,(0,0):-1},
        {(0,0):1,(0,1):1},
        {(0,1):1,(1,1):1},
    ))
    # m7 = (b-d)(g+h) -> C00 += m7, C01 -= m7
    terms.append((
        {(0,1):1,(1,1):-1},
        {(1,0):1,(1,1):1},
        {(0,0):1,(0,1):-1},
    ))
    return terms

def _verify(U, V, W, n, m, p):
    rng = np.random.RandomState(12345)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    C = np.zeros((n, p))
    for r in range(U.shape[1]):
        a_r = np.sum(A * U[:, r].reshape(n, m))
        b_r = np.sum(B * V[:, r].reshape(m, p))
        C += a_r * b_r * W[:, r].reshape(p, n).T
    return np.allclose(C, C_expected, atol=1e-9)

def _build(n, m, p):
    """Build rank-77 decomposition: 49 (4x4x4) + 28 (4x4x2)."""
    S = _strassen_2x2()
    terms = []

    # Part 1: 4x4x4 via Strassen ⊗ Strassen (49 terms)
    for s in range(7):
        us, vs, ws = S[s]
        for t in range(7):
            ut, vt, wt = S[t]
            U = {}; V = {}; W = {}
            for (i1,j1), cu in us.items():
                for (i2,j2), cu2 in ut.items():
                    gi, gj = 2*i1+i2, 2*j1+j2
                    U[(gi,gj)] = U.get((gi,gj),0) + cu*cu2
            for (j1,k1), cv in vs.items():
                for (j2,k2), cv2 in vt.items():
                    gj, gk = 2*j1+j2, 2*k1+k2
                    V[(gj,gk)] = V.get((gj,gk),0) + cv*cv2
            for (i1,k1), cw in ws.items():
                for (i2,k2), cw2 in wt.items():
                    gi, gk = 2*i1+i2, 2*k1+k2
                    W[(gk,gi)] = W.get((gk,gi),0) + cw*cw2
            terms.append((U,V,W))

    # Part 2: 4x4x2 via 4 blocks of Strassen 2x2x2 (28 terms)
    for i1 in range(2):
        for j1 in range(2):
            for s in range(7):
                us, vs, ws = S[s]
                U = {}; V = {}; W = {}
                for (i2,j2), cu in us.items():
                    gi, gj = 2*i1+i2, 2*j1+j2
                    U[(gi,gj)] = U.get((gi,gj),0) + cu
                for (j2,k2), cv in vs.items():
                    gj, gk = 2*j1+j2, 4+k2
                    V[(gj,gk)] = V.get((gj,gk),0) + cv
                for (i2,k2), cw in ws.items():
                    gi, gk = 2*i1+i2, 4+k2
                    W[(gk,gi)] = W.get((gk,gi),0) + cw
                terms.append((U,V,W))

    rank = len(terms)
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    for r,(ud,vd,wd) in enumerate(terms):
        for (i,j),c in ud.items(): U[i*m+j, r] = c
        for (j,k),c in vd.items(): V[j*p+k, r] = c
        for (k,i),c in wd.items(): W[k*n+i, r] = c
    return U, V, W

def _schoolbook(n, m, p):
    rank = n*m*p
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1; V[j*p+k, c] = 1; W[k*n+i, c] = 1; c += 1
    return U, V, W

def _remove_redundant(U, V, W, n, m, p, deadline):
    """Try to drop linearly-redundant terms; bounded by deadline."""
    rank = U.shape[1]
    keep = [True]*rank
    # drop zero columns
    for r in range(rank):
        if np.allclose(U[:,r],0) or np.allclose(V[:,r],0) or np.allclose(W[:,r],0):
            keep[r] = False
    # drop scalar-multiple duplicates
    for r in range(rank):
        if not keep[r]: continue
        for s in range(r+1, rank):
            if not keep[s]: continue
            if np.linalg.norm(V[:,r])>1e-12:
                a = np.dot(V[:,s],V[:,r])/np.dot(V[:,r],V[:,r])
                if (np.allclose(V[:,s],a*V[:,r],atol=1e-10) and
                    np.allclose(W[:,s],a*W[:,r],atol=1e-10) and
                    np.allclose(U[:,s],a*U[:,r],atol=1e-10)):
                    keep[s] = False
    if all(keep):
        return U, V, W
    idx = [r for r in range(rank) if keep[r]]
    return U[:,idx], V[:,idx], W[:,idx]

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    U, V, W = _build(n, m, p)
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    if time.time() < t0 + 140.0:
        U2, V2, W2 = _remove_redundant(U, V, W, n, m, p, t0+140.0)
        if _verify(U2, V2, W2, n, m, p):
            U, V, W = U2, V2, W2
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
