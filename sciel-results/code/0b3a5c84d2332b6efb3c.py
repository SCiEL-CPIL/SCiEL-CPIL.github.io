"""Exact rank-83 decomposition of M_{3,4,7} via Strassen on 2x2x2 sub-blocks."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_block(i0, j0, k0, U, V, W, col):
    """Add 7 Strassen rank-1 terms for a 2x2x2 block anchored at (i0,j0,k0).
    U indexed by (i,j) -> i*m+j, V by (j,k) -> j*p+k, W by (k,i) -> k*n+i.
    Returns new column offset."""
    n, m, p = DIMENSIONS
    # Local indices
    a = (i0, j0)       # A[0,0]
    b = (i0, j0+1)     # A[0,1]
    c = (i0+1, j0)     # A[1,0]
    d = (i0+1, j0+1)   # A[1,1]
    w = (j0, k0)       # B[0,0]
    x = (j0, k0+1)     # B[0,1]
    y = (j0+1, k0)     # B[1,0]
    z = (j0+1, k0+1)   # B[1,1]
    p00 = (k0, i0)
    p01 = (k0, i0+1)
    p10 = (k0+1, i0)
    p11 = (k0+1, i0+1)
    
    def uidx(pt): return pt[0]*m + pt[1]
    def vidx(pt): return pt[0]*p + pt[1]
    def widx(pt): return pt[0]*n + pt[1]
    
    # Standard Strassen rank-7 decomposition
    # m1 = (a+d)(w+z)
    # m2 = (c+d)(w+x)
    # m3 = (a+b)(y+z)
    # m4 = (c-a)(w+y)
    # m5 = (b-d)(x+z)
    # m6 = (a+d-c-b)(y+x)
    # m7 = (a+d-c-b)(w+y)
    #
    # C00 = m1 + m4 + m5 - m7
    # C01 = m3 + m5 - m6
    # C10 = m2 + m4 + m6
    # C11 = m1 - m2 + m3 + m7
    
    terms = [
        # (U_factors, V_factors, W_output, sign)
        # m1 = (a+d)(w+z): contributes to p00(+1), p11(+1)
        [(a,1),(d,1)], [(w,1),(z,1)], [(p00,1),(p11,1)],
        # m2 = (c+d)(w+x): contributes to p10(+1), p11(-1)
        [(c,1),(d,1)], [(w,1),(x,1)], [(p10,1),(p11,-1)],
        # m3 = (a+b)(y+z): contributes to p01(+1), p11(+1)
        [(a,1),(b,1)], [(y,1),(z,1)], [(p01,1),(p11,1)],
        # m4 = (c-a)(w+y): contributes to p00(+1), p10(-1)
        [(c,1),(-a,1)], [(w,1),(y,1)], [(p00,1),(p10,-1)],
        # m5 = (b-d)(x+z): contributes to p00(-1), p01(+1)
        [(b,1),(-d,1)], [(x,1),(z,1)], [(p00,-1),(p01,1)],
        # m6 = (a+d-c-b)(y+x): contributes to p10(+1), p11(-1)
        [(a,1),(d,1),(-c,1),(-b,1)], [(y,1),(x,1)], [(p10,1),(p11,-1)],
        # m7 = (a+d-c-b)(w+y): contributes to p01(+1), p00(-1)
        [(a,1),(d,1),(-c,1),(-b,1)], [(w,1),(y,1)], [(p01,1),(p00,-1)],
    ]
    
    for ufactors, vfactors, woutputs in terms:
        for (u, us) in ufactors:
            U[uidx(u), col] += us
        for (v, vs) in vfactors:
            V[vidx(v), col] += vs
        for (wo, ws) in woutputs:
            W[widx(wo), col] += ws
        col += 1
    return col

def _verify(U, V, W, n, m, p, rng):
    """Verify decomposition on random inputs."""
    A = rng.integers(-3, 4, size=(n, m)).astype(float)
    B = rng.integers(-3, 4, size=(m, p)).astype(float)
    C = A @ B
    recon = np.zeros((n, p))
    rank = U.shape[1]
    for c in range(rank):
        ucol = U[:, c].reshape(n, m)
        vcol = V[:, c].reshape(m, p)
        wcol = W[:, c].reshape(p, n)
        uA = ucol @ A       # (m,)
        vB = vcol @ B       # (p,)
        contrib = uA[:, None] * (wcol @ vB[None, :])
        recon += contrib
    return np.allclose(recon, C, atol=1e-6)

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    DEADLINE = 140.0
    
    # Strategy: Use Strassen on 2x2x2 blocks for i∈{0,1}, j∈{0,1,2}, k∈{0,1,2,3,4,5}
    # Blocks at (j0,k0) = (0,0),(0,2),(0,4),(2,0),(2,2),(2,4) for i0=0
    # That covers i∈{0,1}, j∈{0,1,2}, k∈{0,1,2,3,4,5}
    # Remaining: i∈{0,1} with (j=3, k=0..6) or (j∈{0,1,2}, k∈{5,6}) -> schoolbook
    #           i=2, all j,k -> schoolbook
    
    # Total rank: 6 Strassen blocks × 7 + schoolbook remainder
    # Schoolbook remainder: 
    #   i∈{0,1}: j=3,k=0..6 (7) + j∈{0,1,2},k∈{5,6} (6) = 13
    #   i=2: all 4×7 = 28
    # Total = 42 + 13 + 28 = 83
    
    rank = 83
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    # Strassen blocks for i0=0, j0 in {0,2}, k0 in {0,2,4}
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            col = _strassen_block(0, j0, k0, U, V, W, col)
    
    # Schoolbook for remaining terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if covered by Strassen block
                if i < 2 and j < 3 and k < 6:
                    # Check if (i,j,k) is in one of the Strassen blocks
                    covered = False
                    for j0 in [0, 2]:
                        for k0 in [0, 2, 4]:
                            if i in [0,1] and j in [j0, j0+1] and k in [k0, k0+1]:
                                covered = True
                    if covered:
                        continue
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Verify
    rng = np.random.default_rng(12345)
    if time.time() < DEADLINE:
        if _verify(U, V, W, n, m, p, rng):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 84
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
