"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

# ---------------------------------------------------------------------------
# Small verified rank-1 tensor blocks (U: i*m+j, V: j*p+k, W: k*n+i)
# ---------------------------------------------------------------------------
def _strassen_2x2x2():
    """Rank-7 exact decomposition of 2x2 x 2x2. Returns list of (U,V,W) sparse dicts."""
    terms = []
    # M1 = (a11+a22)(b11+b22)
    terms.append(({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}))
    # M2 = (a21+a22)b11
    terms.append(({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}))
    # M3 = a11(b12-b22)
    terms.append(({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}))
    # M4 = a22(b21-b11)
    terms.append(({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}))
    # M5 = (a11+a12)b22
    terms.append(({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}))
    # M6 = (a21-a11)(b11+b12)
    terms.append(({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}))
    # M7 = (a12-a22)(b21+b22)
    terms.append(({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}))
    return terms

def _schoolbook(n, m, p):
    """Rank n*m*p exact decomposition."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append(({(i,j):1}, {(j,k):1}, {(k,i):1}))
    return terms

def _strassen_2x2x3_candidate():
    """
    Candidate rank-5 decomposition for 2x2 x 2x3.
    (i,j) in U, (j,k) in V, (k,i) in W with i in 0..1, j in 0..1, k in 0..2.
    This is a hypothesis; it is verified at runtime and rejected if invalid.
    """
    terms = []
    # M1 = (a11+a22)(b11+b21)
    terms.append(({(0,0):1,(1,1):1}, {(0,0):1,(1,0):1}, {(0,0):1,(0,1):1}))
    # M2 = (a21+a22)b11
    terms.append(({(1,0):1,(1,1):1}, {(0,0):1}, {(0,1):-1,(1,1):1}))
    # M3 = a11(b12-b22)
    terms.append(({(0,0):1}, {(0,1):1,(1,1):-1}, {(1,0):1,(1,1):-1}))
    # M4 = a22(b21-b11)
    terms.append(({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):-1,(0,1):1}))
    # M5 = (a11+a12)(b13+b23)
    terms.append(({(0,0):1,(0,1):1}, {(0,2):1,(1,2):1}, {(2,0):1,(2,1):-1}))
    return terms

# ---------------------------------------------------------------------------
# Generic Kronecker (tensor-product) builder
# ---------------------------------------------------------------------------
def _kron_terms(termsA, termsB, nA, mA, pA, nB, mB, pB):
    """
    Tensor product of two matrix-multiplication decompositions.
    Block A: nA x mA x pA ; Block B: nB x mB x pB.
    Resulting global dims: n=nA*nB, m=mA*mB, p=pA*pB.
    Global index: i = iA*nB + iB ; j = jA*mB + jB ; k = kA*pB + kB.
    """
    out = []
    for uA, vA, wA in termsA:
        for uB, vB, wB in termsB:
            U = {}
            for (iA, jA), cA in uA.items():
                for (iB, jB), cB in uB.items():
                    key = (iA * nB + iB, jA * mB + jB)
                    U[key] = U.get(key, 0) + cA * cB
            V = {}
            for (jA, kA), cA in vA.items():
                for (jB, kB), cB in vB.items():
                    key = (jA * mB + jB, kA * pB + kB)
                    V[key] = V.get(key, 0) + cA * cB
            W = {}
            for (kA, iA), cA in wA.items():
                for (kB, iB), cB in wB.items():
                    key = (kA * pB + kB, iA * nB + iB)
                    W[key] = W.get(key, 0) + cA * cB
            out.append((U, V, W))
    return out

def _terms_to_matrices(terms, n, m, p):
    rank = len(terms)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    for r, (u, v, w) in enumerate(terms):
        for (i, j), c in u.items():
            U[i * m + j, r] = c
        for (j, k), c in v.items():
            V[j * p + k, r] = c
        for (k, i), c in w.items():
            W[k * n + i, r] = c
    return U, V, W

def _verify(U, V, W, n, m, p, seed=42, atol=1e-9):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        alpha = np.sum(A * U[:, r].reshape(n, m))
        beta = np.sum(B * V[:, r].reshape(m, p))
        C_computed += alpha * beta * W[:, r].reshape(p, n).T
    return bool(np.allclose(C_computed, C_expected, atol=atol))

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 6
    deadline = start_time + 110.0

    strassen = _strassen_2x2x2()

    # Guaranteed-correct base: (2x2x2) x (2x2x3) with schoolbook on the 3-block.
    # n=2*2=4, m=2*2=4, p=2*3=6 -> rank 7 * 12 = 84? No: schoolbook(2,2,3)=12.
    # Use the proven 81-rank path: (2x2x2)x(2x2x2) rank 49 + schoolbook 4x4x2 rank 32.
    def _build_81():
        terms = _kron_terms(strassen, strassen, 2, 2, 2, 2, 2, 2)  # 4x4x4, rank 49
        # schoolbook for the extra 2 columns: 4x4x2, rank 32
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    terms.append(({(i, j): 1}, {(j, 4 + k): 1}, {(4 + k, i): 1}))
        return _terms_to_matrices(terms, 4, 4, 6)

    # Candidate improvement: (2x2x2) x (2x2x3) with a rank-5 2x2x3 block -> rank 35.
    def _build_35():
        terms = _kron_terms(strassen, _strassen_2x2x3_candidate(),
                            2, 2, 2, 2, 2, 3)  # 4x4x6, rank 35
        return _terms_to_matrices(terms, 4, 4, 6)

    best = None
    # Try the low-rank candidate first; only accept if it verifies exactly.
    try:
        U35, V35, W35 = _build_35()
        if _verify(U35, V35, W35, n, m, p):
            best = (U35, V35, W35)
    except Exception:
        best = None

    if time.time() > deadline:
        # out of time; fall through to guaranteed base
        best = None

    if best is None:
        U, V, W = _build_81()
        if _verify(U, V, W, n, m, p):
            best = (U, V, W)
        else:
            # Ultimate fallback: pure schoolbook 4x4x6 (rank 96), always valid.
            terms = _schoolbook(4, 4, 6)
            U, V, W = _terms_to_matrices(terms, 4, 4, 6)
            best = (U, V, W)

    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
