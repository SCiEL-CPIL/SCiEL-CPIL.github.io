import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def _strassen_block(i1, i2, j1, j2, k1, k2):
    """Return (U, V, W) as lists of (index, value) tuples for a 2x2 Strassen block.
    Local 2x2x2: A rows {i1,i2}, B cols {j1,j2}, C cols {k1,k2}.
    """
    U = []
    V = []
    W = []
    def add(u, v, w):
        U.append(u); V.append(v); W.append(w)
    # M1
    add([(i1*m+j1,1),(i2*m+j2,1)], [(j1*p+k1,1),(j2*p+k2,1)], [(k1*n+i1,1),(k2*n+i2,1)])
    # M2
    add([(i2*m+j1,1),(i2*m+j2,1)], [(j1*p+k1,1)], [(k1*n+i2,1),(k2*n+i2,-1)])
    # M3
    add([(i1*m+j1,1)], [(j1*p+k2,1),(j2*p+k2,-1)], [(k2*n+i1,1),(k2*n+i2,1)])
    # M4
    add([(i2*m+j2,1)], [(j2*p+k1,1),(j1*p+k1,-1)], [(k1*n+i1,1),(k1*n+i2,1)])
    # M5
    add([(i1*m+j1,1),(i1*m+j2,1)], [(j2*p+k2,1)], [(k1*n+i1,-1),(k2*n+i1,1)])
    # M6
    add([(i2*m+j1,1),(i1*m+j1,-1)], [(j1*p+k1,1),(j1*p+k2,1)], [(k2*n+i2,1)])
    # M7
    add([(i1*m+j2,1),(i2*m+j2,-1)], [(j2*p+k1,1),(j2*p+k2,1)], [(k1*n+i1,1)])
    return U, V, W

def _direct_block(i_list, j_list, k_list):
    """Direct (rank = |i|*|j|*|k|) block: each (i,j,k) one rank-1 term."""
    U = []
    V = []
    W = []
    for i in i_list:
        for j in j_list:
            for k in k_list:
                U.append([(i*m+j, 1)])
                V.append([(j*p+k, 1)])
                W.append([(k*n+i, 1)])
    return U, V, W

def _build_rank28():
    """Verified rank-28 decomposition: 10 direct + 7 + 7 Strassen + 4 direct."""
    U = []
    V = []
    W = []
    # Part 1: j=0, k=0..4 (10 direct)
    u, v, w = _direct_block([0, 1], [0], [0, 1, 2, 3, 4])
    U += u; V += v; W += w
    # Part 2: j=1,2 k=0,1 (7 Strassen)
    u, v, w = _strassen_block(0, 1, 1, 2, 0, 1)
    U += u; V += v; W += w
    # Part 3: j=1,2 k=2,3 (7 Strassen)
    u, v, w = _strassen_block(0, 1, 1, 2, 2, 3)
    U += u; V += v; W += w
    # Part 4: j=1,2 k=4 (4 direct)
    u, v, w = _direct_block([0, 1], [1, 2], [4])
    U += u; V += v; W += w
    return U, V, W

def _build_rank26():
    """Rank-26 sub-decomposition (drop two direct terms from rank-28)."""
    U = []
    V = []
    W = []
    # Part 1: j=0, k=0..4 (10 direct)
    u, v, w = _direct_block([0, 1], [0], [0, 1, 2, 3, 4])
    U += u; V += v; W += w
    # Part 2: j=1,2 k=0,1 (7 Strassen)
    u, v, w = _strassen_block(0, 1, 1, 2, 0, 1)
    U += u; V += v; W += w
    # Part 3: j=1,2 k=2,3 (7 Strassen)
    u, v, w = _strassen_block(0, 1, 1, 2, 2, 3)
    U += u; V += v; W += w
    # Part 4: j=1,2 k=4 (4 direct)
    u, v, w = _direct_block([0, 1], [1, 2], [4])
    U += u; V += v; W += w
    # Drop the two direct terms for (i=0,j=1,k=4) and (i=1,j=1,k=4)
    # Those are the last two of Part 4 (i=0 first then i=1, both j=1,k=4)
    U = U[:-2]
    V = V[:-2]
    W = W[:-2]
    return U, V, W

def _to_arrays(U, V, W, rank):
    Ua = np.zeros((n*m, rank))
    Va = np.zeros((m*p, rank))
    Wa = np.zeros((p*n, rank))
    for r in range(rank):
        for (idx, val) in U[r]:
            Ua[idx, r] += val
        for (idx, val) in V[r]:
            Va[idx, r] += val
        for (idx, val) in W[r]:
            Wa[idx, r] += val
    return Ua, Va, Wa

def _residual(Ua, Va, Wa, rank):
    """residual[i,j,k] = sum_r Ua[i*m+j,r]*Va[j*p+k,r]*Wa[k*n+i,r]"""
    res = np.zeros((n, m, p))
    for r in range(rank):
        # outer product of Ua[:,r] (n*m), Va[:,r] (m*p), Wa[:,r] (p*n)
        Uv = Ua[:, r].reshape(n, m)
        Vv = Va[:, r].reshape(m, p)
        Wv = Wa[:, r].reshape(p, n)
        # term[i,j,k] = Uv[i,j]*Vv[j,k]*Wv[k,i]
        # build via einsum
        res += np.einsum('ij,jk,ki->ijk', Uv, Vv, Wv)
    return res

def _verify(U, V, W, rank, tol=1e-9):
    Ua, Va, Wa = _to_arrays(U, V, W, rank)
    res = _residual(Ua, Va, Wa, rank)
    target = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i, j, k] = 1.0
    return np.allclose(res, target, atol=tol, rtol=0)

def _round_half(A):
    return np.round(A * 2) / 2

def _try_rank27(deadline):
    """Try to find a rank-27 decomposition by adding one free rank-1 term
    to a rank-26 sub-decomposition. Returns (U,V,W) lists or None."""
    if time.time() > deadline:
        return None
    U26, V26, W26 = _build_rank26()
    Ua26, Va26, Wa26 = _to_arrays(U26, V26, W26, 26)
    res26 = _residual(Ua26, Va26, Wa26, 26)
    target = np.ones((n, m, p))
    # residual to be filled by the 27th term: r[i,j,k] = target - res26
    rem = target - res26
    # 27th term: Ua27[i*m+j] * Va27[j*p+k] * Wa27[k*n+i] = rem[i,j,k]
    # Unknowns: U27 (6), V27 (15), W27 (10) = 31, equations = 30.
    # Use a structured parametrization to make it solvable:
    # Fix a pattern for the 27th term and solve for its scale, OR
    # do a local search over a small parameterization.
    #
    # Strategy: parametrize the 27th term as a "diagonal" correction.
    # rem has a specific structure (the two dropped terms plus Strassen cross terms).
    # Let's just try a numerical local search minimizing ||residual||^2 over
    # a low-dimensional family of 27th terms.

    # Family: 27th term = (a_i,j) * (b_j,k) * (c_k,i)
    # We'll do a coordinate/greedy search. Start from the two dropped direct terms
    # which are (i=0,j=1,k=4) and (i=1,j=1,k=4). A single rank-1 term cannot
    # exactly equal their sum (two different (i,k) pairs with same j). So we need
    # the 27th term to also absorb the Strassen cross-terms.
    #
    # Simpler: do a small local search over the free term's entries.
    best = None
    best_err = np.inf
    rng = np.random.default_rng(0)
    # Initialize 27th term near the dropped direct terms
    U27 = np.zeros(n*m)
    V27 = np.zeros(m*p)
    W27 = np.zeros(p*n)
    # seed: j=1, k=4 row
    V27[1*p+4] = 1.0
    U27[0*m+1] = 1.0
    U27[1*m+1] = 1.0
    W27[4*n+0] = 1.0
    W27[4*n+1] = 1.0

    def full_residual(U27, V27, W27):
        Ua = np.concatenate([Ua26, U27.reshape(n*m,1)])
        Va = np.concatenate([Va26, V27.reshape(m*p,1)])
        Wa = np.concatenate([Wa26, W27.reshape(p*n,1)])
        return _residual(Ua, Va, Wa, 27)

    def err(U27, V27, W27):
        r = full_residual(U27, V27, W27)
        return np.sum((r - target)**2)

    cur_err = err(U27, V27, W27)
    # coordinate descent with random restarts
    for restart in range(200):
        if time.time() > deadline:
            break
        if restart > 0:
            U27 = np.zeros(n*m)
            V27 = np.zeros(m*p)
            W27 = np.zeros(p*n)
            # random sparse seed
            j0 = rng.integers(0, m)
            k0 = rng.integers(0, p)
            V27[j0*p+k0] = 1.0
            U27[rng.integers(0, n*m)] = 1.0
            W27[rng.integers(0, p*n)] = 1.0
            cur_err = err(U27, V27, W27)
        # coordinate descent
        for step in range(200):
            if time.time() > deadline:
                break
            improved = False
            for idx in range(n*m):
                if time.time() > deadline:
                    break
                old = U27[idx]
                # try a few candidate values
                for cand in [0.0, 0.5, 1.0, -0.5, -1.0, old + 0.5, old - 0.5]:
                    U27[idx] = cand
                    e = err(U27, V27, W27)
                    if e < cur_err - 1e-12:
                        cur_err = e
                        improved = True
                    else:
                        U27[idx] = old
                        break
            for idx in range(m*p):
                if time.time() > deadline:
                    break
                old = V27[idx]
                for cand in [0.0, 0.5, 1.0, -0.5, -1.0, old + 0.5, old - 0.5]:
                    V27[idx] = cand
                    e = err(U27, V27, W27)
                    if e < cur_err - 1e-12:
                        cur_err = e
                        improved = True
                    else:
                        V27[idx] = old
                        break
            for idx in range(p*n):
                if time.time() > deadline:
                    break
                old = W27[idx]
                for cand in [0.0, 0.5, 1.0, -0.5, -1.0, old + 0.5, old - 0.5]:
                    W27[idx] = cand
                    e = err(U27, V27, W27)
                    if e < cur_err - 1e-12:
                        cur_err = e
                        improved = True
                    else:
                        W27[idx] = old
                        break
            if not improved:
                break
        # check exact
        if cur_err < 1e-12:
            U27r = _round_half(U27)
            V27r = _round_half(V27)
            W27r = _round_half(W27)
            # verify exact with rounded
            Ua = np.concatenate([Ua26, U27r.reshape(n*m,1)])
            Va = np.concatenate([Va26, V27r.reshape(m*p,1)])
            Wa = np.concatenate([Wa26, W27r.reshape(p*n,1)])
            r = _residual(Ua, Va, Wa, 27)
            if np.allclose(r, target, atol=1e-9, rtol=0):
                # build lists
                U = list(U26)
                V = list(V26)
                W = list(W26)
                U.append([(i, float(U27r[i])) for i in range(n*m) if U27r[i] != 0])
                V.append([(i, float(V27r[i])) for i in range(m*p) if V27r[i] != 0])
                W.append([(i, float(W27r[i])) for i in range(p*n) if W27r[i] != 0])
                return U, V, W
    return None

def construct():
    global n, m, p
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 145  # safety margin before 150s kill

    # Guaranteed baseline: rank-28
    U28, V28, W28 = _build_rank28()
    if _verify(U28, V28, W28, 28):
        baseline = (U28, V28, W28)
    else:
        baseline = None

    # Try rank-27
    if time.time() < deadline:
        r27 = _try_rank27(deadline)
        if r27 is not None:
            U, V, W = r27
            if _verify(U, V, W, 27):
                return {"U": U, "V": V, "W": W}

    # Fallback to verified rank-28 or schoolbook
    if baseline is not None:
        return {"U": baseline[0], "V": baseline[1], "W": baseline[2]}
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
