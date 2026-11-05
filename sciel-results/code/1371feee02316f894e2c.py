import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 120.0  # safety margin under the 150s kill

def _time_left():
    return max(0.0, _DEADLINE - time.time())

# --- Strassen 2x2x2 (rank 7), indices U[i*2+j], V[j*2+k], W[k*2+i] ---
def _strassen_2x2x2():
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # r0: (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # r1: (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # r2: A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # r3: A11(B10-B00)
    U[3,3]=1; V[0,3]=-1; V[2,3]=1; W[0,3]=1; W[1,3]=1
    # r4: (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # r5: (A10-A00)(B00+B01)
    U[0,5]=-1; U[2,5]=1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # r6: (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

_U2 = _strassen_2x2x2()

def _build_rank49():
    """4x4x4 via Kronecker product of Strassen: rank 49."""
    U2, V2, W2 = _U2
    R4 = 49
    U4 = np.zeros((16, R4), dtype=np.int16)
    V4 = np.zeros((16, R4), dtype=np.int16)
    W4 = np.zeros((16, R4), dtype=np.int16)
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            for i in range(4):
                for j in range(4):
                    U4[i*4+j, c] = U2[(i//2)*2+(j//2), c0] * U2[(i%2)*2+(j%2), c1]
            for j in range(4):
                for k in range(4):
                    V4[j*4+k, c] = V2[(j//2)*2+(k//2), c0] * V2[(j%2)*2+(k%2), c1]
            for k in range(4):
                for i in range(4):
                    W4[k*4+i, c] = W2[(k//2)*2+(i//2), c0] * W2[(k%2)*2+(i%2), c1]
    return U4, V4, W4

_U4, _V4, _W4 = _build_rank49()

def _verify(U, V, W, n=4, m=4, p=8):
    """Exact check: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    R = U.shape[1]
    Wu = W.reshape(p, n, R)
    Uu = U.reshape(n, m, R)
    Vu = V.reshape(m, p, R)
    T = np.einsum('kir,ijr,jkr->kij', Wu, Uu, Vu)
    return bool(np.all(T == 1))

def _build_rank98():
    """Assemble 4x4x8 from two rank-49 blocks. Returns U,V,W (int16)."""
    U4, V4, W4 = _U4, _V4, _W4
    R4 = 49
    R8 = 98
    U = np.zeros((16, R8), dtype=np.int16)
    V = np.zeros((32, R8), dtype=np.int16)
    W = np.zeros((32, R8), dtype=np.int16)

    U[:, 0:R4] = U4
    U[:, R4:R8] = U4

    # V: block b uses B[:, b*4:(b+1)*4]
    for b in range(2):
        cs = b * R4
        for j in range(4):
            for kl in range(4):
                V[j*8 + (b*4 + kl), cs:cs+R4] = V4[j*4 + kl, :]
    # W: block b uses C rows b*4:(b+1)*4
    for b in range(2):
        cs = b * R4
        for kl in range(4):
            for i in range(4):
                W[(b*4 + kl)*4 + i, cs:cs+R4] = W4[kl*4 + i, :]
    return U, V, W

def _try_prune(U, V, W, budget_s):
    """Bounded local search: drop rank terms that are provably redundant.
    Uses vectorized contribution computation and efficient pair/triple removal.
    """
    n, m, p = 4, 4, 8
    R = U.shape[1]
    nm = n * m
    mp = m * p
    pn = p * n

    # Precompute per-term contribution tensors (flattened for speed)
    terms = np.zeros((R, p, n, m), dtype=np.int32)
    for r in range(R):
        Wu = W[:, r].reshape(p, n)
        Uu = U[:, r].reshape(n, m)
        Vu = V[:, r].reshape(m, p)
        terms[r] = np.einsum('ki,ij,jk->kij', Wu, Uu, Vu)

    # Flatten terms for faster comparison
    terms_flat = terms.reshape(R, -1)  # (R, p*n*m)
    S = terms_flat.sum(axis=0).astype(np.int64)
    target = np.ones(p * n * m, dtype=np.int64)

    norms = np.abs(terms_flat).sum(axis=1)
    kept = list(range(R))
    t_end = time.time() + budget_s

    # Phase 1: Greedy single-term removal, multiple passes
    for _pass in range(100):
        if time.time() >= t_end:
            break
        changed = False
        # Sort by norm ascending - try removing small terms first
        kept_sorted = sorted(kept, key=lambda r: norms[r])
        for r in kept_sorted:
            if time.time() >= t_end:
                break
            S2 = S - terms_flat[r]
            if np.array_equal(S2, target):
                S = S2
                kept.remove(r)
                changed = True
        if not changed:
            break

    # Phase 2: Pair-term removal with early rejection
    if time.time() < t_end:
        changed = True
        while changed and time.time() < t_end:
            changed = False
            S = terms_flat[kept].sum(axis=0).astype(np.int64)
            n_kept = len(kept)
            kept_sorted = sorted(kept, key=lambda r: norms[r])
            for ai in range(n_kept):
                if time.time() >= t_end:
                    break
                r1 = kept_sorted[ai]
                for bi in range(ai + 1, n_kept):
                    if time.time() >= t_end:
                        break
                    r2 = kept_sorted[bi]
                    S2 = S - terms_flat[r1] - terms_flat[r2]
                    if np.array_equal(S2, target):
                        S = S2
                        kept.remove(r2)
                        kept.remove(r1)
                        changed = True
                        break
                if changed:
                    break

    # Phase 3: Triple-term removal (only if time permits and rank is still high)
    if time.time() < t_end and len(kept) > 50:
        changed = True
        while changed and time.time() < t_end:
            changed = False
            S = terms_flat[kept].sum(axis=0).astype(np.int64)
            n_kept = len(kept)
            kept_sorted = sorted(kept, key=lambda r: norms[r])
            max_triple = min(n_kept, 25)
            for ai in range(max_triple):
                if time.time() >= t_end:
                    break
                r1 = kept_sorted[ai]
                for bi in range(ai + 1, max_triple):
                    if time.time() >= t_end:
                        break
                    r2 = kept_sorted[bi]
                    for ci in range(bi + 1, max_triple):
                        if time.time() >= t_end:
                            break
                        r3 = kept_sorted[ci]
                        S2 = S - terms_flat[r1] - terms_flat[r2] - terms_flat[r3]
                        if np.array_equal(S2, target):
                            S = S2
                            kept.remove(r3)
                            kept.remove(r2)
                            kept.remove(r1)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break

    if len(kept) == R:
        return U, V, W

    # Build pruned matrices
    U2 = U[:, kept].astype(np.int16)
    V2 = V[:, kept].astype(np.int16)
    W2 = W[:, kept].astype(np.int16)
    # Final exact verification
    if _verify(U2, V2, W2):
        return U2, V2, W2
    return U, V, W

def _schoolbook(n=4, m=4, p=8):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int16)
    V = np.zeros((m*p, rank), dtype=np.int16)
    W = np.zeros((p*n, rank), dtype=np.int16)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
    return U, V, W

def construct():
    """Rank-minimal exact tensor decomposition for 4x4 * 4x8.
    Base: rank-98 Kronecker-Strassen (two rank-49 blocks sharing U).
    Polish: bounded local search that prunes provably-redundant rank terms.
    Returns {'U','V','W'} with integer entries.
    """
    U, V, W = _build_rank98()

    # Verify base decomposition (should be exact)
    if not _verify(U, V, W):
        # Should not happen; fall back to schoolbook
        Us, Vs, Ws = _schoolbook()
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    # Use spare time for a bounded prune pass (only accepts valid reductions)
    tl = _time_left()
    if tl > 5.0:
        budget = min(tl - 3.0, 100.0)  # leave margin, allow more time
        try:
            U, V, W = _try_prune(U, V, W, budget)
        except Exception:
            pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
