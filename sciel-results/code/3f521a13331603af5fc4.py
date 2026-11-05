import itertools
import random
import time
import numpy as np

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
M = len(POINTS)  # 256

# ---------- Precompute ----------
# COVER[p] = list of (a,b), a<b, a,b != p, such that triangle (p,a,b) is isosceles.
def _build_cover():
    xs = np.array([p[0] for p in POINTS], dtype=np.int64)
    ys = np.array([p[1] for p in POINTS], dtype=np.int64)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    D2 = dx * dx + dy * dy
    A = np.arange(M)
    iu_a, iu_b = np.triu_indices(M, k=1)
    cover = []
    for p in range(M):
        d_pa = D2[p, A][:, None]
        d_pb = D2[p, A][None, :]
        iso = (d_pa == d_pb) | (d_pa == D2) | (d_pb == D2)
        a = iu_a; b = iu_b
        m = (a != p) & (b != p)
        a = a[m]; b = b[m]
        sel = iso[a, b]
        cover.append(list(zip(a[sel].tolist(), b[sel].tolist())))
    return cover

COVER = _build_cover()

# For fast vectorized cover counting:
#   C_a[p] = np.array of a's, C_b[p] = np.array of b's
C_a = [np.array([ab[0] for ab in COVER[p]]) for p in range(M)]
C_b = [np.array([ab[1] for ab in COVER[p]]) for p in range(M)]

# For fast compatibility check:
#   COMPAT[p] = set of pairs (a,b) that would make p incompatible (i.e. a,b in S -> p blocked)
COMPAT = [set(COVER[p]) for p in range(M)]


def _make_inS(S):
    inS = [False] * M
    for v in S:
        inS[v] = True
    return inS


def _compatible(v, inS):
    for a, b in COVER[v]:
        if inS[a] and inS[b]:
            return False
    return True


def _is_maximal(inS):
    for p in range(M):
        if inS[p]:
            continue
        for a, b in COVER[p]:
            if inS[a] and inS[b]:
                break
        else:
            return False
    return True


def _cover_counts(S):
    """Vectorized: for each outside point p, count isosceles pairs (a,b) in S covering p."""
    Ss = sorted(S)
    if len(Ss) < 2:
        return np.zeros(M, dtype=np.int32)
    Sa = np.array(Ss)
    iu_a, iu_b = np.triu_indices(len(Ss), k=1)
    pa = Sa[iu_a]; pb = Sa[iu_b]
    counts = np.zeros(M, dtype=np.int32)
    for p in range(M):
        if p in S:
            continue
        ca = C_a[p]; cb = C_b[p]
        if ca.size == 0:
            continue
        counts[p] = int(np.sum((ca[:, None] == pa[None, :]) & (cb[:, None] == pb[None, :])))
    return counts


def _shrink(S):
    """Inclusion-minimal maximal: remove any point whose removal keeps maximality.
    Vectorized cover-count check: maximality holds iff every outside point has cover>=1."""
    S = set(S)
    # fast-path: if already maximal, use vectorized cover counting
    inS = _make_inS(S)
    if _is_maximal(inS):
        improved = True
        while improved:
            improved = False
            counts = _cover_counts(S)
            Sset = set(S)
            for x in S:
                if counts[x] > 0:
                    continue
                # removing x: recompute cover for outside points, check all >=1
                S2 = Sset - {x}
                c2 = _cover_counts(S2)
                outside = [p for p in range(M) if p not in S2]
                if all(c2[p] >= 1 for p in outside):
                    S = S2
                    improved = True
                    break
        return S
    # fallback: generic (may need repair first)
    improved = True
    while improved:
        improved = False
        items = list(S)
        random.shuffle(items)
        for x in items:
            inS = _make_inS(S)
            inS[x] = False
            S.discard(x)
            if _is_maximal(inS):
                improved = True
            else:
                S.add(x)
    return S


def _repair(S):
    """Greedy-complete S to a maximal no-isosceles set (adds compatible points)."""
    S = set(S)
    inS = _make_inS(S)
    order = list(range(M))
    random.shuffle(order)
    for v in order:
        if inS[v]:
            continue
        if _compatible(v, inS):
            S.add(v)
            inS[v] = True
    return S


def _greedy_maximal():
    S = set()
    inS = _make_inS(S)
    order = list(range(M))
    random.shuffle(order)
    for v in order:
        if _compatible(v, inS):
            S.add(v)
            inS[v] = True
    return S


def _compatible_fast(v, S):
    """Check if v compatible with set S using COMPAT[v]."""
    for a, b in COMPAT[v]:
        if a in S and b in S:
            return False
    return True


def construct():
    t0 = time.time()
    deadline = t0 + 285.0

    # Initial: diverse greedy maximal sets, shrink, keep best.
    best = None
    best_size = M + 1
    for _ in range(8):
        S = _shrink(_greedy_maximal())
        if len(S) < best_size:
            best = S
            best_size = len(S)

    # Iterated Local Search with vectorized shrink.
    plateau = 0
    strength = 1
    while time.time() < deadline:
        S = set(best)
        r = random.random()
        if r < 0.5:
            # Remove random points (may break maximality).
            k = random.randint(1, max(1, len(S) // 3 + strength))
            for _ in range(k):
                if not S:
                    break
                x = random.choice(list(S))
                S.discard(x)
        elif r < 0.8:
            # Single swap perturbation.
            if S:
                x = random.choice(list(S))
                S.discard(x)
                inS = _make_inS(S)
                for _ in range(12):
                    y = random.randrange(M)
                    if y in S:
                        continue
                    if _compatible(y, inS):
                        S.add(y)
                        inS[y] = True
                        break
        else:
            # Double swap.
            if len(S) >= 2:
                x1 = random.choice(list(S))
                S.discard(x1)
                x2 = random.choice(list(S))
                S.discard(x2)
                inS = _make_inS(S)
                for _ in range(16):
                    y = random.randrange(M)
                    if y in S:
                        continue
                    if _compatible(y, inS):
                        S.add(y)
                        inS[y] = True
                        break

        # Repair to maximal, then shrink to inclusion-minimal maximal.
        S = _repair(S)
        S = _shrink(S)

        if len(S) < best_size:
            best = S
            best_size = len(S)
            plateau = 0
            strength = 1
        else:
            plateau += 1
            if plateau % 20 == 0:
                strength = min(strength + 1, 5)

    # Final shrink if time permits.
    if time.time() < deadline - 2.0:
        best = _shrink(best)

    return [POINTS[i] for i in sorted(best)]
