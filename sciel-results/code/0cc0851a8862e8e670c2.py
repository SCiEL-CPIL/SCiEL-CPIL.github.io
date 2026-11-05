import itertools
import random
import time
import numpy as np

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
M = len(POINTS)  # 256

# ---------- Precompute COVER ----------
# COVER[p] = list of (a,b), a<b, a,b != p, such that triangle (p,a,b) is isosceles.
# This is used both for:
#   - no-isosceles: v compatible with S iff no pair in COVER[v] is fully in S
#   - maximality:   p (outside) is covered iff some pair in COVER[p] is fully in S
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
        d_pa = D2[p, A][:, None]   # M x 1
        d_pb = D2[p, A][None, :]   # 1 x M
        iso = (d_pa == d_pb) | (d_pa == D2) | (d_pb == D2)
        a = iu_a; b = iu_b
        m = (a != p) & (b != p)
        a = a[m]; b = b[m]
        sel = iso[a, b]
        cover.append(list(zip(a[sel].tolist(), b[sel].tolist())))
    return cover

COVER = _build_cover()


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


def _shrink(S):
    """Remove any point whose removal keeps the set maximal (inclusion-minimal maximal)."""
    S = set(S)
    inS = _make_inS(S)
    improved = True
    while improved:
        improved = False
        items = list(S)
        random.shuffle(items)
        for x in items:
            inS[x] = False
            S.discard(x)
            if _is_maximal(inS):
                improved = True
            else:
                inS[x] = True
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


def construct():
    t0 = time.time()
    deadline = t0 + 285.0  # safety margin well under the 330s kill

    # Initial solution: a few diverse greedy maximal sets, shrink each, keep best.
    best = None
    best_size = M + 1
    for _ in range(6):
        S = _shrink(_greedy_maximal())
        if len(S) < best_size:
            best = S
            best_size = len(S)

    # Iterated Local Search.
    plateau = 0
    strength = 1
    while time.time() < deadline:
        # Perturb the best solution.
        S = set(best)
        if random.random() < 0.5:
            # Remove a few random points (may break maximality).
            k = random.randint(1, max(1, len(S) // 2 + strength))
            for _ in range(k):
                if not S:
                    break
                x = random.choice(list(S))
                S.discard(x)
        else:
            # Single swap perturbation.
            if S:
                x = random.choice(list(S))
                S.discard(x)
                inS = _make_inS(S)
                for _ in range(8):
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
            if plateau % 25 == 0:
                strength = min(strength + 1, 4)

    # Final shrink if time permits.
    if time.time() < deadline - 2.0:
        best = _shrink(best)

    return [POINTS[i] for i in sorted(best)]
