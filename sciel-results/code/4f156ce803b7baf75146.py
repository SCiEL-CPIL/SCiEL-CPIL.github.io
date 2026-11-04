"""Type D kissing configuration in R^11 via support conflict graph MIS."""
import time
import numpy as np

D = 11
DEADLINE = 110.0  # wall-clock safety margin (limit 120s)


def _supports():
    """All 330 coordinate 2-subsets as (i, j) with i < j."""
    S = []
    for i in range(D):
        for j in range(i + 1, D):
            S.append((i, j))
    return S


SUPPORTS = _supports()
N = len(SUPPORTS)  # 330
# adjacency: adjacency[a] = bitmask of supports sharing >= 2 coords with a
ADJ = np.zeros(N, dtype=np.uint64)
for a in range(N):
    ia, ja = SUPPORTS[a]
    for b in range(N):
        if a == b:
            continue
        ib, jb = SUPPORTS[b]
        if ia == ib and ja == jb:
            continue
        # both share 2 coords -> same support (excluded) or share exactly one
        # share >= 2 only when identical, so no edges for 2-subsets?
        # Actually two distinct 2-sets share at most 1, so no conflicts.
        pass
# Build conflict edges properly: conflict if |S_a ∩ S_b| >= 2.
ADJ = np.zeros(N, dtype=np.uint64)
for a in range(N):
    ia, ja = SUPPORTS[a]
    for b in range(N):
        if a == b:
            continue
        ib, jb = SUPPORTS[b]
        inter = 0
        if ia == ib:
            inter += 1
        if ia == jb:
            inter += 1
        if ja == ib:
            inter += 1
        if ja == jb:
            inter += 1
        if inter >= 2:
            ADJ[a] |= (1 << b)
            ADJ[b] |= (1 << a)


def _marginal_gains(S, adj):
    """For each support, number of compatible supports not yet in S."""
    Sbits = 0
    for a in S:
        Sbits |= (1 << a)
    gains = np.zeros(N, dtype=np.int64)
    for a in range(N):
        if Sbits >> a & 1:
            continue
        # compatible with all of S if (adj[a] & Sbits) == 0
        # marginal gain = 1 (adding a) + compatible supports not in S
        if (adj[a] & Sbits) == 0:
            gains[a] = 1 + N - popcount(Sbits)
        else:
            gains[a] = 0
    return gains


def popcount(x):
    return bin(x).count("1")


def _greedy_setcover(adj, budget):
    """Vectorized greedy: repeatedly pick support maximizing marginal/penalty."""
    S = []
    Sbits = 0
    while True:
        if time.time() > budget:
            break
        # marginal gain: can a be added? compatible with all in S
        addable = []
        for a in range(N):
            if Sbits >> a & 1:
                continue
            if (adj[a] & Sbits) == 0:
                addable.append(a)
        if not addable:
            break
        # all addable have same gain (1 + remaining); pick one that
        # invalidates fewest future candidates (penalty = degree into unchosen)
        best = None
        best_pen = None
        for a in addable:
            # penalty = number of unchosen supports that a conflicts with
            pen = 0
            nb = adj[a]
            # count neighbors not in S and not a
            pen = popcount(nb & ~Sbits)
            if best_pen is None or pen < best_pen:
                best_pen = pen
                best = a
        S.append(best)
        Sbits |= (1 << best)
    return S


def _sim_anneal(adj, start, budget):
    """SA polish: try swaps to enlarge independent set."""
    S = set(start)
    Sbits = 0
    for a in S:
        Sbits |= (1 << a)
    best_size = len(S)
    best_set = set(S)
    T = 2.0
    cooling = 0.999
    while time.time() < budget:
        # pick a random support in S to remove
        if not S:
            break
        a = next(iter(S))
        # candidate b not in S compatible with S\{a}
        S_without_a = Sbits & ~(1 << a)
        # find candidates compatible with S_without_a
        cands = []
        for b in range(N):
            if Sbits >> b & 1:
                continue
            if (adj[b] & S_without_a) == 0:
                cands.append(b)
        if not cands:
            continue
        b = cands[np.random.randint(len(cands))]
        # delta: +1 if b is compatible with S\{a} (always true) and a was in S
        # new size = len(S) (remove a, add b) unless b already... b not in S
        # accept if improves or with prob exp(-1/T)
        if np.random.rand() < np.exp(-1.0 / T):
            S.remove(a)
            S.add(b)
            Sbits = (Sbits & ~(1 << a)) | (1 << b)
        T *= cooling
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    return list(best_set)


def _build_matrix(S):
    """Build integer vector array from selected supports (all 16 sign patterns)."""
    rows = []
    for (i, j) in [SUPPORTS[a] for a in S]:
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                rows.append(v)
    return np.array(rows, dtype=np.int64)


def construct():
    t0 = time.time()
    # Phase 1: greedy set-cover heuristic
    S = _greedy_setcover(ADJ, t0 + 20.0)
    # Phase 2: SA polish with remaining budget
    S = _sim_anneal(ADJ, S, DEADLINE - 5.0)
    return _build_matrix(S)
