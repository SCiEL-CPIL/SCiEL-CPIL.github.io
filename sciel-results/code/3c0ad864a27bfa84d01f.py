# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

N = 128
VALS = list(range(1, N + 1))
IDX = {v: i for i, v in enumerate(VALS)}


def _ensure_pre():
    global _PRE, _MASK
    if '_PRE' in globals():
        return
    # Precompute gcd table and conflict masks (bit i set in conflict[j] iff
    # {i, j} are in some forbidden triple).
    g = [[0] * N for _ in range(N)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            g[a - 1][b - 1] = g[b - 1][a - 1] = gcd(a, b)
    conflict = [0] * N
    for a in range(N):
        va = VALS[a]
        for b in range(a + 1, N):
            vb = VALS[b]
            gab = g[a][b]
            for c in range(b + 1, N):
                if g[a][c] == gab and g[b][c] == gab:
                    conflict[a] |= (1 << b) | (1 << c)
                    conflict[b] |= (1 << a) | (1 << c)
                    conflict[c] |= (1 << a) | (1 << b)
    _PRE = g
    _MASK = conflict


def _can_add(m, x):
    """True iff adding element x keeps the set (mask m) valid."""
    for i in range(N):
        if (m >> i) & 1:
            if (_PRE[i][x] == _PRE[i][x] and _PRE[i][x] == _PRE[i][x]) and (
                    _PRE[i][x] == _PRE[i][x]):
                pass
    # correct check:
    for i in range(N):
        if (m >> i) & 1:
            if _PRE[i][x] == _PRE[i][x] and _PRE[i][x] == _PRE[i][x]:
                pass
    # do it properly:
    for i in range(N):
        if (m >> i) & 1:
            gix = _PRE[i][x]
            for j in range(i + 1, N):
                if (m >> j) & 1:
                    if _PRE[i][j] == gix and _PRE[j][x] == gix:
                        return False
    return True


def _greedy_construct():
    """Greedy by ascending value; returns (mask, size)."""
    m = 0
    s = 0
    for x in range(N):
        if _can_add(m, x):
            m |= (1 << x)
            s += 1
    return m, s


def _local_add(m, s):
    """Try to add any remaining element (single pass, repeated until stable)."""
    changed = True
    while changed:
        changed = False
        for x in range(N):
            if not ((m >> x) & 1) and _can_add(m, x):
                m |= (1 << x)
                s += 1
                changed = True
    return m, s


def _local_remove_add(m, s):
    """Swap moves: remove one, then greedily re-add as many as possible."""
    best_m, best_s = m, s
    for r in range(N):
        if not ((m >> r) & 1):
            continue
        m2 = m & ~(1 << r)
        s2 = s - 1
        # try adding all others
        for x in range(N):
            if not ((m2 >> x) & 1) and _can_add(m2, x):
                m2 |= (1 << x)
                s2 += 1
        if s2 > best_s:
            best_m, best_s = m2, s2
    return best_m, best_s


def _frac_bound(m, deadline):
    """Upper bound on additional elements addable to mask m via greedy
    fractional packing over the conflict graph. Returns an integer bound on
    total size (|m| + extra)."""
    s0 = bin(m).count('1')
    extra = 0
    remaining = [x for x in range(N) if not ((m >> x) & 1)]
    if not remaining:
        return s0
    # Greedy fractional: repeatedly pick the element with smallest degree
    # (fewest conflicts among remaining) and assign it weight 1/deg-ish.
    # Simpler tight bound: bound by max independent set of conflict graph on
    # remaining, approximated by greedy coloring upper bound (color count).
    # Use a greedy coloring of the induced conflict subgraph on remaining:
    # the chromatic number upper-bounds ... actually we need an UPPER bound on
    # independent set size. Use: alpha <= (sum of degrees + |V|)/|V| is weak.
    # Instead use LP-free bound: alpha <= |V| - (matching lower bound on
    # vertex cover). Compute a greedy maximal matching on conflict graph;
    # vertex cover >= matching size, so independent set <= |V| - matching.
    rem_set = set(remaining)
    used = [False] * len(remaining)
    matching = 0
    pos = {v: i for i, v in enumerate(remaining)}
    for i, u in enumerate(remaining):
        if used[i]:
            continue
        for v in remaining[i + 1:]:
            if used[pos[v]]:
                continue
            if (_MASK[u] >> v) & 1:
                matching += 1
                used[i] = True
                used[pos[v]] = True
                break
    # independent set size on remaining <= |remaining| - matching
    bound_extra = len(remaining) - matching
    return s0 + bound_extra


def _branch_and_bound(m, s, best, deadline):
    """Recursive branch-and-bound on remaining elements."""
    if time.time() > deadline:
        return best
    remaining = [x for x in range(N) if not ((m >> x) & 1)]
    if not remaining:
        return max(best, s)
    ub = _frac_bound(m, deadline)
    if ub <= best:
        return best
    # choose branch variable: the remaining element with highest conflict degree
    deg = [bin(_MASK[x] & m).count('1') + bin(_MASK[x] & ((1 << N) - 1) & ~(1 << x)).count('1') for x in remaining]
    # prefer high degree
    remaining_sorted = sorted(remaining, key=lambda x: bin(_MASK[x]).count('1'), reverse=True)
    x = remaining_sorted[0]
    # branch 1: include x if possible
    if _can_add(m, x):
        best = _branch_and_bound(m | (1 << x), s + 1, best, deadline)
    # branch 2: exclude x
    best = _branch_and_bound(m, s, best, deadline)
    return best


def _solve_bnb(deadline):
    # Start from greedy solution as lower bound
    m0, s0 = _greedy_construct()
    m0, s0 = _local_add(m0, s0)
    best = s0
    # Branch and bound on the full space with the greedy lower bound
    # To keep it tractable, we branch only over a restricted candidate set:
    # elements that are "conflicting" with the greedy set (potential swaps).
    # We do a shallow B&B with a node budget.
    node_budget = [200000]
    def bb(m, s, depth):
        nonlocal best
        if time.time() > deadline or node_budget[0] <= 0 or depth > 12:
            return
        node_budget[0] -= 1
        remaining = [x for x in range(N) if not ((m >> x) & 1)]
        if not remaining:
            if s > best:
                best = s
            return
        ub = _frac_bound(m, deadline)
        if ub <= best:
            return
        # order by degree descending
        rem_sorted = sorted(remaining, key=lambda x: bin(_MASK[x]).count('1'), reverse=True)
        x = rem_sorted[0]
        if _can_add(m, x):
            bb(m | (1 << x), s + 1, depth + 1)
        bb(m, s, depth + 1)
    bb(0, 0, 0)
    return best


def construct():
    start = time.time()
    deadline = start + 82.0  # safety margin before 90s kill
    _ensure_pre()

    # Phase 1: greedy + local improvement
    m, s = _greedy_construct()
    m, s = _local_add(m, s)
    m, s = _local_remove_add(m, s)
    m, s = _local_add(m, s)
    best_mask, best_size = m, s

    # Phase 2: branch-and-bound to try to beat the greedy lower bound
    if time.time() < deadline - 5.0:
        bnb_size = _solve_bnb(deadline - 5.0)
        if bnb_size > best_size:
            # Reconstruct: we only have the size; rebuild a valid set of that size
            # via greedy targeting that size. (Size is an upper-bound-guided value;
            # we accept it only if we can realize a valid set of that size.)
            # Realize by greedy + local search until we reach bnb_size or fail.
            m2, s2 = _greedy_construct()
            m2, s2 = _local_add(m2, s2)
            m2, s2 = _local_remove_add(m2, s2)
            if s2 >= bnb_size:
                best_mask, best_size = m2, s2

    # Final: extract list from best_mask
    result = [VALS[i] for i in range(N) if (best_mask >> i) & 1]
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
