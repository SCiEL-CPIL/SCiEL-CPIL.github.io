import time

_DEADLINE = time.time() + 78.0  # safety margin under the 90s kill


def _cardinality_sums(S, n):
    """Return list where index k is the set of all k-subset sums of S."""
    n = len(S)
    sums = [set() for _ in range(n + 1)]
    sums[0].add(0)
    for v in S:
        for k in range(n, -1, -1):
            if k > 0:
                sums[k].update(s + v for s in sums[k - 1])
    return sums


def _cardinality_disjoint(sums, a, b):
    """True if sums[a] and sums[b] are disjoint (a != b)."""
    if a == b:
        return True
    sa, sb = sums[a], sums[b]
    if len(sa) > len(sb):
        sa, sb = sb, sa
    return not (sa & sb)


def _is_valid(S):
    n = len(S)
    sums = _cardinality_sums(S, n)
    for k in range(1, n + 1):
        for m in range(k + 1, n + 1):
            if not _cardinality_disjoint(sums, k, m):
                return False
    return True


def _is_valid_after_add(sums, S, v):
    """Check validity of S + [v] using the cardinality sums of S.

    sums[k] is the set of k-subset sums of S (0..len(S)).
    New sums: new_sums[k] = sums[k]  union  {s + v : s in sums[k-1]}.
    """
    n = len(S)
    new = [None] * (n + 2)
    new[0] = {0}
    for k in range(1, n + 2):
        a = set(sums[k]) if k <= n else set()
        if k - 1 >= 0:
            a.update(s + v for s in sums[k - 1])
        new[k] = a
    for k in range(1, n + 2):
        for m in range(k + 1, n + 2):
            if not _cardinality_disjoint(new, k, m):
                return False
    return True


def construct():
    start = time.time()
    S = []
    sums = [{0}]
    cands = list(range(1, 257))

    while True:
        if time.time() > _DEADLINE:
            break
        if len(S) >= 60:
            break
        n = len(S)
        # Cheap necessary-condition pruning of the candidate pool.
        # For a new k-subset sum to stay within [0, 256] for all k <= n+1,
        # we need v <= 256 - k*max(S) for the largest k that will be used.
        # A safe (necessary) bound: v + (n+1)*max(S) <= 256.
        mx = max(S) if S else 0
        limit = 256 - (n + 1) * mx
        if limit < 1:
            break
        cands = [c for c in cands if c <= limit]
        if not cands:
            break
        progress = False
        for v in cands:
            if time.time() > _DEADLINE:
                return S
            if _is_valid_after_add(sums, S, v):
                S.append(v)
                sums = _cardinality_sums(S, len(S))
                cands = [c for c in cands if c > v]
                progress = True
                break
        if not progress:
            # No candidate in the pruned pool works; try the full range once.
            full = [c for c in range(1, 257) if c not in S]
            done = False
            for v in full:
                if time.time() > _DEADLINE:
                    return S
                if _is_valid_after_add(sums, S, v):
                    S.append(v)
                    sums = _cardinality_sums(S, len(S))
                    done = True
                    break
            if not done:
                break

    if not isinstance(S, list):
        S = list(S)
    S = [int(x) for x in S]
    return S
