import time

_P = 27720  # lcm(1..12)
_N = 12
_UB = 144   # contract: raw f(12,m) <= 144


def _first_multiple_gt(r, i):
    # smallest multiple of i strictly greater than r
    return (r // i + 1) * i


def _has_matching(r, L):
    """Check if there is a system of distinct representatives:
    assign each i in 1..N a distinct b in (r, r+L] with i | b."""
    # Build adjacency: for each i (0..N-1 representing i+1), list of candidate b values
    adj = []
    b_to_idx = {}
    idx = 0
    for i in range(1, _N + 1):
        lst = []
        b = _first_multiple_gt(r, i)
        hi = r + L
        while b <= hi:
            if b not in b_to_idx:
                b_to_idx[b] = idx
                idx += 1
            lst.append(b_to_idx[b])
            b += i
        adj.append(lst)

    if idx == 0:
        return False

    # Hall's theorem quick check: for every subset of left nodes, |N(S)| >= |S|
    # For N=12 we can check all 2^12 subsets, but that's 4096 which is fine.
    # Actually, let's just do the matching directly with Hopcroft-Karp or DFS.
    # For 12 left nodes, simple DFS augmenting path is fast enough.

    matchR = [-1] * idx  # matchR[v] = u means right node v matched to left node u

    def dfs(u, seen):
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            if matchR[v] == -1 or dfs(matchR[v], seen):
                matchR[v] = u
                return True
        return False

    cnt = 0
    for u in range(_N):
        seen = [False] * idx
        if dfs(u, seen):
            cnt += 1
    return cnt == _N


def _f_for_residue(r):
    """Binary search for the smallest L in [1, UB] such that matching exists."""
    lo, hi = 1, _UB
    if not _has_matching(r, _UB):
        return _UB
    ans = _UB
    while lo <= hi:
        mid = (lo + hi) // 2
        if _has_matching(r, mid):
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans


def construct():
    deadline = time.time() + 85.0  # safety margin under 90s
    best_r = 0
    best_f = -1

    # Precompute f for all residues mod P. Periodicity: f(12, m) depends only on m mod P.
    # We scan all P residues; if time runs out, return best found.
    for r in range(_P):
        if time.time() > deadline:
            break
        f = _f_for_residue(r)
        if f > best_f:
            best_f = f
            best_r = r

    return int(best_r)
