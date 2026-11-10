"""Rewritten solver: exact cochromatic partition search with branch-and-bound."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

_N = P['n']
_ADJ = [set() for _ in range(_N)]
for _a, _b in P['edges']:
    _ADJ[_a].add(_b)
    _ADJ[_b].add(_a)
_NBR = [frozenset(_ADJ[v]) for v in range(_N)]
_START = time.time()
_DEADLINE = _START + 80.0


def _is_clique(s):
    for a in s:
        for b in s:
            if a != b and b not in _ADJ[a]:
                return False
    return True


def _is_indep(s):
    for a in s:
        for b in s:
            if a != b and b in _ADJ[a]:
                return False
    return True


def _greedy():
    es = {tuple(e) for e in P['edges']}
    parts = []
    for v in range(_N):
        placed = False
        for part in parts:
            trial = list(part) + [v]
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i + 1:]]
            if all(hits) or not any(hits):
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return parts


def _solve(k):
    """Return a list of k parts (clique or indep) covering all vertices, or None."""
    # Precompute: for every subset, whether it's a clique, an indep set, and its size.
    n = _N
    allc = [False] * (1 << n)
    allc[0] = True
    allin = [False] * (1 << n)
    allin[0] = True
    for s in range(1, 1 << n):
        lsb = s & (-s)
        v = lsb.bit_length() - 1
        rest = s ^ lsb
        allc[s] = allc[rest] and all(v in _ADJ[u] for u in range(n) if (rest >> u) & 1)
        allin[s] = allin[rest] and all(v not in _ADJ[u] for u in range(n) if (rest >> u) & 1)

    # Precompute max clique size and max indep size for every subset (for lower bounds).
    # mc[s] = max clique size within s ; mi[s] = max indep size within s
    mc = [0] * (1 << n)
    mi = [0] * (1 << n)
    for s in range(1, 1 << n):
        lsb = s & (-s)
        v = lsb.bit_length() - 1
        rest = s ^ lsb
        mc[s] = max(mc[rest], 1 + sum(1 for u in range(n) if (rest >> u) & 1 and v in _ADJ[u]))
        mi[s] = max(mi[rest], 1 + sum(1 for u in range(n) if (rest >> u) & 1 and v not in _ADJ[u]))

    # Precompute max clique size and max indep size for subsets that are cliques/indeps
    # (these equal the subset size). For a candidate part we only use clique/indep.

    best = {}  # frozenset(parts) -> True, for dominance pruning
    # Order parts canonically by sorted tuple for the key.
    parts = []
    full = (1 << n) - 1

    def canon(parts_list):
        return tuple(sorted(tuple(sorted(p)) for p in parts_list))

    def rec(uncovered, parts_list):
        # parts_list: list of frozensets already chosen (clique or indep)
        # uncovered: bitmask of vertices not yet covered
        if time.time() > _DEADLINE:
            return None
        if uncovered == 0:
            if len(parts_list) <= k:
                # pad to exactly k by splitting? We need exactly k parts, nonempty.
                # If we used fewer, we can split a part to fill (splitting a clique into
                # singletons keeps them cliques; splitting indep into singletons keeps indep).
                return list(parts_list)
            return None

        # Lower bound: need at least ceil of something.
        # Each remaining part can cover at most max clique size (if clique) or max indep (if indep)
        # of the uncovered set. So parts needed >= max( ceil(|uncovered| / maxpart), ... )
        # Better: we need to cover 'uncovered' with clique-or-indep parts.
        # Lower bound: the uncovered set has clique number w and independence number alpha.
        # Any clique part can cover at most w vertices of uncovered; any indep part at most alpha.
        # Worst case all parts are of the larger type: need >= ceil(|U| / max(w, alpha)).
        U = uncovered
        cnt = bin(U).count('1')
        maxw = mc[U]
        maxa = mi[U]
        mb = max(maxw, maxa)
        if mb == 0:
            return None
        need = (cnt + mb - 1) // mb
        used = len(parts_list)
        if used + need > k:
            return None

        # Pick the "hardest" uncovered vertex: highest degree within uncovered.
        # We'll branch on which part (existing or new) it joins, and clique/indep type.
        # Choose vertex v in U with max degree in G[U].
        v = -1
        bestdeg = -1
        for u in range(n):
            if (U >> u) & 1:
                d = bin((U & _ADJ[u] if u < n else 0)).count('1')
                if d > bestdeg:
                    bestdeg = d
                    v = u
        if v < 0:
            return None

        # Candidate part types for v:
        # Option A: v joins an existing part (must keep it clique or indep).
        # Option B: v starts a new part (clique or indep).
        # To reduce branching, consider: for each existing part, check if adding v keeps
        # it a clique (then new part = clique) or keeps it indep. For new part, v alone.

        # First, try to add v to existing parts (prefer to reuse).
        for idx, part in enumerate(parts_list):
            if v in part:
                continue
            newpart = part | (1 << v)
            if allc[newpart]:
                nl = list(parts_list)
                nl[idx] = newpart
                # prune duplicate: check if this configuration seen
                key = canon(nl) + (U ^ (1 << v),)
                if key in best:
                    continue
                best[key] = True
                r = rec(U ^ (1 << v), nl)
                if r is not None:
                    return r
            if allin[newpart]:
                nl = list(parts_list)
                nl[idx] = newpart
                key = canon(nl) + (U ^ (1 << v),)
                if key in best:
                    continue
                best[key] = True
                r = rec(U ^ (1 << v), nl)
                if r is not None:
                    return r

        # Option: start a new part containing v (if we have budget).
        if used < k:
            # new part = {v}
            nl = parts_list + [1 << v]
            key = canon(nl) + (U ^ (1 << v),)
            if key in best:
                pass
            else:
                best[key] = True
                r = rec(U ^ (1 << v), nl)
                if r is not None:
                    return r

        return None

    res = rec(full, [])
    if res is None:
        return None
    # Convert to lists of ints, ensure exactly k nonempty parts.
    out = [sorted(list(p)) for p in res]
    # Pad if fewer than k: split parts (singletons keep property).
    while len(out) < k:
        # split the largest part
        idx = max(range(len(out)), key=lambda i: len(out[i]))
        p = out[idx]
        if len(p) >= 2:
            out[idx] = p[:1]
            out.append(p[1:])
        else:
            # can't split; this shouldn't happen if we have budget
            break
    return out


def construct():
    # Greedy fallback.
    g = _greedy()
    best = g
    # Try to find a partition with fewer parts, starting from the lower bound.
    # Lower bound: max(chromatic(G), chromatic(complement)). We compute both via DP.
    n = _N
    # chromatic number via subset DP (standard).
    def chrom(adj):
        allc = [False] * (1 << n)
        allc[0] = True
        for s in range(1, 1 << n):
            lsb = s & (-s)
            v = lsb.bit_length() - 1
            rest = s ^ lsb
            allc[s] = allc[rest] and all(v in adj[u] for u in range(n) if (rest >> u) & 1)
        dp = [10 ** 9] * (1 << n)
        dp[0] = 0
        for s in range(1, 1 << n):
            if allc[s]:
                dp[s] = 1
            # iterate submasks that are cliques
            sub = s
            while sub:
                if allc[sub]:
                    r = s ^ sub
                    if dp[r] + 1 < dp[s]:
                        dp[s] = dp[r] + 1
                sub = (sub - 1) & s
        return dp[(1 << n) - 1]

    chi_g = chrom(_ADJ)
    comp = [set() for _ in range(n)]
    for a in range(n):
        for b in range(n):
            if a != b and b not in _ADJ[a]:
                comp[a].add(b)
    chi_c = chrom(comp)
    lb = max(chi_g, chi_c)

    # Try from lb upward, but cap at len(best)-1 (we already have a solution of len(best)).
    for k in range(lb, len(best)):
        if time.time() > _DEADLINE:
            break
        r = _solve(k)
        if r is not None:
            # validate
            ok = True
            seen = set()
            for part in r:
                if not part:
                    ok = False
                    break
                for x in part:
                    if x in seen:
                        ok = False
                        break
                    seen.add(x)
                if not (_is_clique(part) or _is_indep(part)):
                    ok = False
                    break
            if ok and len(seen) == n and len(r) == k:
                best = r
                break
    # Return as list of lists of ints.
    out = []
    for part in best:
        out.append([int(x) for x in part])
    return out
# EVOLVE-BLOCK-END
