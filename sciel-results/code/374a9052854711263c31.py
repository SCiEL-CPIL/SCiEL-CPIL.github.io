"""Rewritten: greedy clique-subdivision maximization with deadline guard."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

N = P['n']
ADJ = [set() for _ in range(N)]
for a, b in P['edges']:
    ADJ[a].add(b)
    ADJ[b].add(a)

DEADLINE = time.time() + 80.0  # safety margin before 90s kill


def _build_result(branches):
    """Try to assign pairwise interior-disjoint simple paths among branches."""
    if len(branches) < 2:
        return None
    bset = set(branches)
    # order pairs by a cheap difficulty proxy: fewer available interior helpers first
    pairs = list(combinations(branches, 2))
    pairs.sort(key=lambda p: len(ADJ[p[0]] & ADJ[p[1]] - bset))
    used = set()
    paths = {}
    for u, v in pairs:
        # BFS from u to v avoiding branches (except endpoints) and used interiors
        prev = {u: None}
        q = [u]
        found = False
        while q:
            cur = q.pop(0)
            if cur == v:
                found = True
                break
            for nb in ADJ[cur]:
                if nb in prev:
                    continue
                if nb in bset and nb != v:
                    continue
                if nb in used:
                    continue
                prev[nb] = cur
                q.append(nb)
        if not found:
            return None
        # reconstruct
        path = [v]
        while path[-1] != u:
            path.append(prev[path[-1]])
        path.reverse()
        for x in path[1:-1]:
            used.add(x)
        paths[(u, v)] = path
    # build dict keyed by unordered pair
    out = {}
    for (u, v), path in paths.items():
        out[frozenset((u, v))] = path
    # ensure every unordered pair present
    for u, v in combinations(branches, 2):
        out[frozenset((u, v))]
    return out


def _greedy_clique():
    """Greedy maximal clique using degree ordering."""
    order = sorted(range(N), key=lambda x: -len(ADJ[x]))
    clique = []
    for v in order:
        if all(v in ADJ[u] for u in clique):
            clique.append(v)
    return clique


def construct():
    t0 = time.time()
    best = None  # (size, branches, pathsdict)

    # candidate branch sets: greedy clique + many random-ish seeds
    candidates = []
    g = _greedy_clique()
    candidates.append(g)
    # several deterministic orderings
    for start in range(0, N, 3):
        order = sorted(range(N), key=lambda x: ((x - start) % N) - len(ADJ[x]))
        clique = []
        for v in order:
            if all(v in ADJ[u] for u in clique):
                clique.append(v)
        if clique:
            candidates.append(clique)

    for cand in candidates:
        if time.time() > DEADLINE:
            break
        if len(cand) < 2:
            continue
        paths = _build_result(cand)
        if paths is not None:
            if best is None or len(cand) > len(best[1]):
                best = (len(cand), list(cand), paths)

    # fallback: guaranteed valid 2-branch edge
    if best is None:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    branches = [int(x) for x in best[1]]
    paths_list = []
    for u, v in combinations(branches, 2):
        p = best[2][frozenset((u, v))]
        paths_list.append([int(x) for x in p])
    return {'branches': branches, 'paths': paths_list}
# EVOLVE-BLOCK-END
