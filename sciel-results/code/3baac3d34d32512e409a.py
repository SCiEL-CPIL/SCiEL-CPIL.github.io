"""Edge-level set-cover / MILP approach for exact biclique edge partition."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _validate(bicliques, adj, all_edges_set):
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        if set(left).intersection(right):
            return False
        for u in left:
            for v in right:
                if v not in adj[u]:
                    return False
                e = (min(u, v), max(u, v))
                if e in covered:
                    return False
                covered.add(e)
    return covered == all_edges_set


def _biclique_edges(left, right):
    return {(min(u, v), max(u, v)) for u in left for v in right}


def _expand_maximal(start_left, start_right, adj, n):
    left = set(start_left)
    right = set(start_right)
    changed = True
    while changed:
        changed = False
        for u in range(n):
            if u not in left and u not in right and right.issubset(adj[u]):
                left.add(u)
                changed = True
        for v in range(n):
            if v not in left and v not in right and left.issubset(adj[v]):
                right.add(v)
                changed = True
    left -= right
    right -= left
    if not left or not right:
        return None
    return (left, right)


def _generate_candidates(adj, n, all_edges_set, deadline):
    """Generate candidate bicliques: maximal expansions of every edge,
    plus random edge expansions for diversity."""
    cand_set = set()
    for (u, v) in all_edges_set:
        if time.time() > deadline:
            break
        bc = _expand_maximal({u}, {v}, adj, n)
        if bc:
            cand_set.add((frozenset(bc[0]), frozenset(bc[1])))
        bc = _expand_maximal({v}, {u}, adj, n)
        if bc:
            cand_set.add((frozenset(bc[0]), frozenset(bc[1])))

    random.seed(123)
    edge_list = list(all_edges_set)
    for _ in range(200):
        if time.time() > deadline:
            break
        e = random.choice(edge_list)
        bc = _expand_maximal({e[0]}, {e[1]}, adj, n)
        if bc:
            cand_set.add((frozenset(bc[0]), frozenset(bc[1])))
        bc = _expand_maximal({e[1]}, {e[0]}, adj, n)
        if bc:
            cand_set.add((frozenset(bc[0]), frozenset(bc[1])))

    candidates = []
    for l, r in cand_set:
        edges = _biclique_edges(l, r)
        if edges:
            candidates.append((set(l), set(r), edges))
    return candidates


def _greedy_set_cover(candidates, all_edges_set, deadline):
    """Greedy set cover over candidate bicliques."""
    remaining = set(all_edges_set)
    chosen = []
    while remaining:
        if time.time() > deadline:
            break
        best_i = -1
        best_cov = 0
        for i, (l, r, edges) in enumerate(candidates):
            cov = len(edges & remaining)
            if cov > best_cov:
                best_cov = cov
                best_i = i
        if best_i < 0:
            e = next(iter(remaining))
            chosen.append(({e[0]}, {e[1]}))
            remaining.discard(e)
        else:
            l, r, edges = candidates[best_i]
            chosen.append((l, r))
            remaining -= edges
    return chosen


def _local_search(bicliques, adj, all_edges_set, deadline):
    """Try to merge pairs of bicliques if their union is a valid biclique
    covering exactly the same edge set."""
    bicliques = [[list(l), list(r)] for l, r in bicliques]

    def edge_set_of(bc):
        return _biclique_edges(set(bc[0]), set(bc[1]))

    improved = True
    while improved and time.time() < deadline:
        improved = False
        edge_sets = [edge_set_of(bc) for bc in bicliques]
        for i in range(len(bicliques)):
            for j in range(i + 1, len(bicliques)):
                if time.time() > deadline:
                    break
                li, ri = bicliques[i]
                lj, rj = bicliques[j]
                Li, Ri, Lj, Rj = set(li), set(ri), set(lj), set(rj)
                left_opts = [Li, Lj, Li | Lj, Li & Lj]
                right_opts = [Ri, Rj, Ri | Rj, Ri & Rj]
                found = False
                for nl in left_opts:
                    if not nl:
                        continue
                    for nr in right_opts:
                        if not nr or nl & nr:
                            continue
                        valid = all(v in adj[u] for u in nl for v in nr)
                        if not valid:
                            continue
                        merged_edges = _biclique_edges(nl, nr)
                        if merged_edges != edge_sets[i] | edge_sets[j]:
                            continue
                        overlap = False
                        for k in range(len(bicliques)):
                            if k == i or k == j:
                                continue
                            if merged_edges & edge_sets[k]:
                                overlap = True
                                break
                        if overlap:
                            continue
                        bicliques[i] = [list(nl), list(nr)]
                        del bicliques[j]
                        improved = True
                        found = True
                        break
                    if found:
                        break
                if improved:
                    break
            if improved:
                break
    return bicliques


def _branch_and_bound(candidates, all_edges_set, deadline, incumbent):
    """Exact B&B over candidate pool. incumbent is a list of bicliques
    (left,right) whose count is the current best."""
    edge_to_cands = {e: [] for e in all_edges_set}
    for i, (l, r, edges) in enumerate(candidates):
        for e in edges:
            edge_to_cands[e].append(i)

    best = [len(incumbent)]
    best_sol = [list(incumbent)]

    # Sort candidates by size descending for better pruning
    order = sorted(range(len(candidates)), key=lambda i: -len(candidates[i][2]))

    remaining = set(all_edges_set)
    chosen = []

    def rec(remaining, chosen, depth):
        if time.time() > deadline:
            return
        if not remaining:
            if len(chosen) < best[0]:
                best[0] = len(chosen)
                best_sol[0] = [list(c) for c in chosen]
            return
        if len(chosen) + 1 >= best[0]:
            return
        # Pick edge with fewest candidate options
        e = min(remaining, key=lambda x: len(edge_to_cands[x]))
        opts = edge_to_cands[e]
        # Sort options by coverage descending
        opts_sorted = sorted(opts, key=lambda i: -len(candidates[i][2] & remaining))
        for i in opts_sorted:
            l, r, edges = candidates[i]
            new_rem = remaining - edges
            chosen.append((l, r))
            rec(new_rem, chosen, depth + 1)
            chosen.pop()
            if time.time() > deadline:
                return
            if len(chosen) + 1 >= best[0]:
                return

    rec(remaining, chosen, 0)
    return best_sol[0] if best[0] < len(incumbent) else None


def _milp_solve(candidates, all_edges_set, deadline):
    """Solve the set cover ILP via scipy.optimize.milp."""
    m = len(all_edges_set)
    k = len(candidates)
    if k == 0:
        return None

    edge_index = {e: i for i, e in enumerate(all_edges_set)}
    A = np.zeros((m, k), dtype=np.int8)
    for j, (l, r, edges) in enumerate(candidates):
        for e in edges:
            A[edge_index[e], j] = 1

    c = np.ones(k)
    constraints = [LinearConstraint(A, 1, np.inf)]
    integrality = np.ones(k)
    bounds = Bounds(0, 1)

    try:
        res = milp(c, integrality=integrality, bounds=bounds, constraints=constraints,
                   options={"time_limit": max(1, int(deadline - time.time()))})
        if res.status == 0 and res.x is not None:
            chosen = [i for i in range(k) if res.x[i] > 0.5]
            bicliques = []
            for i in chosen:
                l, r, _ = candidates[i]
                bicliques.append((l, r))
            return bicliques
    except Exception:
        pass
    return None


def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s kill

    n = P['n']
    adj = _build_adjacency()
    all_edges_set = set((min(u, v), max(u, v)) for u, v in P['edges'])

    # 1. Generate candidate pool
    candidates = _generate_candidates(adj, n, all_edges_set, deadline)

    # 2. Greedy set cover as initial incumbent
    incumbent = _greedy_set_cover(candidates, all_edges_set, deadline)
    if not _validate(incumbent, adj, all_edges_set):
        # Fallback: single-edge bicliques
        incumbent = [({u}, {v}) for (u, v) in all_edges_set]

    best_result = incumbent
    best_count = len(incumbent)

    # 3. Local search polish
    if time.time() < deadline:
        improved = _local_search(incumbent, adj, all_edges_set, deadline)
        if _validate(improved, adj, all_edges_set) and len(improved) < best_count:
            best_result = improved
            best_count = len(improved)

    # 4. MILP polish
    if time.time() < deadline:
        milp_res = _milp_solve(candidates, all_edges_set, deadline)
        if milp_res and _validate(milp_res, adj, all_edges_set):
            if len(milp_res) < best_count:
                best_count = len(milp_res)
                best_result = milp_res
            if time.time() < deadline:
                improved = _local_search(milp_res, adj, all_edges_set, deadline)
                if _validate(improved, adj, all_edges_set) and len(improved) < best_count:
                    best_count = len(improved)
                    best_result = improved

    # 5. B&B polish (only if candidate pool is small enough to be tractable)
    if time.time() < deadline and len(candidates) <= 200:
        bb_res = _branch_and_bound(candidates, all_edges_set, deadline, best_result)
        if bb_res and _validate(bb_res, adj, all_edges_set) and len(bb_res) < best_count:
            best_count = len(bb_res)
            best_result = bb_res

    # Convert to required types
    result = []
    for left, right in best_result:
        result.append([list(int(x) for x in left), list(int(x) for x in right)])
    return result
# EVOLVE-BLOCK-END
