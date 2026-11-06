import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def _build_graph(n, p, seed):
    """Build adjacency sets from the same PRNG stream as the seed program."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj


def _bron_kerbosch_max_cliques(adj, min_size=2):
    """Enumerate all maximal cliques of size >= min_size using Bron-Kerbosch with pivot."""
    n = len(adj)
    cliques = []

    def bk(R, P, X):
        if not P and not X:
            if len(R) >= min_size:
                cliques.append(frozenset(R))
            return
        U = P | X
        if not U:
            return
        pivot = max(U, key=lambda u: len(P & adj[u]))
        candidates = P - adj[pivot]
        for v in list(candidates):
            R.add(v)
            bk(R, P & adj[v], X & adj[v])
            R.remove(v)
            P.discard(v)
            X.add(v)

    bk(set(), set(range(n)), set())
    return cliques


# ---------- Maximum Weight Independent Set (greedy-coloring bound, Tomita) ----------
def _greedy_color_order(nodes, adj_set):
    """Return (order, color_bounds): greedy coloring to get an upper bound sequence."""
    # nodes: list of node ids; adj_set: dict node -> set of neighbors (within nodes)
    uncolored = set(nodes)
    color = {}
    order = []
    while uncolored:
        # pick a node, assign smallest color not used by already-colored neighbors
        # We process in a fixed order for determinism; color class = independent set
        for v in list(uncolored):
            used = set()
            for nb in adj_set.get(v, ()):
                if nb in color:
                    used.add(color[nb])
            c = 0
            while c in used:
                c += 1
            color[v] = c
            order.append(v)
            uncolored.discard(v)
            break
    # color_bounds[i] = max color used among order[:i+1]
    color_bounds = []
    mx = 0
    for v in order:
        mx = max(mx, color[v] + 1)
        color_bounds.append(mx)
    return order, color_bounds


def _mwis_upper_bound(nodes, adj_set):
    """Tomita-style greedy coloring upper bound on MWIS size for the induced subgraph on `nodes`."""
    if not nodes:
        return 0
    order, color_bounds = _greedy_color_order(nodes, adj_set)
    # The upper bound for the whole set is color_bounds[-1]
    return color_bounds[-1]


def _max_independent_set(nodes, adj_set, weight, time_limit=None, start_time=None):
    """
    Exact (or best-effort) maximum weight independent set on graph (nodes, adj_set)
    with node weights `weight`. Returns (weight, set_of_nodes).
    Uses branch-and-bound with greedy-coloring bound.
    """
    if start_time is None:
        start_time = time.time()
    best_w = [0]
    best_set = [set()]

    # Precompute adjacency restricted
    node_set = set(nodes)
    adj = {v: (adj_set.get(v, set()) & node_set) for v in nodes}

    # Order nodes by descending weight for better pruning
    nodes_sorted = sorted(nodes, key=lambda v: weight.get(v, 1), reverse=True)
    # Build a position map for coloring bound indexing
    pos = {v: i for i, v in enumerate(nodes_sorted)}

    def expand(R, P, X, depth):
        nonlocal best_w, best_set
        if time_limit is not None and (time.time() - start_time) > time_limit:
            return False
        # P = candidate set (independent from R), X = excluded
        if not P:
            w = sum(weight.get(v, 1) for v in R)
            if w > best_w[0]:
                best_w[0] = w
                best_set[0] = set(R)
            return True

        # Upper bound: |R| + greedy coloring bound on P
        P_list = list(P)
        ub = len(R) + _mwis_upper_bound(P_list, {v: adj.get(v, set()) for v in P_list})
        if len(R) + ub <= best_w[0]:
            return False

        # Branch: pick a node with high weight
        # Order P by weight desc
        P_sorted = sorted(P_list, key=lambda v: weight.get(v, 1), reverse=True)
        for v in P_sorted:
            R.add(v)
            P_new = P - adj.get(v, set()) - {v}
            X_new = X | (adj.get(v, set()) & P)
            expand(R, P_new, X_new, depth + 1)
            R.remove(v)
            P.discard(v)
            X.add(v)
            # If we can't beat best even adding all remaining, stop
            # (recompute cheap bound)
            if P:
                ub2 = len(R) + _mwis_upper_bound(list(P), {u: adj.get(u, set()) for u in P})
                if len(R) + ub2 <= best_w[0]:
                    return False
        return True

    expand(set(), set(nodes), set(), 0)
    return best_w[0], best_set[0]


def _solve_instance(adj, time_limit=40.0):
    """Solve the minimum transversal (hitting set) problem for maximal cliques."""
    n = len(adj)
    cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
    if not cliques:
        return []

    m = len(cliques)

    # vertex -> bitmask of cliques it covers
    clique_mask = [0] * n
    for i, clique in enumerate(cliques):
        for v in clique:
            clique_mask[v] |= (1 << i)

    vertex_freq = [bin(clique_mask[v]).count('1') for v in range(n)]

    # Precompute clique vertex lists sorted by freq desc
    clique_vertices = []
    for i, clique in enumerate(cliques):
        clique_vertices.append(sorted(clique, key=lambda v: vertex_freq[v], reverse=True))
    clique_sizes = [len(c) for c in cliques]

    # ---------- Initial upper bound: multi-start greedy + local search ----------
    def greedy(seed=0):
        rng = random.Random(seed)
        S = set()
        uncovered = (1 << m) - 1
        while uncovered:
            best_v, best_cnt = -1, -1
            for v in range(n):
                if v in S:
                    continue
                cnt = bin(clique_mask[v] & uncovered).count('1')
                if cnt > best_cnt or (cnt == best_cnt and rng.random() < 0.5):
                    best_cnt = cnt
                    best_v = v
            if best_v == -1 or best_cnt == 0:
                break
            S.add(best_v)
            uncovered &= ~clique_mask[best_v]
        return S if not uncovered else None

    best_solution = None
    best_size = m + 1
    for a in range(20):
        g = greedy(a)
        if g is not None and len(g) < best_size:
            best_size = len(g)
            best_solution = set(g)

    # Local search polish (1-remove, 1-swap)
    def polish(S, budget):
        S = set(S)
        improved = True
        while improved and time.time() < budget:
            improved = False
            for v in list(S):
                cand = S - {v}
                ok = all(c & cand for c in cliques)
                if ok:
                    S = cand
                    improved = True
                    break
            if not improved:
                for vr in list(S):
                    for va in range(n):
                        if va in S:
                            continue
                        cand = (S - {vr}) | {va}
                        if len(cand) >= len(S):
                            continue
                        if all(c & cand for c in cliques):
                            S = cand
                            improved = True
                            break
                    if improved:
                        break
        return S

    if best_solution is not None:
        polished = polish(best_solution, time.time() + min(3.0, time_limit * 0.1))
        if len(polished) < best_size:
            best_size = len(polished)
            best_solution = set(polished)

    # ---------- Lower bound via Maximum Independent Set on vertex-disjointness graph ----------
    # Nodes = cliques. Edge between i,j if C_i and C_j are vertex-disjoint.
    # A set of pairwise vertex-disjoint cliques must be hit by distinct vertices.
    # So max size of such a set (MWIS on disjointness graph) is a lower bound.
    disjoint_adj = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if not (cliques[i] & cliques[j]):
                disjoint_adj[i].add(j)
                disjoint_adj[j].add(i)

    def packing_lower_bound(uncovered_mask):
        """Exact (bounded) packing lower bound via MWIS on the disjointness graph induced by uncovered cliques."""
        if not uncovered_mask:
            return 0
        nodes = [i for i in range(m) if (uncovered_mask >> i) & 1]
        if len(nodes) <= 2:
            return len(nodes)
        # Use a fast greedy packing as a cheap bound; MWIS if small enough
        if len(nodes) <= 60:
            sub_adj = {i: disjoint_adj[i] & set(nodes) for i in nodes}
            w, _ = _max_independent_set(nodes, sub_adj, {i: 1 for i in nodes},
                                         time_limit=min(1.5, time_limit * 0.05))
            return w
        else:
            # Greedy packing fallback
            remaining = set(nodes)
            pack = 0
            while remaining:
                # pick a random-ish clique, add to pack, remove all that intersect it
                # choose the one with fewest disjoint neighbors to maximize packing
                v = min(remaining, key=lambda x: len(disjoint_adj[x] & remaining))
                pack += 1
                remaining -= (disjoint_adj[v] | {v})
            return pack

    # ---------- Branch and Bound ----------
    start_time = time.time()
    full_mask = (1 << m) - 1

    # Dominance: vertex a dominates b (among a set) if cover(a) & set == cover(b) & set and a cheaper
    # Precompute for quick use
    def dfs(current_size, uncovered_mask, path):
        nonlocal best_size, best_solution
        if time.time() - start_time > time_limit:
            return False
        if uncovered_mask == 0:
            if current_size < best_size:
                best_size = current_size
                best_solution = set(path)
            return True
        if current_size >= best_size:
            return False

        lb = packing_lower_bound(uncovered_mask)
        if current_size + lb >= best_size:
            return False

        # Pick most constrained uncovered clique (fewest candidate vertices)
        target = -1
        min_cand = 10 ** 9
        um = uncovered_mask
        while um:
            bit = um & -um
            idx = bit.bit_length() - 1
            # count vertices in clique idx that cover at least one uncovered clique
            cnt = 0
            for v in clique_vertices[idx]:
                if clique_mask[v] & uncovered_mask:
                    cnt += 1
            if cnt < min_cand:
                min_cand = cnt
                target = idx
            um ^= bit

        if target == -1:
            target = (uncovered_mask & -uncovered_mask).bit_length() - 1

        # Candidate vertices for this clique, sorted by uncovered coverage desc
        verts = [v for v in clique_vertices[target] if clique_mask[v] & uncovered_mask]
        verts.sort(key=lambda v: bin(clique_mask[v] & uncovered_mask).count('1'), reverse=True)

        # Dominance pruning: skip v if another v' covers a superset of v's uncovered cliques
        seen = {}
        used = set()
        for v in verts:
            cm = clique_mask[v] & uncovered_mask
            # if there's v' with (cm' & cm) == cm and cm' != cm, skip v
            dominated = False
            for v2, cm2 in seen.items():
                if (cm2 & cm) == cm and cm2 != cm:
                    dominated = True
                    break
            if dominated:
                continue
            seen[v] = cm
            path.append(v)
            dfs(current_size + 1, uncovered_mask & ~clique_mask[v], path)
            path.pop()
            if best_size <= current_size:
                return False
        return False

    dfs(0, full_mask, [])

    if best_solution is not None:
        return sorted(best_solution)

    # Fallback greedy
    for a in range(10):
        g = greedy(1000 + a)
        if g is not None:
            return sorted(g)
    return list(range(n))


def construct():
    deadline = time.time() + 145  # 5s safety margin
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    time_weights = [0.20, 0.35, 0.45]

    for idx, (n, p, seed) in enumerate(instances):
        remaining_time = deadline - time.time() - 3
        if remaining_time < 5:
            adj = _build_graph(n, p, seed)
            cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
            if not cliques:
                results.append([])
                continue
            m = len(cliques)
            S = set()
            uncovered = (1 << m) - 1
            vmask = [0] * n
            for i, c in enumerate(cliques):
                for v in c:
                    vmask[v] |= (1 << i)
            while uncovered:
                bv, bc = -1, -1
                for v in range(n):
                    if v in S:
                        continue
                    cnt = bin(vmask[v] & uncovered).count('1')
                    if cnt > bc:
                        bc, bv = cnt, v
                if bv == -1:
                    break
                S.add(bv)
                uncovered &= ~vmask[bv]
            results.append(sorted(S))
            continue

        time_budget = remaining_time * time_weights[idx]
        adj = _build_graph(n, p, seed)
        sol = _solve_instance(adj, time_limit=time_budget)
        results.append(sol)

    return results

# EVOLVE-BLOCK-END
