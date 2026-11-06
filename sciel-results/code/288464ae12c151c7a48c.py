import time
import random

_T0 = time.time()
DEADLINE = 145.0  # safety margin under 150s wall-clock kill


def _remaining():
    return DEADLINE - time.time()


def _random_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
                edges.add((u, v))
    return adj, edges


def _common_neighbors(u, v, adj):
    """Vertices connected to both u and v."""
    return adj[u] & adj[v]


def _grow_clique(seed_edge, adj, current_edges, trial_rng):
    """Grow a clique from seed_edge, only using edges still in current_edges."""
    u, v = seed_edge
    clique = {u, v}
    # candidates: common neighbors via uncovered edges
    cand = set()
    for w in adj[u]:
        if w == v:
            continue
        if (min(u, w), max(u, w)) in current_edges and (min(v, w), max(v, w)) in current_edges:
            cand.add(w)
    while cand:
        # pick candidate with max degree in current_edges (tie -> random)
        best_w = None
        max_score = -1
        for w in cand:
            score = sum(1 for a, b in current_edges if w in (a, b))
            if score > max_score:
                max_score = score
                best_w = w
            elif score == max_score and best_w is not None:
                if trial_rng.random() < 0.5:
                    best_w = w
        if best_w is None:
            break
        # verify it can be added (connected to all clique members via uncovered edges)
        can_add = True
        for w in clique:
            if (min(best_w, w), max(best_w, w)) not in current_edges:
                can_add = False
                break
        if not can_add:
            cand.discard(best_w)
            continue
        clique.add(best_w)
        new_cand = set()
        for x in cand:
            if x == best_w:
                continue
            if (min(best_w, x), max(best_w, x)) in current_edges:
                new_cand.add(x)
        cand = new_cand
    return clique


def _remove_clique_edges(clique, current_edges):
    cl = list(clique)
    for i in range(len(cl)):
        for j in range(i + 1, len(cl)):
            a, b = cl[i], cl[j]
            e = (min(a, b), max(a, b))
            current_edges.discard(e)


def _local_search_merge(cliques, adj):
    """Repeatedly merge two cliques whose union is a clique."""
    cliques = [list(c) for c in cliques]
    improved = True
    while improved and _remaining() > 3:
        improved = False
        for i in range(len(cliques)):
            if _remaining() < 3:
                break
            for j in range(i + 1, len(cliques)):
                if _remaining() < 3:
                    break
                union = set(cliques[i]) | set(cliques[j])
                if len(union) <= max(len(cliques[i]), len(cliques[j])):
                    continue
                # check union is a clique
                ok = True
                u_list = list(union)
                for a in range(len(u_list)):
                    for b in range(a + 1, len(u_list)):
                        if u_list[b] not in adj[u_list[a]]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    new = [c for k, c in enumerate(cliques) if k != i and k != j]
                    new.append(list(union))
                    cliques = new
                    improved = True
                    break
            if improved:
                break
    return cliques


def _one_trial(edges, adj, n, trial_rng):
    current_edges = set(edges)
    partition = []
    while current_edges:
        if _remaining() < 2:
            for e in list(current_edges):
                partition.append(list(e))
            break
        # choose seed edge weighted by number of common neighbors (potential size)
        best_edge = None
        best_weight = -1
        for e in current_edges:
            u, v = e
            w = len(_common_neighbors(u, v, adj))
            if w > best_weight:
                best_weight = w
                best_edge = e
            elif w == best_weight and best_edge is not None:
                if trial_rng.random() < 0.3:
                    best_edge = e
        if best_edge is None:
            best_edge = next(iter(current_edges))
        clique = _grow_clique(best_edge, adj, current_edges, trial_rng)
        if len(clique) < 2:
            clique = set(best_edge)
        partition.append(list(clique))
        _remove_clique_edges(clique, current_edges)
    return partition


def _solve_instance(n, p, seed, time_limit):
    adj, edges = _random_graph(n, p, seed)
    if not edges:
        return []
    trial_rng = random.Random(seed + 9999)
    best = None
    best_count = float('inf')
    start = time.time()
    trial = 0
    while True:
        if time.time() - start > time_limit or _remaining() < 5:
            break
        partition = _one_trial(edges, adj, n, trial_rng)
        partition = _local_search_merge(partition, adj)
        if len(partition) < best_count:
            best_count = len(partition)
            best = partition
        trial += 1
    if best is None:
        best = _local_search_merge([list(e) for e in edges], adj)
    return best


def construct():
    # Split budget: smaller instance less time, larger more.
    result1 = _solve_instance(32, 0.6, 601, time_limit=40)
    result2 = _solve_instance(64, 0.55, 602, time_limit=95)
    return [result1, result2]
