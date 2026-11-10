import time
import random
from collections import defaultdict

def _instance(n, p, seed):
    """Exact copy of the verifier's instance generator."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adj(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _edge_set(edges):
    return {tuple(sorted(e)) for e in edges}


def _paths_from_edges(edge_list, adj):
    """Cover edges with simple paths. Uses a greedy trail-based method:
    treat it as finding edge-disjoint trails, then ensure simplicity."""
    remaining = _edge_set(edge_list)
    deg = defaultdict(int)
    for u, v in remaining:
        deg[u] += 1
        deg[v] += 1
    paths = []
    rng = random.Random(12345)
    while remaining:
        # Pick a start vertex: prefer odd-degree or low-degree
        candidates = [v for v, d in deg.items() if d > 0]
        if not candidates:
            break
        # Prefer leaves
        leaf = [v for v in candidates if deg[v] == 1]
        if leaf:
            start = rng.choice(leaf)
        else:
            start = min(candidates, key=lambda v: deg[v])
        path = [start]
        used_edges = set()
        while True:
            v = path[-1]
            nbrs = []
            for u in adj[v]:
                e = tuple(sorted((v, u)))
                if e in remaining and e not in used_edges:
                    nbrs.append(u)
            if not nbrs:
                break
            # Prefer unvisited
            unvis = [u for u in nbrs if u not in path]
            if unvis:
                nxt = min(unvis, key=lambda u: deg[u])
            else:
                break
            e = tuple(sorted((v, nxt)))
            used_edges.add(e)
            remaining.discard(e)
            deg[v] -= 1
            deg[nxt] -= 1
            if deg[v] <= 0:
                del deg[v]
            if deg[nxt] <= 0:
                del deg[nxt]
            path.append(nxt)
        if len(path) >= 2:
            paths.append(path)
    # leftover
    for e in remaining:
        paths.append([e[0], e[1]])
    return paths


def _get_path_edges(path):
    return [tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1)]


def _verify(paths, edges):
    adj = _build_adj(edges)
    eset = _edge_set(edges)
    used = set()
    for p in paths:
        if len(p) < 2:
            return False
        if len(set(p)) != len(p):
            return False
        for i in range(len(p)-1):
            e = tuple(sorted((p[i], p[i+1])))
            if e not in eset:
                return False
            if e in used:
                return False
            used.add(e)
    return used == eset


def _merge_paths(paths, adj, eset):
    """Try to reduce path count via 2-path merge.
    For two paths P1=[a,...,b], P2=[c,...,d], if edges (b,c) and (a,d) or (b,d) and (a,c) exist and aren't used in other ways, we can sometimes merge.
    Actually a simpler local op: if endpoint of P1 is adjacent to endpoint of P2 via an unused edge, we can concatenate. But all edges are used.
    Alternative: find two paths where an internal edge can be 'flipped'. 
    For now, try: for each pair of paths, check if reversing one and checking endpoints adjacency helps.
    """
    n = len(paths)
    for i in range(n):
        for j in range(i+1, n):
            P1 = list(paths[i])
            P2 = list(paths[j])
            # Try: if P1[-1] == P2[0] (not possible in simple paths without shared vertex... actually paths can share vertices, just not edges)
            # Wait - simple paths in edge decomposition can share vertices! Only edges must be disjoint.
            # So if P1 ends at x and P2 starts at x, we can merge: P1 + P2[1:]
            if P1[-1] == P2[0]:
                merged = P1 + P2[1:]
                if len(set(merged)) == len(merged):  # still simple
                    paths[i] = merged
                    paths.pop(j)
                    return True
            if P1[-1] == P2[-1]:
                P2r = P2[::-1]
                merged = P1 + P2r[1:]
                if len(set(merged)) == len(merged):
                    paths[i] = merged
                    paths.pop(j)
                    return True
            if P1[0] == P2[0]:
                P1r = P1[::-1]
                merged = P1r + P2[1:]
                if len(set(merged)) == len(merged):
                    paths[i] = merged
                    paths.pop(j)
                    return True
            if P1[0] == P2[-1]:
                merged = P1[::-1] + P2[1:]
                if len(set(merged)) == len(merged):
                    paths[i] = merged
                    paths.pop(j)
                    return True
    return False


def _improve_local(paths, edges, adj, time_limit=5.0, rng=None):
    """Local search: repeatedly try merges until no improvement."""
    if rng is None:
        rng = random.Random(777)
    start_t = time.time()
    improved = True
    while improved:
        if time.time() - start_t > time_limit:
            break
        improved = _merge_paths(paths, adj, _edge_set(edges))
    return paths


def _solve_one(edges, time_limit=40.0):
    adj = _build_adj(edges)
    best = None
    best_count = float('inf')
    t0 = time.time()
    for trial in range(200):
        if time.time() - t0 > time_limit - 1.0:
            break
        seed = trial * 31 + 17
        paths = _paths_from_edges(edges, adj)
        paths = _improve_local(paths, edges, adj, time_limit=0.1, rng=random.Random(seed))
        cnt = len(paths)
        if cnt < best_count:
            best_count = cnt
            best = [list(p) for p in paths]
            if best_count <= 1:
                break
    if best is None:
        best = _paths_from_edges(edges, adj)
    return best


def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    n_graphs = len(specs)
    for i, (n, p, seed) in enumerate(specs):
        remaining = deadline - time.time()
        if remaining < 5:
            edges = _instance(n, p, seed)
            adj = _build_adj(edges)
            results.append(_paths_from_edges(edges, adj))
            continue
        alloc = max(2.0, (remaining - 1.0) / (n_graphs - i))
        edges = _instance(n, p, seed)
        paths = _solve_one(edges, time_limit=alloc)
        results.append(paths)
    return results


if __name__ == "__main__":
    res = construct()
    for i, paths in enumerate(res):
        print(f"Graph {i}: {len(paths)} paths")
    print(f"Total: {sum(len(p) for p in res)}")
