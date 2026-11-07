"""Dense odd-order graphs avoiding equal-degree path endpoints."""
import time

# EVOLVE-BLOCK-START
_N = 15

def _generate_partitions(n, max_part):
    """All integer partitions of n with each part <= max_part, non-increasing order."""
    if n == 0:
        yield []
        return
    for first in range(min(max_part, n), 0, -1):
        for rest in _generate_partitions(n - first, first):
            yield [first] + rest

def _max_bipartite_edges(n):
    """Max sum over partitions of n of sum(parts*(parts-1)//2)."""
    best = 0
    best_parts = None
    for parts in _generate_partitions(n, n):
        e = sum(p * (p - 1) // 2 for p in parts)
        if e > best:
            best = e
            best_parts = list(parts)
    return best, best_parts

def _build_graph(parts):
    """Build complete bipartite components with given part sizes. Returns edge list."""
    edges = []
    v = 0
    for p in parts:
        if p <= 1:
            v += p
            continue
        a = p // 2
        b = p - a
        left = list(range(v, v + a))
        right = list(range(v + a, v + p))
        for x in left:
            for y in right:
                edges.append([x, y])
        v += p
    return edges

def _is_valid(edges, n):
    """Check: no simple 3-edge path s-u-v-t with deg(s)==deg(t)."""
    adj = [set() for _ in range(n)]
    for e in edges:
        u, v = e[0], e[1]
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]
    for u in range(n):
        for v in list(adj[u]):
            for s in list(adj[u]):
                if s == v:
                    continue
                for t in list(adj[v]):
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def construct():
    start = time.time()
    # Compute optimal partition
    best_edges_count, best_parts = _max_bipartite_edges(_N)
    edges = _build_graph(best_parts)
    # Verify validity (safety)
    if not _is_valid(edges, _N):
        # Fallback: seed star (always valid)
        edges = [[0, v] for v in range(1, _N)]
    # Deadline guard
    if time.time() - start > 50:
        return edges
    # Return with correct types
    result = [[int(u), int(v)] for u, v in edges]
    return result
# EVOLVE-BLOCK-END
