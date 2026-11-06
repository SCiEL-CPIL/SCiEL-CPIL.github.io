"""Search for large graphs with no regular induced subgraph on 6+ vertices."""
import time
import random
import itertools

_START_TIME = None
_TIME_LIMIT = 140

def _elapsed():
    return time.time() - _START_TIME

def _time_exceeded():
    return _START_TIME is None or _elapsed() > _TIME_LIMIT

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_regular(adj, subset):
    """Check if induced subgraph on subset is regular."""
    k = len(subset)
    if k < 2:
        return True
    subset_set = set(subset)
    first_deg = len(adj[subset[0]] & subset_set)
    for v in subset[1:]:
        if len(adj[v] & subset_set) != first_deg:
            return False
    return True

def _has_regular_subgraph(adj, n, min_size=6):
    """Returns (True, subset) if any subset of size >= min_size is regular."""
    for size in range(min_size, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular(adj, subset):
                return True, subset
    return False, None

def _validate(n, edges):
    adj = _build_adj(n, edges)
    has_reg, _ = _has_regular_subgraph(adj, n, min_size=6)
    return not has_reg

def _random_graph(n, p, seed=None):
    rng = random.Random(seed)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append([i, j])
    return edges

def _flip(n, edges, u, v):
    edge_set = set()
    for e in edges:
        a, b = e
        edge_set.add((min(a,b), max(a,b)))
    key = (min(u,v), max(u,v))
    if key in edge_set:
        edge_set.remove(key)
    else:
        edge_set.add(key)
    return [[a,b] for a,b in edge_set]

def _local_search(n, edges, max_iters=3000, seed=None):
    rng = random.Random(seed)
    cur = list(edges)
    for _ in range(max_iters):
        if _time_exceeded():
            break
        adj = _build_adj(n, cur)
        has_reg, subset = _has_regular_subgraph(adj, n, min_size=6)
        if not has_reg:
            return cur, True
        # Flip a random edge within the violating subset
        if len(subset) >= 2:
            u = subset[rng.randint(0, len(subset)-1)]
            v = subset[rng.randint(0, len(subset)-1)]
            if u != v:
                cur = _flip(n, cur, u, v)
    adj = _build_adj(n, cur)
    has_reg, _ = _has_regular_subgraph(adj, n, min_size=6)
    return cur, not has_reg

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_n = 5
    best_edges = []
    
    for n in range(6, 23):
        if _time_exceeded():
            break
        found = False
        for attempt in range(300):
            if _time_exceeded():
                break
            seed = attempt * 104729 + n * 31
            p_choices = [0.15, 0.25, 0.35, 0.45, 0.55]
            p = p_choices[attempt % len(p_choices)]
            edges = _random_graph(n, p, seed=seed)
            if _validate(n, edges):
                found = True
                best_n = n
                best_edges = edges
                break
            # Local search
            edges_ls, valid = _local_search(n, edges, max_iters=1500, seed=seed)
            if valid:
                found = True
                best_n = n
                best_edges = edges_ls
                break
        if not found and n >= 8:
            # If we can't find for n, larger n is unlikely
            # But give a few more tries for n=7,8
            if n > 10:
                break
    
    return {'n': best_n, 'edges': best_edges}
