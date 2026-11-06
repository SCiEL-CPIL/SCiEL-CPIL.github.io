"""Construct a graph with no regular induced subgraph on >= 6 vertices."""
import time
import random
from itertools import combinations

# Deadline guard: evaluator kills at 150s; keep a safety margin.
_START_TIME = time.time()
DEADLINE = 125.0  # seconds from start; leave buffer before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _is_regular(subset, adj):
    """Check if the induced subgraph on 'subset' is regular.
    adj is a list of sets (or frozensets) for fast membership.
    """
    n = len(subset)
    if n < 6:
        return False
    degs = []
    for u in subset:
        d = 0
        for v in subset:
            if u != v and v in adj[u]:
                d += 1
        degs.append(d)
    # Regular iff all degrees equal
    first = degs[0]
    for d in degs[1:]:
        if d != first:
            return False
    return True

def _has_violation(n, adj):
    """Check if any subset of size >= 6 induces a regular subgraph.
    adj: list of sets.
    Returns True if violation exists.
    """
    vertices = list(range(n))
    # Check all subsets of size 6, 7, ..., n
    for size in range(6, n + 1):
        for sub in combinations(vertices, size):
            if _is_regular(sub, adj):
                return True
    return False

def _has_violation_fast(n, adj, check_sizes=None):
    """Faster violation check using bitmasks for small n.
    adj: list of int bitmasks.
    """
    if check_sizes is None:
        check_sizes = list(range(6, n + 1))
    for size in check_sizes:
        for sub in combinations(range(n), size):
            # Compute degrees
            degs = []
            for i, u in enumerate(sub):
                d = 0
                for j, v in enumerate(sub):
                    if i != j:
                        if (adj[u] >> v) & 1:
                            d += 1
                degs.append(d)
            first = degs[0]
            ok = True
            for d in degs[1:]:
                if d != first:
                    ok = False
                    break
            if ok:
                return True
    return False

def _build_adj_list(edges, n):
    """Build adjacency as list of sets."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _build_adj_bitmask(edges, n):
    """Build adjacency as list of int bitmasks."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _random_graph(n, p=0.5, seed=None):
    """Generate a random graph with edge probability p."""
    rng = random.Random(seed)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append([i, j])
    return n, edges

def _greedy_construct(n, seed=42):
    """Try to construct a graph on n vertices with no regular induced subgraph on >= 6 vertices.
    Uses randomized local search.
    """
    rng = random.Random(seed)
    
    # Start with a random graph
    p = 0.5
    n_curr, edges = _random_graph(n, p, seed)
    adj = _build_adj_list(edges, n_curr)
    
    # Check initial
    if not _has_violation(n_curr, adj):
        return {'n': n_curr, 'edges': edges}
    
    best_n = n_curr
    best_edges = edges
    
    # Local search: try flipping edges to eliminate violations
    max_iter = 5000
    for it in range(max_iter):
        if _elapsed() > DEADLINE:
            break
        
        # Find a violating subset
        violation_found = False
        for size in range(6, n_curr + 1):
            # Sample some subsets to save time
            attempts = 0
            while attempts < 200:
                sub = tuple(sorted(rng.sample(range(n_curr), size)))
                if _is_regular(sub, adj):
                    violation_found = True
                    break
                attempts += 1
            if violation_found:
                break
        
        if not violation_found:
            # No violation found in sampling; verify fully
            if not _has_violation(n_curr, adj):
                return {'n': n_curr, 'edges': edges}
            # Still has violation, continue searching
            continue
        
        # Try to fix by flipping a random edge
        # Pick a random edge (u,v) and flip it
        u = rng.randint(0, n_curr - 1)
        v = rng.randint(0, n_curr - 1)
        if u == v:
            continue
        u, v = min(u, v), max(u, v)
        
        # Flip edge (u, v)
        if v in adj[u]:
            adj[u].discard(v)
            adj[v].discard(u)
            edges = [e for e in edges if not (e[0] == u and e[1] == v)]
        else:
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
    
    return {'n': n_curr, 'edges': edges}

def _try_all_seeds(n, num_seeds=100):
    """Try multiple random seeds to find a valid graph on n vertices."""
    for seed in range(num_seeds):
        if _elapsed() > DEADLINE:
            break
        result = _greedy_construct(n, seed)
        if result['n'] == n:
            adj = _build_adj_list(result['edges'], n)
            if not _has_violation(n, adj):
                return result
    return None

def _known_20_vertex_graph():
    """Return a known 20-vertex graph with no regular induced subgraph on >= 6 vertices.
    This is based on literature constructions. We'll use a random search to find one.
    """
    # Try to find a valid 20-vertex graph via random search
    for seed in range(500):
        if _elapsed() > DEADLINE:
            break
        n, edges = _random_graph(20, 0.5, seed)
        adj = _build_adj_list(edges, n)
        if not _has_violation(n, adj):
            return {'n': n, 'edges': edges}
    return None

def construct():
    """Construct a graph with no regular induced subgraph on >= 6 vertices.
    Maximize n.
    """
    global _START_TIME, DEADLINE
    _START_TIME = time.time()
    DEADLINE = _START_TIME + 125.0
    
    best_result = None
    
    # First, try to find a 20-vertex graph
    # Use a more systematic approach: try many random graphs
    found_20 = False
    for seed in range(2000):
        if _elapsed() > DEADLINE - 5:
            break
        n, edges = _random_graph(20, 0.5, seed)
        adj = _build_adj_list(edges, n)
        if not _has_violation(n, adj):
            best_result = {'n': 20, 'edges': edges}
            found_20 = True
            break
    
    if found_20:
        # Try to extend to 21 vertices
        if _elapsed() < DEADLINE - 10:
            base_edges = best_result['edges']
            for seed in range(200):
                if _elapsed() > DEADLINE - 5:
                    break
                # Try adding vertex 20 with random connections
                new_edges = list(base_edges)
                rng = random.Random(seed)
                for i in range(20):
                    if rng.random() < 0.5:
                        new_edges.append([20, i])
                adj = _build_adj_list(new_edges, 21)
                if not _has_violation(21, adj):
                    best_result = {'n': 21, 'edges': new_edges}
                    break
    
    if best_result is None:
        # Fallback: find a smaller valid graph
        for target_n in [19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
            if _elapsed() > DEADLINE - 2:
                break
            for seed in range(500):
                if _elapsed() > DEADLINE - 2:
                    break
                n, edges = _random_graph(target_n, 0.5, seed)
                adj = _build_adj_list(edges, n)
                if not _has_violation(n, adj):
                    best_result = {'n': n, 'edges': edges}
                    break
            if best_result is not None:
                break
    
    if best_result is None:
        # Absolute fallback: empty graph on 5 vertices
        best_result = {'n': 5, 'edges': []}
    
    return best_result
