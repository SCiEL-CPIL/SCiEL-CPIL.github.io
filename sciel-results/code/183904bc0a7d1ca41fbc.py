"""Search for dense triangle-free graphs on 16 vertices that are not 3-colourable."""
import time
import random
import sys

# Deadline guard
_DEADLINE = 150.0  # seconds; leave margin for evaluator overhead


def _time_left():
    return _DEADLINE - time.time()


def _is_triangle_free(adj, n):
    """Check if graph is triangle-free given adjacency matrix."""
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                for k in range(j + 1, n):
                    if adj[i][k] and adj[j][k]:
                        return False
    return True


def _is_3_colorable(adj, n, time_budget=5.0):
    """Check if graph is 3-colorable using backtracking with time limit.
    Returns True if 3-colorable, False if not (or if timeout, we assume not colorable conservatively
    by trying harder first). Actually returns (colorable: bool, found: bool).
    We want graphs that are NOT 3-colorable, so we return True if we can prove it's not 3-colorable.
    """
    start = time.time()
    colors = [-1] * n

    # Order vertices by degree descending for better pruning
    order = list(range(n))
    order.sort(key=lambda v: sum(adj[v]), reverse=True)
    pos = {v: i for i, v in enumerate(order)}

    def backtrack(idx):
        if time.time() - start > time_budget:
            return None  # timeout
        if idx == n:
            return True
        v = order[idx]
        for c in range(3):
            ok = True
            for u in range(n):
                if u == v:
                    continue
                if adj[v][u] and colors[u] == c:
                    ok = False
                    break
            if ok:
                colors[v] = c
                result = backtrack(idx + 1)
                if result is not None:
                    return result
                colors[v] = -1
        return False

    result = backtrack(0)
    if result is None:
        # Timeout - uncertain. Try with more time or use a different method.
        return None  # uncertain
    return not result  # True if NOT 3-colorable


def _edge_count(adj, n):
    return sum(sum(row) for row in adj) // 2


def _build_adj(edges, n):
    adj = [[0] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
    return adj


def _random_maximal_triangle_free(n, rng):
    """Generate a random maximal triangle-free graph by random edge addition."""
    adj = [[0] * n for _ in range(n)]
    # Start with empty graph, add edges in random order if they don't create triangles
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_edges)
    for u, v in all_edges:
        # Check if adding (u,v) creates a triangle
        creates_tri = False
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                creates_tri = True
                break
        if not creates_tri:
            adj[u][v] = 1
            adj[v][u] = 1
    return adj


def _local_search(adj, n, rng, time_limit):
    """Improve a triangle-free graph by trying to add edges while maintaining triangle-free and non-3-colorable."""
    start = time.time()
    best_edges = _edge_count(adj, n)
    best_adj = [row[:] for row in adj]
    
    # We want to maximize edges while keeping triangle-free and not 3-colorable
    # Strategy: try to add random edges that keep triangle-free, then check 3-colorability
    # If adding an edge makes it 3-colorable, we can't add it (or we need to also remove some edge)
    
    # Actually, the constraint is: triangle-free AND not 3-colorable.
    # Adding edges preserves triangle-free only if no triangle is created.
    # Adding edges can make a non-3-colorable graph become... well, adding edges can't make a non-3-colorable graph become 3-colorable!
    # Wait: if G is not 3-colorable, then any supergraph of G is also not 3-colorable (since a 3-coloring of the supergraph restricts to a 3-coloring of G).
    # So we just need to maximize edges while keeping triangle-free!
    # The non-3-colorable constraint is automatically satisfied if we start from a non-3-colorable graph and only add edges.
    
    # So the problem reduces to: find the densest triangle-free graph on 16 vertices that is not 3-colorable.
    # But we need to ensure it's not 3-colorable. Since adding edges preserves non-3-colorability,
    # we should start from a known non-3-colorable triangle-free graph and add as many edges as possible.
    
    # Current approach: start with adj, try adding every possible edge that maintains triangle-free,
    # and keep the densest result. Since non-3-colorability is preserved, we just need triangle-free.
    
    # Actually, we need to be careful: we want the FINAL graph to be not 3-colorable.
    # If we start from a not-3-colorable graph and add edges, it stays not-3-colorable.
    # So the optimal strategy is: find the maximum triangle-free graph (max edges) that contains
    # a non-3-colorable subgraph.
    
    # Simple approach: just find the maximum triangle-free graph on 16 vertices, and check if it's 3-colorable.
    # The Clebsch graph on 16 vertices has 40 edges and is triangle-free. Is it 3-colorable?
    # The Clebsch graph has chromatic number 4, so it's not 3-colorable!
    
    # Let me just return the densest triangle-free graph I can find.
    
    # Greedy: try to add every missing edge
    for u in range(n):
        for v in range(u + 1, n):
            if time.time() - start > time_limit:
                break
            if adj[u][v]:
                continue
            # Check if adding (u,v) creates a triangle
            creates_tri = False
            for w in range(n):
                if w != u and w != v and adj[u][w] and adj[v][w]:
                    creates_tri = True
                    break
            if not creates_tri:
                adj[u][v] = 1
                adj[v][u] = 1
                best_edges = _edge_count(adj, n)
    
    # Now do local search: try swapping edges
    # Try removing one edge and adding another to increase edge count (shouldn't happen if we already added all possible)
    # Actually, the above greedy should give a maximal triangle-free graph. Let's verify and return.
    
    return adj


def _clebsch_graph():
    """Construct the Clebsch graph on 16 vertices.
    Vertices are binary strings of length 4. Two vertices are adjacent if their Hamming distance is 2.
    This is a triangle-free graph with 40 edges and chromatic number 4.
    """
    n = 16
    adj = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            # Compute Hamming distance
            dist = bin(i ^ j).count('1')
            if dist == 2:
                adj[i][j] = 1
                adj[j][i] = 1
    return adj


def _verify_clebsch(adj, n):
    """Verify Clebsch graph properties."""
    assert _is_triangle_free(adj, n), "Clebsch graph should be triangle-free"
    count = _edge_count(adj, n)
    print(f"Clebsch graph: {count} edges", file=sys.stderr)
    # Check 3-colorability
    result = _is_3_colorable(adj, n, time_budget=10.0)
    if result is not None:
        print(f"Clebsch is 3-colorable: {not result}", file=sys.stderr)
        assert result, "Clebsch graph should NOT be 3-colorable"
    return count


def construct():
    """Construct a dense triangle-free graph on 16 vertices that is not 3-colourable."""
    n = 16
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second margin
    
    # Start with the Clebsch graph (40 edges, triangle-free, chromatic number 4)
    adj = _clebsch_graph()
    best_adj = [row[:] for row in adj]
    best_edges = _edge_count(adj, n)
    print(f"Initial Clebsch: {best_edges} edges", file=sys.stderr)
    
    # The Clebsch graph is already maximal triangle-free on 16 vertices?
    # Let's check: can we add more edges?
    # Try to add edges
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                continue
            creates_tri = False
            for w in range(n):
                if w != u and w != v and adj[u][w] and adj[v][w]:
                    creates_tri = True
                    break
            if not creates_tri:
                adj[u][v] = 1
                adj[v][u] = 1
                print(f"Added edge ({u},{v})", file=sys.stderr)
    
    edges_now = _edge_count(adj, n)
    print(f"After greedy add: {edges_now} edges", file=sys.stderr)
    
    # Now try local search: perturb and re-optimize
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Create a perturbed version
        trial = [row[:] for row in best_adj]
        # Randomly remove a few edges
        for _ in range(rng.randint(1, 5)):
            u, v = rng.randint(0, n - 1), rng.randint(0, n - 1)
            if u < v and trial[u][v]:
                trial[u][v] = 0
                trial[v][u] = 0
        # Re-add all possible edges
        for i in range(n):
            for j in range(i + 1, n):
                if trial[i][j]:
                    continue
                creates_tri = False
                for w in range(n):
                    if w != i and w != j and trial[i][w] and trial[j][w]:
                        creates_tri = True
                        break
                if not creates_tri:
                    trial[i][j] = 1
                    trial[j][i] = 1
        
        trial_edges = _edge_count(trial, n)
        if trial_edges > best_edges:
            best_adj = [row[:] for row in trial]
            best_edges = trial_edges
            print(f"Improved to {best_edges} edges", file=sys.stderr)
    
    # Also try starting from random maximal triangle-free graphs
    for seed in range(100):
        if time.time() >= deadline:
            break
        trial = _random_maximal_triangle_free(n, random.Random(seed))
        trial_edges = _edge_count(trial, n)
        if trial_edges > best_edges:
            # Check if it's not 3-colorable
            is_not_3col = _is_3_colorable(trial, n, time_budget=2.0)
            if is_not_3col is True:  # confirmed not 3-colorable
                best_adj = [row[:] for row in trial]
                best_edges = trial_edges
                print(f"New best from random: {best_edges} edges", file=sys.stderr)
            elif is_not_3col is None:
                # Uncertain, skip for safety
                pass
    
    # Convert back to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u][v]:
                edges.append([u, v])
    
    return edges
