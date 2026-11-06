"""Rewritten program: local-search / simulated-annealing style K4-free graph construction
   to minimize the size of the largest triangle-free induced subgraph on 18 vertices."""

import random
import time
import math

N = 18
# Deadline guard: keep a safety margin before the 120s limit
DEADLINE = 115.0  # seconds


def _edge_set_to_edges(es):
    return [[u, v] for (u, v) in sorted(es)]


def _build_adj(es):
    """Return adjacency as list of sets."""
    adj = [set() for _ in range(N)]
    for u, v in es:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _is_k4_free(es):
    """Check if the edge set forms a K4-free graph."""
    adj = _build_adj(es)
    for i in range(N):
        for j in range(i + 1, N):
            if j in adj[i]:
                common = adj[i] & adj[j]
                for k in common:
                    if k == i or k == j:
                        continue
                    for l in common:
                        if l <= k or l == i or l == j:
                            continue
                        if l in adj[k]:
                            return False
    return True


def _largest_triangle_free_induced(es):
    """
    Find the largest subset of vertices that induces a triangle-free subgraph.
    Uses branch-and-bound over the 2^18 subsets, but with pruning.
    Actually, 2^18 = 262144, which is manageable.
    """
    adj = _build_adj(es)
    best = 0

    # Precompute for each vertex, the set of neighbors
    # We'll enumerate subsets via bitmask
    # For each subset, check if it's triangle-free
    # To speed up, we can use the fact that a set is triangle-free iff
    # for every edge (u,v) in the induced subgraph, u and v have no common neighbor in the set.

    # Alternative: iterate over all subsets and check triangle-free
    # 2^18 = 262144 subsets, each check is O(n^2) = 324, total ~85M ops, might be slow in Python.
    # Let's optimize: precompute for each pair (u,v) whether they are adjacent.
    # A subset S is triangle-free iff there is no triple (u,v,w) in S that forms a triangle.
    # We can check: for each u in S, for each v in S with v>u, if (u,v) is an edge,
    # then no w in S with w>v should be adjacent to both u and v.

    # Let's use a smarter approach:
    # For each subset, represented as bitmask, check if it contains any triangle.
    # Precompute for each vertex u, the bitmask of its neighbors.
    # A triangle (u,v,w) exists in S iff u,v,w in S and (u,v),(v,w),(u,w) all edges.
    # We can iterate over all triangles in the full graph and check if all three vertices are in S.

    # First, find all triangles in the graph.
    triangles = []
    for i in range(N):
        for j in range(i + 1, N):
            if j in adj[i]:
                common = adj[i] & adj[j]
                for k in common:
                    if k > j:
                        triangles.append((i, j, k))

    # Now for each subset, check if it contains any triangle
    # If it does, it's not triangle-free.
    # We want the largest subset with no triangle.

    # Use branch and bound:
    # Order vertices by degree (descending) to prune earlier.
    # Actually, let's just do a simple enumeration with pruning.

    # Better: use the fact that the complement of a triangle-free graph is a graph
    # with no independent set of size 3... not helpful.

    # Let's just enumerate all 2^18 subsets. 262144 is fine.
    # For each subset, check if it's triangle-free by checking all triangles.
    # If no triangle is fully contained in the subset, it's triangle-free.

    # Precompute triangle masks
    tri_masks = []
    for (i, j, k) in triangles:
        mask = (1 << i) | (1 << j) | (1 << k)
        tri_masks.append(mask)

    for mask in range(1 << N):
        if bin(mask).count('1') <= best:
            continue
        # Check if this subset contains any triangle
        is_tf = True
        for tm in tri_masks:
            if (mask & tm) == tm:
                is_tf = False
                break
        if is_tf:
            best = bin(mask).count('1')

    return best


def _initial_graph():
    """Start with a random K4-free graph, e.g., a random tripartite graph."""
    # Random partition into 3 parts
    parts = list(range(N))
    random.shuffle(parts)
    sizes = [6, 6, 6]
    p0 = parts[:6]
    p1 = parts[6:12]
    p2 = parts[12:18]

    es = set()
    # Add all edges between different parts (complete tripartite)
    for a in p0:
        for b in p1:
            es.add((min(a,b), max(a,b)))
    for a in p0:
        for b in p2:
            es.add((min(a,b), max(a,b)))
    for a in p1:
        for b in p2:
            es.add((min(a,b), max(a,b)))

    return es


def _remove_random_edge(es):
    if not es:
        return es
    e = random.choice(list(es))
    es.discard(e)
    return es


def _add_random_edge(es):
    """Try to add a random edge that keeps the graph K4-free."""
    # Pick a random non-edge
    all_possible = [(i, j) for i in range(N) for j in range(i+1, N)]
    non_edges = [e for e in all_possible if e not in es]
    if not non_edges:
        return es
    e = random.choice(non_edges)
    u, v = e
    # Check if adding (u,v) creates a K4
    adj = _build_adj(es)
    common = adj[u] & adj[v]
    # Check if any common neighbor w has another common neighbor x with u and v
    # i.e., is there a w in common such that (u,w) and (v,w) and (w,x) and (u,x) and (v,x) for some x?
    # Actually, adding (u,v) creates a K4 iff there exist two distinct vertices w1, w2 in common
    # such that (w1,w2) is an edge.
    for w1 in common:
        for w2 in common:
            if w1 < w2 and (min(w1,w2), max(w1,w2)) in es:
                return es  # Would create K4
    es.add(e)
    return es


def construct():
    start_time = time.time()
    random.seed(42)

    # Start with a complete tripartite graph
    es = _initial_graph()
    best_score = _largest_triangle_free_induced(es)
    best_es = set(es)

    current_temp = 1.0
    cooling = 0.9995
    min_temp = 0.01

    while time.time() - start_time < DEADLINE:
        # Try a move: add or remove a random edge
        move_type = random.random()
        new_es = set(es)

        if move_type < 0.5:
            new_es = _add_random_edge(new_es)
            if new_es == es:
                continue
        else:
            new_es = _remove_random_edge(new_es)
            if new_es == es:
                continue

        # Check K4-free
        if not _is_k4_free(new_es):
            continue

        new_score = _largest_triangle_free_induced(new_es)

        if new_score < best_score:
            best_score = new_score
            best_es = set(new_es)
            es = new_es
        elif new_score <= best_score:
            # Accept (same or better)
            es = new_es
        else:
            # Simulated annealing: accept with probability exp(-delta/temp)
            delta = new_score - best_score
            if random.random() < math.exp(-delta / current_temp):
                es = new_es

        current_temp = max(min_temp, current_temp * cooling)

        if best_score <= 5:
            break

    return _edge_set_to_edges(best_es)
