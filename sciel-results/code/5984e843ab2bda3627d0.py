"""Program to construct a large triangle-free graph with independence number <= 9."""
import random
from itertools import combinations

def _is_triangle_free(n, adj):
    """Check if graph is triangle-free using adjacency matrix."""
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                for k in range(j + 1, n):
                    if adj[i][k] and adj[j][k]:
                        return False
    return True

def _max_independent_set_size(n, adj):
    """Branch and bound to find the size of the maximum independent set."""
    best = [0]
    # Order vertices by degree (ascending) for better pruning
    order = sorted(range(n), key=lambda v: sum(adj[v]))
    # Map to new order
    new_adj = [[0] * n for _ in range(n)]
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
        for u in range(n):
            if adj[v][u]:
                new_adj[i][pos[u]] = 1
    
    def branch(idx, size, candidates):
        if size + len(candidates) <= best[0]:
            return
        if not candidates:
            if size > best[0]:
                best[0] = size
            return
        # Pick vertex with most conflicts among candidates
        v = max(candidates, key=lambda u: sum(new_adj[u][w] for w in candidates))
        # Branch 1: exclude v
        branch(idx + 1, size, [w for w in candidates if w != v])
        # Branch 2: include v
        new_candidates = [w for w in candidates if w != v and not new_adj[v][w]]
        branch(idx + 1, size + 1, new_candidates)
    
    branch(0, 0, list(range(n)))
    return best[0]

def _all_max_independent_sets(n, adj, max_count=1000):
    """Find all maximum independent sets (up to max_count)."""
    # First find the size
    alpha = _max_independent_set_size(n, adj)
    sets = []
    
    order = sorted(range(n), key=lambda v: sum(adj[v]))
    new_adj = [[0] * n for _ in range(n)]
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
        for u in range(n):
            if adj[v][u]:
                new_adj[i][pos[u]] = 1
    
    current = []
    
    def branch(idx, candidates):
        if len(sets) >= max_count:
            return
        if not candidates:
            if len(current) == alpha:
                sets.append(current[:])
            return
        # Pruning: even if we take all remaining, can't reach alpha
        if len(current) + len(candidates) < alpha:
            return
        v = max(candidates, key=lambda u: sum(new_adj[u][w] for w in candidates))
        # Exclude
        branch(idx + 1, [w for w in candidates if w != v])
        # Include
        if len(current) < alpha:
            current.append(v)
            new_candidates = [w for w in candidates if w != v and not new_adj[v][w]]
            branch(idx + 1, new_candidates)
            current.pop()
    
    branch(0, list(range(n)))
    return sets, alpha

def _find_independent_hitting_set(n, adj, max_sets):
    """Find an independent set S that hits all max independent sets."""
    # Greedy: repeatedly pick a vertex that is in the most uncovered max independent sets
    # and is not adjacent to already chosen vertices.
    uncovered = [set(s) for s in max_sets]
    if not uncovered:
        return []
    
    chosen = []
    remaining = list(uncovered)
    
    while remaining:
        # Find vertex that covers most remaining sets and is independent with chosen
        best_v = None
        best_count = -1
        for v in range(n):
            # Check if v is independent with all chosen
            if any(adj[v][c] for c in chosen):
                continue
            count = sum(1 for s in remaining if v in s)
            if count > best_count:
                best_count = count
                best_v = v
        if best_v is None or best_count <= 0:
            return None
        chosen.append(best_v)
        remaining = [s for s in remaining if best_v not in s]
    
    return chosen

def _add_vertex(n, adj, max_sets):
    """Try to add a new vertex. Returns new adj or None."""
    # Find independent hitting set
    S = _find_independent_hitting_set(n, adj, max_sets)
    if S is None or not S:
        return None
    
    # Create new graph with n+1 vertices
    new_n = n + 1
    new_adj = [[0] * new_n for _ in range(new_n)]
    for i in range(n):
        for j in range(n):
            new_adj[i][j] = adj[i][j]
    for v in S:
        new_adj[n][v] = 1
        new_adj[v][n] = 1
    
    # Verify triangle-free
    if not _is_triangle_free(new_n, new_adj):
        return None
    
    return new_adj

def construct():
    # Start with Andrasfai graph A_9: 26 vertices
    # i ~ j iff (j - i) mod 26 is 1 mod 3
    n0 = 26
    adj = [[0] * n0 for _ in range(n0)]
    for i in range(n0):
        for j in range(i + 1, n0):
            if (j - i) % n0 % 3 == 1:
                adj[i][j] = 1
                adj[j][i] = 1
    
    # Verify starting graph
    assert _is_triangle_free(n0, adj), "A_9 should be triangle-free"
    alpha0 = _max_independent_set_size(n0, adj)
    assert alpha0 == 9, f"A_9 should have alpha=9, got {alpha0}"
    
    # Try to extend
    current_n = n0
    current_adj = adj
    max_extensions = 15
    
    for _ in range(max_extensions):
        # Get all max independent sets
        max_sets, alpha = _all_max_independent_sets(current_n, current_adj, max_count=500)
        if alpha > 9:
            break
        
        new_adj = _add_vertex(current_n, current_adj, max_sets)
        if new_adj is None:
            break
        
        # Verify new graph
        new_n = current_n + 1
        if not _is_triangle_free(new_n, new_adj):
            break
        new_alpha = _max_independent_set_size(new_n, new_adj)
        if new_alpha > 9:
            break
        
        current_n = new_n
        current_adj = new_adj
    
    # Build edge list
    edges = []
    for i in range(current_n):
        for j in range(i + 1, current_n):
            if current_adj[i][j]:
                edges.append([i, j])
    
    return {'n': current_n, 'edges': edges}
