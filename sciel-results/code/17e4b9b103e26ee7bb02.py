"""K2,3-free dense graph on 20 vertices via simulated annealing on edge bitmask."""
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    n = 20
    m = n * (n - 1) // 2  # 190 possible edges

    # Precompute edge index mapping
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_idx[(u, v)] = idx
            idx_to_edge.append((u, v))
            idx += 1

    # Precompute for each edge, the list of other edges that would be affected
    # When we add/remove edge (u,v), the common-neighbor counts change for:
    #   - pair (u,v) itself
    #   - pairs (u,w) for w in N(v) \ {u}
    #   - pairs (v,w) for w in N(u) \ {v}
    #
    # We'll use a different approach: maintain a "common neighbor count" matrix
    # and track violations.

    # For fast feasibility checking, precompute for each pair of vertices (i,j)
    # the list of edges that connect i to some common neighbor of j, etc.
    # Actually, let's use a direct approach with bitmasks.

    # Represent adjacency as a list of bitmasks
    # adj[i] is a bitmask of neighbors of vertex i

    def popcount(x):
        return bin(x).count('1')

    # For each pair of vertices (u,v), u<v, define:
    # common(u,v) = popcount(adj[u] & adj[v])
    # The constraint is: for all pairs (u,v), common(u,v) <= 2

    # We'll track a "violation count" for each pair and total violations.

    # Precompute: for each edge e=(u,v), which pairs' common-neighbor counts are affected?
    # Adding edge (u,v):
    #   - common(u,v) increases by 0 (u and v are now adjacent, but common neighbors are shared neighbors)
    #     Wait, common(u,v) = |N(u) ∩ N(v)|. Adding edge (u,v) means v ∈ N(u) and u ∈ N(v).
    #     So v becomes a common neighbor of u and every w ∈ N(v) (w≠u), and u becomes a common neighbor of v and every w ∈ N(u) (w≠v).
    #     Also, common(u,v) itself: v is in N(u), u is in N(v), but common(u,v) counts common neighbors, not the vertices themselves.
    #     So common(u,v) doesn't change directly from adding the edge, but:
    #     For each w ∈ N(v), w≠u: common(u,w) increases by 1 (because v is now a common neighbor of u and w)
    #     For each w ∈ N(u), w≠v: common(v,w) increases by 1 (because u is now a common neighbor of v and w)
    #     And common(u,v) itself: does it change? N(u) gains v, N(v) gains u. 
    #     common(u,v) = |N(u) ∩ N(v)|. Before: N(u) and N(v) didn't include each other.
    #     After: N(u) includes v, N(v) includes u. v ∈ N(u) but is v ∈ N(v)? No (simple graph). u ∈ N(v) but is u ∈ N(u)? No.
    #     So common(u,v) is unchanged by adding edge (u,v).

    # So the affected pairs when adding edge (u,v) are:
    #   (u,w) for w in N(v), w != u
    #   (v,w) for w in N(u), w != v
    #   And also (u,v) itself? No, as argued above.
    #   Wait, I need to double-check. Let me reconsider.
    #   
    #   Actually, I realize the common neighbor count for pair (u,v) is |N(u) ∩ N(v)|.
    #   When we add edge (u,v):
    #   - N(u) gains v, N(v) gains u
    #   - For pair (u,v): new common = |(N(u)∪{v}) ∩ (N(v)∪{u})| = |N(u)∩N(v) ∪ {v if v∈N(v)} ∪ {u if u∈N(u)} ∪ {u,v if u∈N(v) and v∈N(u)}|
    #   - Since it's a simple graph, v∉N(v) and u∉N(u). So common(u,v) is unchanged.
    #   - For pair (u,w) where w∈N(v), w≠u: v is now in N(u) and was already in N(v) (since w∈N(v) means v∈N(w) and w∈N(v), but we need v∈N(v)? No.)
    #   
    #   Let me be more careful. w ∈ N(v) means v and w are adjacent. 
    #   common(u,w) = |N(u) ∩ N(w)|. 
    #   Before adding (u,v): v ∉ N(u) (since we're adding it).
    #   After adding (u,v): v ∈ N(u). Is v ∈ N(w)? Yes, since w ∈ N(v) means v-w is an edge, so v ∈ N(w).
    #   So v becomes a new common neighbor of u and w. Thus common(u,w) increases by 1.
    #   
    #   Similarly for pair (v,w) where w∈N(u), w≠v: u becomes a new common neighbor.

    # So the update rule is clear. Let's implement simulated annealing.

    def make_initial_adj():
        """Create a random initial valid graph."""
        adj = [0] * n
        edges = list(range(m))
        random.shuffle(edges)
        for e in edges:
            u, v = idx_to_edge[e]
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def can_add(adj, u, v):
        """Check if adding edge (u,v) keeps K2,3-free."""
        if (adj[u] >> v) & 1:
            return True
        # Check pair (u,v)
        if popcount(adj[u] & adj[v]) > 2:
            return False
        # Check pairs (u,w) for w in N(v)
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj[v] >> w) & 1:
                if popcount(adj[u] & adj[w]) > 1:
                    return False
        # Check pairs (v,w) for w in N(u)
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj[u] >> w) & 1:
                if popcount(adj[v] & adj[w]) > 1:
                    return False
        return True

    def can_remove(adj, u, v):
        """Check if removing edge (u,v) keeps K2,3-free (always true, but we need to verify)."""
        # Removing an edge can only decrease common neighbor counts, so it's always valid.
        return True

    def edge_count(adj):
        count = 0
        for u in range(n):
            count += popcount(adj[u])
        return count // 2

    def is_valid(adj):
        """Verify the graph is K2,3-free."""
        for u in range(n):
            for v in range(u + 1, n):
                if popcount(adj[u] & adj[v]) > 2:
                    return False
        return True

    def simulate_annealing(adj, start_temp, end_temp, max_iters, time_limit):
        """Simulated annealing to maximize edges while maintaining K2,3-free."""
        current_adj = adj[:]
        current_edges = edge_count(current_adj)
        best_adj = current_adj[:]
        best_edges = current_edges

        # Precompute for each edge, the list of vertices in N(u) and N(v) for fast updates
        # We'll recompute these as needed

        for iteration in range(max_iters):
            if time.time() > time_limit:
                break

            # Choose a random edge to flip
            e = random.randint(0, m - 1)
            u, v = idx_to_edge[e]

            if (current_adj[u] >> v) & 1:
                # Edge is present, try to remove it
                # Always valid to remove
                new_adj = current_adj[:]
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
                new_edges = current_edges - 1
                delta = new_edges - current_edges  # -1
                # Always accept removal in SA (or with probability)
                if delta >= 0 or random.random() < math.exp(delta / start_temp * (1 - iteration / max_iters) + 0.001):
                    current_adj = new_adj
                    current_edges = new_edges
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
            else:
                # Edge is absent, try to add it
                if can_add(current_adj, u, v):
                    new_adj = current_adj[:]
                    new_adj[u] |= (1 << v)
                    new_adj[v] |= (1 << u)
                    new_edges = current_edges + 1
                    delta = new_edges - current_edges  # +1
                    # Always accept if valid
                    current_adj = new_adj
                    current_edges = new_edges
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]

        return best_adj, best_edges

    # Main: try multiple random starts with simulated annealing
    random.seed(42)

    best_adj = [0] * n
    best_edges = 0

    num_starts = 50
    for s in range(num_starts):
        if time.time() > deadline:
            break

        adj = make_initial_adj()
        edges = edge_count(adj)

        # Simulated annealing
        adj, edges = simulate_annealing(adj, 1.0, 0.01, 50000, deadline)

        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]

    # Final polish: try to add any remaining edges greedily
    for u in range(n):
        for v in range(u + 1, n):
            if time.time() > deadline:
                break
            if not ((best_adj[u] >> v) & 1):
                if can_add(best_adj, u, v):
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
                    best_edges += 1

    # Verify
    if not is_valid(best_adj):
        # Fallback
        adj = make_initial_adj()
        best_adj = adj
        best_edges = edge_count(adj)

    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])

    return edges
# EVOLVE-BLOCK-END
