"""Triangle-free diameter-four augmentation on 24-vertex path."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def construct():
    start_time = time.time()
    deadline = 75.0  # safety margin before 90s kill

    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((min(u, v), max(u, v)))

    # Precompute: which pairs are "forbidden" for triangle-free
    # Two edges (a,b) and (c,d) conflict (form triangle) if they share a vertex
    # and the third pair exists. Actually, adding edge (u,v) creates a triangle
    # if there exists w such that (u,w) and (w,v) are both in the graph.

    # Build adjacency from base
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)

    # An edge (u,v) is "triangle-free safe" with respect to current graph
    # if there is no w with w in adj[u] and w in adj[v].

    def is_edge_safe(u, v, adj, base_edges):
        """Check if adding (u,v) keeps graph triangle-free."""
        if (min(u, v), max(u, v)) in base_edges:
            return False
        # Triangle-free check: no common neighbor
        common = adj[u] & adj[v]
        return len(common) == 0

    def compute_diameter(adj, n):
        """BFS from each node, return max shortest path distance."""
        max_diam = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            idx = 0
            while idx < len(queue):
                u = queue[idx]
                idx += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            # Check connectivity
            for i in range(n):
                if dist[i] == -1:
                    return -1  # not connected
            max_diam = max(max_diam, max(dist))
        return max_diam

    def is_valid_solution(added_edges, base_edges, n):
        """Check all constraints: no dup, triangle-free, connected, diameter <= 4."""
        adj = [set() for _ in range(n)]
        all_edges = set(base_edges)
        for u, v in added_edges:
            e = (min(u, v), max(u, v))
            if e in all_edges:
                return False  # duplicate
            all_edges.add(e)
            adj[u].add(v)
            adj[v].add(u)
        # Check triangle-free
        for u in range(n):
            for v in adj[u]:
                for w in adj[u]:
                    if w > v and w in adj[v]:
                        return False
        # Check diameter
        diam = compute_diameter(adj, n)
        if diam == -1:
            return False
        if diam > 4:
            return False
        return True

    # Strategy: Greedy + local search (swap-based)
    # First, find a feasible solution via greedy construction
    # Then improve via local search

    # Greedy: try to add edges to reduce diameter
    # Initial: just the path. Diameter is 23. We need to add edges.

    # Better approach: work with the set of added edges
    # Start with a known feasible solution or build one

    # Let's think about what edges to add.
    # The path has 24 vertices. We need diameter <= 4.
    # In a path of 24 vertices, to get diameter 4, we need significant shortcuts.
    # Each added edge reduces distances.

    # Greedy approach: repeatedly add the edge that most reduces the diameter
    # while maintaining triangle-freeness.

    def greedy_construct(n, base_edges, deadline):
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        added = []
        current_diam = compute_diameter(adj, n)
        if current_diam == -1:
            return None

        # Candidate edges: all non-adjacent pairs that are safe
        while time.time() < deadline and current_diam > 4:
            best_edge = None
            best_diam = current_diam
            # Try all possible edges
            for u in range(n):
                for v in range(u + 1, n):
                    e = (u, v)
                    if e in base_edges:
                        continue
                    # Check if already added
                    if any((min(a, b), max(a, b)) == e for a, b in added):
                        continue
                    # Check triangle-free safety
                    if not is_edge_safe(u, v, adj, base_edges):
                        continue
                    # Tentatively add and check diameter
                    adj[u].add(v)
                    adj[v].add(u)
                    new_diam = compute_diameter(adj, n)
                    adj[u].remove(v)
                    adj[v].remove(u)
                    if new_diam == -1:
                        continue
                    if new_diam < best_diam:
                        best_diam = new_diam
                        best_edge = e
            if best_edge is None:
                break
            u, v = best_edge
            adj[u].add(v)
            adj[v].add(u)
            added.append([u, v])
            current_diam = best_diam

        if current_diam <= 4:
            return added
        return None

    # Try greedy
    result = greedy_construct(n, base_edges, deadline)

    if result is None:
        # Fallback: try a different ordering or more aggressive approach
        # For now, return empty (invalid) rather than crash
        return []

    # Local search: try to remove edges or swap to reduce count
    # Keep trying until deadline
    best_solution = result[:]
    best_len = len(best_solution)

    def try_remove(added, base_edges, n, deadline):
        """Try removing each edge and check if still valid."""
        for i in range(len(added)):
            if time.time() >= deadline:
                break
            candidate = added[:i] + added[i+1:]
            if is_valid_solution(candidate, base_edges, n):
                return candidate
        return None

    def try_swap(added, base_edges, n, deadline):
        """Try replacing one edge with another to reduce count (via removal first)."""
        # Actually swap doesn't reduce count directly.
        # Let's do: try removing one edge, then if invalid, try adding a different
        # edge to fix it while keeping count same or lower.
        # Simpler: just do removal-based pruning first, then swap to improve.
        for i in range(len(added)):
            if time.time() >= deadline:
                break
            removed = added[i]
            candidate = added[:i] + added[i+1:]
            # Check if valid without the removed edge
            if is_valid_solution(candidate, base_edges, n):
                return candidate
        return None

    # Iterative improvement
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removal
        new_sol = try_remove(best_solution, base_edges, n, deadline)
        if new_sol is not None:
            best_solution = new_sol
            best_len = len(best_solution)
            improved = True
            continue

        # Try swap: replace one edge with another, hoping to enable further removal
        # Or: try all single-edge replacements
        for i in range(len(best_solution)):
            if time.time() >= deadline:
                break
            removed = best_solution[i]
            rest = best_solution[:i] + best_solution[i+1:]
            # Try adding different edges to 'rest' to see if we get a valid solution
            # with <= len(best_solution) edges
            # This is expensive; limit it
            # For now, skip swap and just do removal
            pass

    # Final validation
    if not is_valid_solution(best_solution, base_edges, n):
        # Fallback: try to construct a valid solution from scratch
        # Use a hardcoded approach: add specific edges
        # For a 24-vertex path, we can add edges to create a structure
        # with small diameter.
        # E.g., add edges (0,12), (1,13), (2,14), ..., to create chords
        # But must be triangle-free.
        
        # Let's try a structured approach:
        # Add edges that connect distant vertices without creating triangles.
        # In a path, adding edge (u,v) is triangle-free iff no vertex is adjacent
        # to both u and v. In the path, only u-1, u+1 are adjacent to u, etc.
        # So (u,v) is triangle-free in the path if |u-v| > 2 (not already an edge)
        # and the path structure doesn't create triangles. Actually in a pure path,
        # any non-adjacent edge is triangle-free initially. But as we add more edges,
        # we need to check.
        
        # Just return the greedy result if it was valid, else empty
        return []

    # Convert to proper types
    result_list = [[int(u), int(v)] for u, v in best_solution]
    return result_list


# EVOLVE-BLOCK-END
