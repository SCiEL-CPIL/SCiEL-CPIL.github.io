"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {
    'n': 24,
    'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
              [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
              [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]],
    'diameter': 4
}

def construct():
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((min(u, v), max(u, v)))

    # The base graph is a path 0-1-2-...-23.
    # We need to add edges to make diameter <= 4 while keeping it triangle-free.
    #
    # Strategy:
    # 1. The path has diameter 23. We need to reduce it to 4.
    # 2. Triangle-free constraint: no three vertices can form a cycle of length 3.
    #    In a bipartite graph (which a path is), adding edges only between vertices
    #    of opposite parity preserves bipartiteness, and thus triangle-freeness.
    #    Vertices in the path: even indices {0,2,4,...,22} and odd indices {1,3,...,23}.
    #    So we can only add edges between even and odd vertices.
    #
    # 3. To minimize edges while achieving diameter 4, we want to create "shortcuts".
    #    A known optimal structure for a path of length 23 (24 vertices) with diameter 4
    #    and triangle-free (bipartite) constraint is to use a "ladder" or "grid" like structure,
    #    but since we start with a path, we just add chords.
    #
    # Let's think about the minimum number of edges.
    # If we add edges to create a structure where every vertex is within distance 2 of a "hub",
    # and hubs are within distance 2 of each other, we get diameter 4.
    #
    # A simple approach:
    # Add edges (0, 22), (1, 21), (2, 20), ... but we must check triangle-free.
    # Actually, let's just use a greedy or constructive approach.
    #
    # Consider adding edges that connect far apart vertices of opposite parity.
    # For example, a "star" like structure around a center won't work because it creates triangles
    # if not careful, but since we're bipartite, we can have a complete bipartite subgraph.
    #
    # Let's try to find a small set of edges.
    # The path is 0-1-2-3-...-23.
    # If we add edges (0, 4), (1, 5), (2, 6), ..., we reduce the effective length.
    #
    # Let's use a known result or constructive method:
    # To get diameter 4 on a path of 24 vertices with bipartite constraint,
    # we can add edges to make it a "4-regular" bipartite graph or similar.
    #
    # Actually, let's just try a specific set and verify.
    # A good candidate is to add edges that jump by roughly 1/5 of the path length.
    # Path length 23. 23/5 ~ 4.6.
    # Let's try adding edges (i, i+5) for i in 0..18, but only if opposite parity.
    # i and i+5 have opposite parity.
    # Edges: (0,5), (1,6), (2,7), (3,8), (4,9), (5,10), (6,11), (7,12), (8,13), (9,14),
    #        (10,15), (11,16), (12,17), (13,18), (14,19), (15,20), (16,21), (17,22), (18,23)
    # That's 19 edges. Diameter might still be too large.
    #
    # Let's try a more aggressive shortcut.
    # Add edges (0, 22), (1, 21), (2, 20), (3, 19), (4, 18), (5, 17), (6, 16), (7, 15),
    # (8, 14), (9, 13), (10, 12).
    # Check parity:
    # (0,22): even-even -> INVALID (creates triangle? No, but not bipartite edge).
    # Wait, the base path is bipartite. Adding an edge between same-parity vertices
    # creates an odd cycle (triangle or longer), violating triangle-free?
    # Actually, adding an edge between same-parity vertices in a bipartite graph creates an odd cycle.
    # The shortest such cycle would be a triangle if the vertices are close, or longer otherwise.
    # The constraint is "triangle-free", not "bipartite".
    # However, any odd cycle contains a triangle? No. A 5-cycle is triangle-free but not bipartite.
    # So we CAN add same-parity edges as long as they don't create a triangle.
    # But it's safer and easier to stick to bipartite edges (opposite parity) to guarantee triangle-free.
    #
    # So, only add edges (u, v) where u and v have opposite parity.
    #
    # Let's try to minimize edges.
    # We can use a BFS-based heuristic:
    # Start with the path. While diameter > 4, find the pair of vertices with maximum distance,
    # and add an edge that reduces the diameter the most, subject to constraints.
    #
    # However, for a fixed small graph (24 vertices), we can hardcode a known optimal or near-optimal solution.
    #
    # Let's try this set of edges (all opposite parity):
    # (0, 22) -> 0 even, 22 even -> SAME PARITY. Invalid for bipartite.
    # (0, 23) -> 0 even, 23 odd -> OK.
    # (1, 22) -> 1 odd, 22 even -> OK.
    # (2, 21) -> 2 even, 21 odd -> OK.
    # (3, 20) -> 3 odd, 20 even -> OK.
    # (4, 19) -> 4 even, 19 odd -> OK.
    # (5, 18) -> 5 odd, 18 even -> OK.
    # (6, 17) -> 6 even, 17 odd -> OK.
    # (7, 16) -> 7 odd, 16 even -> OK.
    # (8, 15) -> 8 even, 15 odd -> OK.
    # (9, 14) -> 9 odd, 14 even -> OK.
    # (10, 13) -> 10 even, 13 odd -> OK.
    # (11, 12) -> 11 odd, 12 even -> OK. But (11,12) is already in the path! Duplicate.
    #
    # So we can't add (11,12).
    #
    # Let's refine. We want to connect the ends to the middle.
    # Edges:
    # (0, 23)
    # (1, 22)
    # (2, 21)
    # (3, 20)
    # (4, 19)
    # (5, 18)
    # (6, 17)
    # (7, 16)
    # (8, 15)
    # (9, 14)
    # (10, 13)
    #
    # Let's check the diameter of this graph.
    # This creates a "cycle" 0-1-2-...-23-0 plus chords.
    # Actually, it's a cycle of length 24 plus chords (i, 23-i) for i=0..10.
    # The distance between any two nodes should be small.
    #
    # Let's verify with a quick BFS in our head or code.
    # For now, let's use a greedy algorithm in the code to be safe, with a time limit.
    #
    # Greedy approach:
    # 1. Start with base graph.
    # 2. Compute all-pairs shortest paths.
    # 3. Find max distance. If <= 4, done.
    # 4. Else, find a pair (u, v) with max distance.
    # 5. Try adding edges (u, v) if valid (not in base, not duplicate, triangle-free).
    #    But adding (u,v) might not be the best.
    #    Better: for each candidate edge (a, b) that is valid, compute new diameter.
    #    Pick the one that minimizes the new diameter.
    # 6. Repeat.
    #
    # Since n=24, the number of possible edges is 24*23/2 = 276.
    # We can afford to try all candidates at each step.
    #
    # Validity check for adding edge (u, v):
    # - (u, v) not in current graph.
    # - No triangle: for all w, not (w adjacent to u AND w adjacent to v).
    #
    # Let's implement this.

    def is_triangle_free(graph, u, v):
        # Check if adding (u, v) creates a triangle
        neighbors_u = set(graph.get(u, []))
        neighbors_v = set(graph.get(v, []))
        common = neighbors_u.intersection(neighbors_v)
        return len(common) == 0

    def compute_diameter(graph, n):
        # BFS from each node
        max_dist = 0
        for start in range(n):
            visited = [-1] * n
            visited[start] = 0
            queue = deque([start])
            while queue:
                curr = queue.popleft()
                for neighbor in graph.get(curr, []):
                    if visited[neighbor] == -1:
                        visited[neighbor] = visited[curr] + 1
                        max_dist = max(max_dist, visited[neighbor])
                        queue.append(neighbor)
        return max_dist

    def greedy_construct(n, base_edges):
        # Initialize graph as adjacency list
        graph = {i: [] for i in range(n)}
        for u, v in base_edges:
            graph[u].append(v)
            graph[v].append(u)
            
        added_edges = []
        
        while True:
            diam = compute_diameter(graph, n)
            if diam <= 4:
                break
            
            # Find best edge to add
            best_edge = None
            best_diam = diam
            
            # Candidate edges: all pairs (u, v) not in graph
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) in base_edges or (u, v) in added_edges:
                        continue
                    # Check if already in graph (from previous additions)
                    if v in graph[u]:
                        continue
                    
                    # Check triangle-free
                    if not is_triangle_free(graph, u, v):
                        continue
                    
                    # Tentatively add edge
                    graph[u].append(v)
                    graph[v].append(u)
                    
                    new_diam = compute_diameter(graph, n)
                    
                    # Remove edge
                    graph[u].remove(v)
                    graph[v].remove(u)
                    
                    if new_diam < best_diam:
                        best_diam = new_diam
                        best_edge = (u, v)
            
            if best_edge is None:
                # No valid edge can reduce diameter
                break
            
            u, v = best_edge
            graph[u].append(v)
            graph[v].append(u)
            added_edges.append((u, v))
            
            # Safety: limit iterations
            if len(added_edges) > 50:
                break
        
        return added_edges

    added = greedy_construct(n, base_edges)
    
    # Convert to list of lists
    return [[u, v] for u, v in added]

# EVOLVE-BLOCK-END
