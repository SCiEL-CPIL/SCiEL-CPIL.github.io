import time
import random
import sys

# Set recursion limit for backtracking if needed
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Helper to check if adding edge (u, v) creates a triangle
    def creates_triangle(adj, u, v):
        # Check common neighbors
        # adj is a list of sets or bitmasks for efficiency
        # Using bitmasks for speed
        # n=16, so 16-bit integers
        common = adj[u] & adj[v]
        return common != 0

    def count_edges(adj):
        return sum(bin(x).count('1') for x in adj) // 2

    def is_3_colorable(adj):
        # Backtracking search for 3-coloring
        n = 16
        color = [-1] * n
        
        def backtrack(vertex):
            if vertex == n:
                return True
            for c in range(3):
                if all(color[neighbor] != c for neighbor in range(n) if (adj[neighbor] >> vertex) & 1):
                    color[vertex] = c
                    if backtrack(vertex + 1):
                        return True
                    color[vertex] = -1
            return False
        
        return backtrack(0)

    def greedy_add_edges(adj, deadline):
        # Try to add edges greedily
        n = 16
        # Order edges by some heuristic? Just iterate
        # To make it deterministic but varied, we can shuffle the edge list
        edges_to_consider = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Randomize order to find different local optima
        random.shuffle(edges_to_consider)
        
        for u, v in edges_to_consider:
            if time.time() > deadline:
                break
            if not ((adj[u] >> v) & 1):
                if not creates_triangle(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    # Seed 1: Grötzsch Graph
    # Vertices 0-4: Cycle
    # Vertices 5-9: Shadows
    # Vertex 10: Top
    grötzsch_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (0, 5), (1, 5), (1, 6), (2, 6), (2, 7), (3, 7), (3, 8), (4, 8), (4, 9), (0, 9),
        (5, 10), (6, 10), (7, 10), (8, 10), (9, 10)
    ]
    
    # Seed 2: Chvátal Graph (12 vertices)
    # The Chvátal graph is the Mycielski of the 5-cycle? No.
    # It is a 12-vertex graph.
    # Let's just use the Grötzsch graph and add 5 isolated vertices, then greedy.
    # And maybe try a different seed: Two disjoint C5s? No, chi=3.
    # We need a seed with chi >= 4.
    
    best_adj = None
    best_edge_count = 0
    
    # Run multiple restarts
    iterations = 0
    while time.time() < deadline:
        # Initialize with Grötzsch graph
        adj = [0] * 16
        for u, v in grötzsch_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Greedy add
        adj = greedy_add_edges(adj, deadline)
        
        edge_count = count_edges(adj)
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_adj = adj[:]
        
        iterations += 1
        # If we've done many iterations, maybe break or slow down
        if iterations > 10000:
            break
            
    if best_adj is None:
        # Fallback to Grötzsch
        adj = [0] * 16
        for u, v in grötzsch_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        best_adj = adj
        best_edge_count = count_edges(adj)
        
    # Convert back to list of edges
    edges = []
    for u in range(16):
        for v in range(u + 1, 16):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Helper to check if adding edge (u, v) creates a triangle
    def creates_triangle(adj, u, v):
        common = adj[u] & adj[v]
        return common != 0

    def count_edges(adj):
        return sum(bin(x).count('1') for x in adj) // 2

    def greedy_add_edges(adj, deadline):
        n = 16
        edges_to_consider = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(edges_to_consider)
        
        for u, v in edges_to_consider:
            if time.time() > deadline:
                break
            if not ((adj[u] >> v) & 1):
                if not creates_triangle(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    # Grötzsch Graph Edges
    grötzsch_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (0, 5), (1, 5), (1, 6), (2, 6), (2, 7), (3, 7), (3, 8), (4, 8), (4, 9), (0, 9),
        (5, 10), (6, 10), (7, 10), (8, 10), (9, 10)
    ]
    
    best_adj = None
    best_edge_count = 0
    
    # Run multiple restarts
    while time.time() < deadline:
        adj = [0] * 16
        for u, v in grötzsch_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        adj = greedy_add_edges(adj, deadline)
        
        edge_count = count_edges(adj)
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_adj = adj[:]
            
    if best_adj is None:
        adj = [0] * 16
        for u, v in grötzsch_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        best_adj = adj
        
    edges = []
    for u in range(16):
        for v in range(u + 1, 16):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
# EVOLVE-BLOCK-END
