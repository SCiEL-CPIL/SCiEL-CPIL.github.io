import time
import random
from itertools import combinations

def construct():
    """Construct a dense triangle-free graph on 16 vertices that is not 3-colorable."""
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    n = 16
    best_edges = []
    best_count = 0
    
    # Use a known construction: start with a 4-chromatic triangle-free graph
    # The Grötzsch graph has 11 vertices. We can extend it.
    # Alternative: use a randomized approach with backtracking
    
    # Start with a base structure and try to add edges greedily
    # while maintaining triangle-free property and checking non-3-colorability
    
    # A better approach: use the fact that Mycielski construction preserves
    # triangle-free and increases chromatic number. But we need exactly 16 vertices.
    
    # Let's use a systematic search:
    # 1. Generate candidate edge sets
    # 2. Check triangle-free
    # 3. Check not 3-colorable (using backtracking)
    # 4. Maximize edge count
    
    def is_triangle_free(edges):
        """Check if graph with given edges is triangle-free."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        for u in range(n):
            for v in adj[u]:
                if u >= v:
                    continue
                # Check common neighbors
                common = adj[u] & adj[v]
                if common:
                    return False
        return True
    
    def is_3_colorable(edges):
        """Check if graph with given edges is 3-colorable using backtracking."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        color = [-1] * n
        
        def backtrack(vertex):
            if vertex == n:
                return True
            for c in range(3):
                if all(color[neighbor] != c for neighbor in adj[vertex]):
                    color[vertex] = c
                    if backtrack(vertex + 1):
                        return True
                    color[vertex] = -1
            return False
        
        return backtrack(0)
    
    # Strategy: Start with a known 4-chromatic triangle-free graph on fewer vertices
    # and extend it, or use a dense random initialization with local search
    
    # Let's try a different approach: use the complement of a sparse graph
    # Triangle-free means no three mutually adjacent vertices
    
    # Start with a random dense graph and remove edges to make it triangle-free
    # while trying to keep it not 3-colorable
    
    random.seed(42)
    
    # Generate many random triangle-free graphs and check if they're not 3-colorable
    # Focus on graphs with high edge count
    
    # A systematic approach: use the fact that for triangle-free graphs,
    # by Turan's theorem, max edges for triangle-free is n^2/4 = 64 for n=16
    # But we need chromatic number >= 4, which is much more restrictive
    
    # Let's try constructing from known structures:
    # The Grötzsch graph (11 vertices, 20 edges, 4-chromatic, triangle-free)
    # We can add 5 more vertices and connect them appropriately
    
    # Actually, let's use a simpler approach: try to build the graph incrementally
    # with a greedy algorithm that prioritizes adding edges while maintaining
    # triangle-free property, then verify non-3-colorability
    
    # Use a randomized local search:
    # 1. Start with a random triangle-free graph
    # 2. Try to add edges while maintaining triangle-free
    # 3. Check if not 3-colorable
    # 4. If yes, record; if no, try to modify to make it not 3-colorable
    
    # For efficiency, let's use a heuristic:
    # Graphs with high minimum degree are more likely to have high chromatic number
    # among triangle-free graphs
    
    # Let's try a structured approach based on known extremal examples
    # The Clebsch graph is 16 vertices, triangle-free, but it's 3-colorable
    # We need a different structure
    
    # Try: use a bipartite base and add edges within parts carefully
    # Or: use a 5-cycle structure replicated
    
    # Practical approach: try many random graphs with varying densities
    # and keep the best not-3-colorable triangle-free graph
    
    best_edges = []
    best_edge_count = 0
    
    for trial in range(5000):
        if time.time() > deadline:
            break
        
        # Generate a random triangle-free graph
        # Start with empty and add edges randomly, checking triangle-free
        edges = []
        adj = [set() for _ in range(n)]
        
        # Try to add edges in random order
        all_pairs = list(combinations(range(n), 2))
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            # Check if adding (u,v) creates a triangle
            creates_triangle = False
            for w in adj[u]:
                if w in adj[v]:
                    creates_triangle = True
                    break
            
            if not creates_triangle:
                # Add edge with some probability to control density
                if random.random() < 0.6:  # Tune this parameter
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        
        edge_count = len(edges)
        if edge_count <= best_edge_count:
            continue
        
        # Check if triangle-free (should be by construction, but verify)
        if not is_triangle_free(edges):
            continue
        
        # Check if not 3-colorable
        if not is_3_colorable(edges):
            best_edges = edges[:]
            best_edge_count = edge_count
    
    # Convert to list of lists as required
    return [[u, v] for u, v in best_edges]
