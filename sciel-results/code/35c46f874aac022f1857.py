import time
import random
import itertools

def construct():
    """
    Constructs a simple graph on 20 vertices with edge colours 0..19.
    Maximizes the number of edges while ensuring every 4-cycle is rainbow.
    
    Strategy: Greedy edge addition with incremental 4-cycle constraint checking.
    We attempt to add edges in a random order. For each candidate edge (u, v),
    we identify all potential 4-cycles that would be formed by adding this edge.
    A 4-cycle is formed if there is a vertex w such that (u, w) and (w, v) are
    already in the graph. Actually, a 4-cycle involving (u, v) requires a path
    u-w-x-v of length 3. So for a new edge (u, v), we look for all w != u, v
    such that (u, w) is an edge. Then for each such w, we look for x != u, v, w
    such that (w, x) and (x, v) are edges.
    
    If adding (u, v) creates a 4-cycle, we must ensure the 4 edges of that cycle
    have distinct colors. Since we assign colors greedily, we can check if there
    is a color available for (u, v) that is distinct from the other 3 edges in
    the cycle. If multiple cycles are created, the color must be distinct from
    all edges in all those cycles.
    
    If no valid color exists for (u, v) given the current graph, we skip the edge.
    This allows us to include edges that form cycles, provided the coloring is valid.
    """
    n = 20
    deadline = time.time() + 115  # Safety margin for 120s limit
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_edge_count = 0
    best_edges = []
    
    def check_and_add(edges_set, adj, color_of_edge, candidate):
        """
        Attempts to add 'candidate' edge to the graph defined by 'adj' and 'color_of_edge'.
        Returns True if added, False otherwise.
        Modifies adj, color_of_edge, and edges_set in place if successful.
        """
        u, v = candidate
        
        # Find all 4-cycles that would be formed by adding (u, v)
        # A 4-cycle is u-w-x-v-u.
        # We need w such that (u, w) is in graph.
        # Then x such that (w, x) and (x, v) are in graph.
        
        forbidden_colors = set()
        
        # Iterate over neighbors of u
        for w in adj[u]:
            if w == v:
                continue
            # For each neighbor w of u, find common neighbors of w and v
            # x must be a neighbor of w and a neighbor of v
            # x != u (since (u,x) would make a triangle or shorter cycle, 
            # but we are looking for 4-cycles u-w-x-v-u. 
            # Note: If x==u, it's a triangle u-w-u-v-u? No, x is distinct from u in a simple 4-cycle path u-w-x-v.
            # The cycle is u-w-x-v-u. Vertices u,w,x,v must be distinct.
            for x in adj[w]:
                if x == u or x == v or x == w:
                    continue
                if x in adj[v]:
                    # We found a 4-cycle: u-w-x-v-u
                    # The edges are (u,w), (w,x), (x,v), (v,u)
                    # We need color(v,u) to be distinct from colors of (u,w), (w,x), (x,v)
                    c1 = color_of_edge.get((u, w))
                    c2 = color_of_edge.get((w, x))
                    c3 = color_of_edge.get((x, v))
                    
                    # Since (u,w), (w,x), (x,v) are existing edges, they must have colors
                    if c1 is None or c2 is None or c3 is None:
                        # This shouldn't happen if we maintain consistency
                        continue
                        
                    forbidden_colors.add(c1)
                    forbidden_colors.add(c2)
                    forbidden_colors.add(c3)
        
        # Check if there is a valid color for (u, v)
        # Colors are 0..19
        valid_color = None
        for c in range(20):
            if c not in forbidden_colors:
                valid_color = c
                break
        
        if valid_color is None:
            return False
        
        # Add the edge
        adj[u].add(v)
        adj[v].add(u)
        color_of_edge[(u, v)] = valid_color
        color_of_edge[(v, u)] = valid_color
        edges_set.add(candidate)
        return True

    # Randomized greedy search with restarts
    # We try to build a graph by shuffling edges and adding them greedily.
    # We keep the best result found across multiple restarts.
    
    while time.time() < deadline:
        random.shuffle(all_edges)
        adj = {i: set() for i in range(n)}
        color_of_edge = {}
        edges_set = set()
        
        for u, v in all_edges:
            if time.time() > deadline - 5:
                break
            
            # Check if adding (u, v) is valid
            if check_and_add(edges_set, adj, color_of_edge, (u, v)):
                pass # Added
        
        current_count = len(edges_set)
        if current_count > best_edge_count:
            best_edge_count = current_count
            # Store the result
            best_edges = [[u, v, color_of_edge[(u, v)]] for u, v in edges_set]
            
            # If we reached a very high number of edges, we might be close to optimal
            # Max possible is 190. C4-free max is ~60-70. 
            # With coloring, we can likely exceed C4-free limits.
            # If we hit 100+, it's a good sign.
            if best_edge_count > 100:
                 # We can stop early if we are very confident, but let's keep searching a bit more
                 # unless time is critical.
                 pass

    if not best_edges:
        # Fallback to star
        return [[0, v, 0] for v in range(1, n)]
        
    return best_edges

