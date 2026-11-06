"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # The theorem suggests we need a graph with 40 vertices and 401 edges
    # that minimizes the largest book (max triangles per edge).
    # K_{20,20} has 400 edges and no triangles. Adding one edge creates triangles.
    # The seed uses K_{20,20} + [0,1] which creates 20 triangles containing edge [0,1].
    # But wait - in K_{20,20}, vertices 0..19 are in one part, 20..39 in the other.
    # Edge [0,1] is within the first part. Triangles containing [0,1] are formed by
    # any vertex v in the second part: 0-v-1-0. So 20 triangles.
    # 
    # The lower bound is 40/6 ≈ 6.67, so we need at least 7.
    # We can do better than 20 by using a different construction.
    #
    # Let's try: start with a random 40-vertex graph with 401 edges and optimize
    # by edge swaps to minimize the maximum book size.
    
    def compute_books(edges, n=40):
        """Compute the book size for each edge. Returns max book size."""
        edge_set = set()
        for u, v in edges:
            if u > v:
                u, v = v, u
            edge_set.add((u, v))
        
        # Build adjacency for fast lookup
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # For each edge, count triangles containing it
        max_book = 0
        for u, v in edge_set:
            # Common neighbors of u and v
            common = adj[u] & adj[v]
            book_size = len(common)
            if book_size > max_book:
                max_book = book_size
        return max_book
    
    # Start with a good initial graph
    # Strategy: Use a nearly complete bipartite graph but with some modifications
    # to reduce the book size.
    
    # Let's try a different approach: use a random regular-ish graph and optimize.
    # Or better: start with K_{20,20} and modify edges.
    
    # Actually, let's think more carefully. The key insight is that we want to
    # distribute triangles evenly. 
    # 
    # One idea: use a 20-regular graph on 40 vertices with 400 edges (which is K_{20,20})
    # and add one more edge. But K_{20,20} + one edge gives book size 20.
    #
    # Better idea: use a different base graph. What if we use a union of two K_{10,10}
    # plus some cross edges? No, that's still bipartite.
    #
    # Let's try: take a random graph with 401 edges and do local search.
    
    def random_graph(n=40, m=401):
        """Generate a random graph with n vertices and m edges."""
        edges = []
        edge_set = set()
        max_edges = n * (n - 1) // 2
        while len(edges) < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) not in edge_set:
                edge_set.add((u, v))
                edges.append([u, v])
        return edges
    
    def edge_swap(edges, n=40):
        """Perform a random edge swap: remove one edge, add another."""
        edge_set = set()
        for u, v in edges:
            if u > v:
                u, v = v, u
            edge_set.add((u, v))
        
        # Pick a random edge to remove
        edge_list = list(edge_set)
        if not edge_list:
            return edges
        remove_idx = random.randint(0, len(edge_list) - 1)
        u, v = edge_list[remove_idx]
        
        # Pick a random edge to add (that doesn't already exist)
        new_u, new_v = u, v
        attempts = 0
        while attempts < 100:
            new_u = random.randint(0, n - 1)
            new_v = random.randint(0, n - 1)
            if new_u == new_v:
                continue
            if new_u > new_v:
                new_u, new_v = new_v, new_u
            if (new_u, new_v) not in edge_set:
                break
            attempts += 1
        else:
            return edges  # Couldn't find a valid new edge
        
        # Apply the swap
        edge_set.remove((u, v))
        edge_set.add((new_u, new_v))
        
        return [[a, b] for a, b in edge_set]
    
    # Try multiple random restarts
    best_edges = None
    best_score = float('inf')
    
    # First, try the K_{20,20} + edge approach with different edge placements
    # In K_{20,20}, parts are {0..19} and {20..39}
    # Adding edge within a part creates 20 triangles.
    # But what if we use a different partition?
    
    # Let's try local search from K_{20,20} + [0,1]
    edges = [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    current_score = compute_books(edges)
    best_edges = edges[:]
    best_score = current_score
    
    # Local search: try to swap edges to reduce max book
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline and no_improve < max_no_improve:
        new_edges = edge_swap(edges)
        if new_edges == edges:
            no_improve += 1
            continue
        
        new_score = compute_books(new_edges)
        
        if new_score < current_score:
            edges = new_edges
            current_score = new_score
            no_improve = 0
            
            if current_score < best_score:
                best_score = current_score
                best_edges = edges[:]
        else:
            no_improve += 1
    
    # Try a few random restarts if time permits
    if time.time() < deadline - 30:
        for restart in range(3):
            if time.time() > deadline - 20:
                break
            
            edges = random_graph()
            current_score = compute_books(edges)
            
            no_improve = 0
            while time.time() < deadline - 20 and no_improve < 500:
                new_edges = edge_swap(edges)
                if new_edges == edges:
                    no_improve += 1
                    continue
                
                new_score = compute_books(new_edges)
                
                if new_score < current_score:
                    edges = new_edges
                    current_score = new_score
                    no_improve = 0
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = edges[:]
                else:
                    no_improve += 1
    
    # Ensure we return exactly 401 distinct edges
    edge_set = set()
    for u, v in best_edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    # If we don't have exactly 401 edges, fix it
    while len(edge_set) < 401:
        u = random.randint(0, 39)
        v = random.randint(0, 39)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    # If we have more than 401, remove extras
    if len(edge_set) > 401:
        edge_list = list(edge_set)
        random.shuffle(edge_list)
        edge_set = set(edge_list[:401])
    
    return [[u, v] for u, v in edge_set]
# EVOLVE-BLOCK-END
