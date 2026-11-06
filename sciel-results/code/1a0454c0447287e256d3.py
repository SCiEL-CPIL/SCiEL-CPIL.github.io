import random
import time

def construct():
    """
    Constructs a graph with 40 vertices and 401 edges minimizing the largest book size.
    Uses local search on a random regular-like graph.
    """
    n = 40
    target_edges = 401
    
    # Degree sequence: 38 vertices of degree 20, 2 vertices of degree 21
    # Sum = 38*20 + 2*21 = 760 + 42 = 802 = 2 * 401
    degrees = [20] * 38 + [21, 21]
    
    def generate_random_graph(n, degrees):
        # Havel-Hakimi or simple random matching with retries
        # For small n, we can use a simple algorithm
        # Create a list of vertices with their remaining degrees
        remaining = degrees[:]
        edges = set()
        
        # Simple random greedy: pick a vertex with max remaining degree, connect to random others
        # This might get stuck, so we use a more robust method: 
        # Generate a random permutation of the multiset of degrees and pair them up?
        # Better: Use networkx-style generation if available, but we only have stdlib.
        # Let's use a simple backtracking or retry loop for random generation.
        
        # Actually, for n=40, we can just generate a random graph and adjust degrees?
        # Or use the "Configuration Model" which creates multi-graphs, then remove loops/multi-edges.
        
        # Configuration Model:
        # Create a list of 'stubs'
        stubs = []
        for i in range(n):
            stubs.extend([i] * degrees[i])
        
        while True:
            random.shuffle(stubs)
            edges_candidate = set()
            valid = True
            # Pair up stubs
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if (u, v) in edges_candidate or (v, u) in edges_candidate:
                    valid = False
                    break
                edges_candidate.add((u, v))
            if valid:
                return [list(e) for e in edges_candidate]

    # Initial graph
    edges = generate_random_graph(n, degrees)
    
    # Helper to calculate book sizes
    def get_books(edge_list):
        adj = {i: set() for i in range(n)}
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
            
        books = {}
        for u, v in edge_list:
            common = len(adj[u].intersection(adj[v]))
            books[(u, v)] = common
            books[(v, u)] = common
        return adj, books

    adj, books = get_books(edges)
    max_book = max(books.values())
    
    # Local Search
    # We try to swap edges to reduce max book.
    # Swap (u, v) with (x, y) where u,v,x,y are distinct.
    # This preserves degrees.
    
    start_time = time.time()
    timeout = 110 # Safety margin
    
    best_edges = edges
    best_max_book = max_book
    
    while time.time() - start_time < timeout:
        # Find an edge with high book size
        # To be efficient, we can just pick a random edge and try to improve it, 
        # or pick the worst edge.
        # Let's pick a random edge from the current set, and try to swap it.
        
        u, v = random.choice(edges)
        current_book = books[(u, v)]
        
        # Try to find a swap that reduces the max book
        # We need to remove (u, v) and add (x, y)
        # x, y must not be u or v.
        # Also (x, y) must not be an existing edge.
        
        # To keep degrees same, we just swap.
        # We want to reduce the book size of the new edge (x, y) and potentially others.
        
        # Let's try a few random swaps
        for _ in range(50): # Attempts per iteration
            x, y = random.sample([i for i in range(n) if i not in (u, v)], 2)
            if (x, y) in [(a, b) for a, b in edges] or (y, x) in [(a, b) for a, b in edges]:
                continue
            
            # Calculate change in book sizes
            # Removing (u, v) affects books of edges incident to u or v? 
            # Actually, book of (a,b) is |N(a) ∩ N(b)|.
            # If we remove (u,v):
            # - Book of (u,v) disappears.
            # - For any edge (u, w), N(u) loses v. So book(u,w) might decrease.
            # - For any edge (v, w), N(v) loses u. So book(v,w) might decrease.
            # - For other edges, no change?
            #   Wait, if (a,b) is not incident to u or v, N(a) and N(b) are unchanged.
            
            # Adding (x, y):
            # - New book for (x,y) is |N(x) ∩ N(y)|.
            # - For any edge (x, w), N(x) gains y. So book(x,w) might increase.
            # - For any edge (y, w), N(y) gains x. So book(y,w) might increase.
            
            # This calculation is expensive if done naively.
            # Given n=40, we can just recalculate the whole book distribution for the candidate graph?
            # 401 edges. Recalc is O(E * (deg^2)) or similar.
            # Let's just generate the new edge list and check max book.
            
            new_edges = [e for e in edges if e != [u, v] and e != [v, u]]
            new_edges.append([x, y])
            
            # Check validity: no duplicates
            # Since we checked (x,y) not in edges, and we removed (u,v), and u,v != x,y, 
            # and graph was simple, it should be simple.
            
            # Calculate max book for new_edges
            adj_new = {i: set() for i in range(n)}
            for a, b in new_edges:
                adj_new[a].add(b)
                adj_new[b].add(a)
                
            max_b = 0
            for a, b in new_edges:
                b_size = len(adj_new[a].intersection(adj_new[b]))
                if b_size > max_b:
                    max_b = b_size
            
            if max_b < best_max_book:
                best_max_book = max_b
                best_edges = new_edges
                edges = new_edges
                adj, books = get_books(edges)
                # Restart randomization? Or continue?
                # Continue, but we updated the global state.
                # We should probably break out of the inner loop to restart selection
                break
            elif max_b == best_max_book:
                # Tie-breaker: minimize sum of squares?
                # Or just accept to explore?
                # Let's accept with small probability to escape local minima
                if random.random() < 0.1:
                    best_edges = new_edges
                    edges = new_edges
                    adj, books = get_books(edges)
                    break

    return best_edges
