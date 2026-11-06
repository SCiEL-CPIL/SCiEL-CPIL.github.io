"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a graph on 40 vertices with exactly 401 edges minimizing the largest book.
    
    Strategy:
    1. Start with a near-extremal bipartite graph K_{20,20} (400 edges).
    2. Add 1 extra edge. The choice of this edge determines the book size.
    3. The theoretical lower bound is 7 (from Edwards-Khadzhiivanov-Nikiforov theorem: n/6 = 40/6 ≈ 6.67, so at least 7).
    4. To achieve book size 7, we need to carefully choose which edge to add.
    
    Analysis:
    - In K_{20,20}, every edge connects a vertex in A (0-19) to a vertex in B (20-39).
    - If we add an edge within A, say (u,v) where u,v in A:
      - The book size of (u,v) = number of common neighbors in B = 20 (since all B vertices connect to both u and v).
      - This gives book size 20, which is bad.
    
    - If we add an edge within B, similar issue.
    
    - So we must restructure the graph to reduce book sizes.
    
    Better approach:
    - Start with a random graph or a structured graph that has low book sizes.
    - Use local search to minimize the maximum book size.
    
    Key insight: The optimal graph likely has a structure where no edge is in too many triangles.
    
    Let's try a different construction:
    - Use a cycle or path-based structure to control triangle formation.
    
    Actually, let's think about this more carefully.
    
    The extremal number for books is related to the number of triangles. 
    For n=40 and m=401, we want to minimize the maximum number of triangles containing any edge.
    
    A good strategy:
    1. Generate a random graph with 401 edges.
    2. Use local search: repeatedly swap edges to reduce the maximum book size.
    
    Let's implement a local search approach.
    """
    n = 40
    target_edges = 401
    deadline = time.time() + 110  # 110 second deadline to be safe
    
    # Helper to compute book sizes
    def compute_books(adj_matrix, edge_list):
        """Compute the number of triangles containing each edge."""
        books = {}
        for u, v in edge_list:
            count = 0
            for w in range(n):
                if w != u and w != v and adj_matrix[u][w] and adj_matrix[w][v]:
                    count += 1
            books[(u, v)] = count
            books[(v, u)] = count
        return books
    
    # Helper to get max book
    def max_book(books):
        if not books:
            return 0
        return max(books.values())
    
    # Initial construction: start with a random graph
    def random_graph():
        edges = set()
        while len(edges) < target_edges:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                if u > v:
                    u, v = v, u
                edges.add((u, v))
        return list(edges)
    
    # Local search
    def local_search(edges, max_iter=100000):
        edges_set = set(edges)
        adj = np.zeros((n, n), dtype=bool)
        for u, v in edges_set:
            adj[u][v] = True
            adj[v][u] = True
        
        books = compute_books(adj, list(edges_set))
        current_max = max_book(books)
        best_max = current_max
        best_edges = list(edges_set)
        
        iter_count = 0
        while iter_count < max_iter and time.time() < deadline:
            iter_count += 1
            
            # Try swapping an edge: remove (u,v) and add (x,y)
            # Choose an edge to remove: prefer edges with high book size
            high_book_edges = [(e, b) for e, b in books.items() if e[0] < e[1] and b >= current_max - 1]
            if not high_book_edges:
                high_book_edges = [(e, b) for e, b in books.items() if e[0] < e[1]]
            
            if not high_book_edges:
                break
                
            u, v = random.choice(high_book_edges)[0]
            
            # Choose a new edge to add
            # Prefer edges that would have low book size
            candidates = []
            for x in range(n):
                for y in range(x+1, n):
                    if (x, y) not in edges_set and (y, x) not in edges_set:
                        # Estimate book size of (x,y)
                        est = 0
                        for z in range(n):
                            if z != x and z != y and adj[x][z] and adj[z][y]:
                                est += 1
                        candidates.append((est, x, y))
            
            if not candidates:
                break
            
            # Sort by estimated book size and pick a few low ones
            candidates.sort()
            if len(candidates) > 20:
                candidates = candidates[:20]
            
            x, y = random.choice(candidates)[1], random.choice(candidates)[2]
            # Actually, let's just pick the best few
            x, y = candidates[0][1], candidates[0][2]
            
            # Perform swap
            edges_set.remove((u, v)) if (u, v) in edges_set else edges_set.remove((v, u))
            adj[u][v] = False
            adj[v][u] = False
            
            edges_set.add((x, y))
            adj[x][y] = True
            adj[y][x] = True
            
            # Recompute books (this is expensive, so we could optimize)
            books = compute_books(adj, list(edges_set))
            new_max = max_book(books)
            
            if new_max <= current_max:
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_edges = list(edges_set)
            else:
                # Revert
                edges_set.remove((x, y))
                adj[x][y] = False
                adj[y][x] = False
                
                edges_set.add((u, v))
                adj[u][v] = True
                adj[v][u] = True
                
                books = compute_books(adj, list(edges_set))
                current_max = max_book(books)
        
        return best_edges, best_max
    
    # Start with a good initial graph
    # Use a structured initial graph: a cycle plus chords
    def structured_initial():
        edges = set()
        # Add a cycle
        for i in range(n):
            edges.add((i, (i+1) % n))
        
        # Add random edges until we have 401
        while len(edges) < target_edges:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                if u > v:
                    u, v = v, u
                if (u, v) not in edges:
                    edges.add((u, v))
        return list(edges)
    
    # Try multiple random restarts
    best_overall = None
    best_score = float('inf')
    
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        if r == 0:
            # First try: structured initial
            edges = structured_initial()
        else:
            # Random initial
            edges = random_graph()
        
        edges, score = local_search(edges, max_iter=50000)
        
        if score < best_score:
            best_score = score
            best_overall = edges
        
        if best_score <= 7:
            break
    
    if best_overall is None:
        # Fallback: return a valid graph
        best_overall = random_graph()
    
    return [[u, v] for u, v in best_overall]
# EVOLVE-BLOCK-END
