"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a graph with 40 vertices and 401 edges minimizing the largest book size.
    Uses a fast local search with incremental book updates, targeting worst edges.
    """
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n = 40
    m = 401
    
    # Helper: compute max book from adjacency sets
    def compute_max_book(adj, edges):
        max_b = 0
        for e in edges:
            u, v = e
            b = len(adj[u] & adj[v])
            if b > max_b:
                max_b = b
        return max_b
    
    # Helper: compute all books
    def compute_all_books(adj, edges):
        books = {}
        for e in edges:
            u, v = e
            books[e] = len(adj[u] & adj[v])
        return books
    
    # Build initial graph: K_{20,20} with some modifications to reduce book size
    # Strategy: Start with a structure that has low triangle density
    
    def make_initial_graph():
        """Create an initial graph with 401 edges and relatively low max book."""
        adj = [set() for _ in range(n)]
        edges = set()
        
        # Base: K_{20,20} bipartite (no triangles)
        for a in range(20):
            for b in range(20, 40):
                edges.add((a, b))
                adj[a].add(b)
                adj[b].add(a)
        
        # We need 1 more edge. Adding any intra-part edge creates 20 triangles.
        # Instead, let's remove some bipartite edges and add intra-part edges
        # in a way that distributes triangles more evenly.
        
        # Remove 5 random bipartite edges
        bip_edges = [(a, b) for a in range(20) for b in range(20, 40)]
        random.shuffle(bip_edges)
        to_remove = bip_edges[:5]
        for u, v in to_remove:
            edges.remove((u, v))
            adj[u].remove(v)
            adj[v].remove(u)
        
        # Add 6 intra-part edges: distribute among vertices to minimize max book
        # In part A (0..19), add edges that share few common neighbors
        # Each intra-A edge (u,v) creates triangles with each common neighbor in B
        # Since we removed 5 edges, some vertices in A have degree < 20 in B
        
        # Add edges in part A: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11)
        intra_edges = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11)]
        for u, v in intra_edges:
            edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
        
        # Verify count
        assert len(edges) == 401, f"Initial edges: {len(edges)}"
        
        return edges, adj
    
    def random_initial_graph():
        """Generate a random graph with 401 edges."""
        adj = [set() for _ in range(n)]
        edges = set()
        
        # Randomly add 401 edges
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        for u, v in all_possible[:401]:
            edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
        
        return edges, adj
    
    # Try multiple initial constructions
    best_edges = None
    best_score = float('inf')
    best_adj = None
    
    # Initial graph 1: Modified K_{20,20}
    edges, adj = make_initial_graph()
    books = compute_all_books(adj, edges)
    current_score = max(books.values())
    
    if current_score < best_score:
        best_score = current_score
        best_edges = set(edges)
        best_adj = [s.copy() for s in adj]
    
    # Local search with incremental updates
    def local_search(edges, adj, books, max_iter=100000):
        """Local search targeting worst edges with incremental book updates."""
        current_score = max(books.values())
        no_improve = 0
        max_no_improve = 5000
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Find worst edges
            worst_edges = [e for e in edges if books[e] == current_score]
            if not worst_edges:
                break
            random.shuffle(worst_edges)
            
            improved = False
            
            for worst_e in worst_edges[:3]:  # Limit work per iteration
                if time.time() > deadline:
                    break
                
                u, v = worst_e
                
                # Try to remove worst_e and add a new edge
                # The new edge should have low book size
                # Removal of (u,v) reduces books of edges sharing u or v
                
                for _ in range(30):  # Sample candidates
                    x = random.randint(0, n-1)
                    y = random.randint(0, n-1)
                    if x == y:
                        continue
                    if x > y:
                        x, y = y, x
                    new_e = (x, y)
                    if new_e in edges:
                        continue
                    if new_e == worst_e:
                        continue
                    
                    # Quick check: new edge book size (before swap)
                    # adj[x] and adj[y] don't change due to removing (u,v) 
                    # unless x or y is u or v
                    common_new = adj[x] & adj[y]
                    if len(common_new) >= current_score:
                        continue
                    
                    # Perform swap
                    edges.remove(worst_e)
                    edges.add(new_e)
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].add(y)
                    adj[y].add(x)
                    
                    # Incremental update of books
                    affected = set()
                    # Edges incident to u
                    for w in list(adj[u]):
                        affected.add((min(u,w), max(u,w)))
                    # Edges incident to v
                    for w in list(adj[v]):
                        affected.add((min(v,w), max(v,w)))
                    # Edges incident to x
                    for w in list(adj[x]):
                        affected.add((min(x,w), max(x,w)))
                    # Edges incident to y
                    for w in list(adj[y]):
                        affected.add((min(y,w), max(y,w)))
                    
                    # Update books for affected edges
                    for e in affected:
                        if e in edges:
                            eu, ev = e
                            books[e] = len(adj[eu] & adj[ev])
                    
                    # Add new edge's book
                    books[new_e] = len(adj[x] & adj[y])
                    # Remove old edge's book
                    del books[worst_e]
                    
                    # Check improvement
                    new_max = max(books.values())
                    
                    if new_max < current_score:
                        current_score = new_max
                        no_improve = 0
                        improved = True
                        break
                    else:
                        # Revert
                        edges.remove(new_e)
                        edges.add(worst_e)
                        adj[u].add(v)
                        adj[v].add(u)
                        adj[x].remove(y)
                        adj[y].remove(x)
                        
                        # Revert books
                        for e in affected:
                            if e in edges:
                                eu, ev = e
                                books[e] = len(adj[eu] & adj[ev])
                        books[worst_e] = len(adj[u] & adj[v])
                        del books[new_e]
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if not improved:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
        
        return current_score, edges, adj, books
    
    # Run local search from initial graph
    score, edges, adj, books = local_search(edges, adj, books)
    if score < best_score:
        best_score = score
        best_edges = set(edges)
        best_adj = [s.copy() for s in adj]
    
    # Try random restarts if time permits
    if time.time() < deadline - 30:
        for restart in range(5):
            if time.time() > deadline - 20:
                break
            
            edges, adj = random_initial_graph()
            books = compute_all_books(adj, edges)
            score, edges, adj, books = local_search(edges, adj, books, max_iter=50000)
            
            if score < best_score:
                best_score = score
                best_edges = set(edges)
                best_adj = [s.copy() for s in adj]
    
    # Ensure we return exactly 401 distinct edges
    edge_list = list(best_edges)
    
    # If we don't have exactly 401 edges, fix it
    while len(edge_list) < 401:
        u = random.randint(0, 39)
        v = random.randint(0, 39)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        e = (u, v)
        if e not in best_edges:
            best_edges.add(e)
            edge_list.append(e)
    
    # If we have more than 401, remove extras
    if len(edge_list) > 401:
        random.shuffle(edge_list)
        edge_list = edge_list[:401]
        best_edges = set(edge_list)
    
    return [[u, v] for u, v in edge_list]
# EVOLVE-BLOCK-END

