"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s limit
    
    def compute_max_book(adj_matrix, n=40):
        """Compute max book size from adjacency matrix."""
        max_book = 0
        for u in range(n):
            for v in range(u+1, n):
                if adj_matrix[u][v]:
                    # Count common neighbors
                    common = 0
                    for w in range(n):
                        if w != u and w != v and adj_matrix[u][w] and adj_matrix[v][w]:
                            common += 1
                    if common > max_book:
                        max_book = common
        return max_book
    
    def compute_max_book_edges(edges, n=40):
        """Compute max book size from edge list."""
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            adj[u][v] = 1
            adj[v][u] = 1
        return compute_max_book(adj, n)
    
    def random_quasi_regular(n=40, target_edges=400, seed=0):
        """Generate a random graph with ~target edges that's close to regular."""
        rng = random.Random(seed)
        adj = np.zeros((n, n), dtype=np.int8)
        
        # Start with a random 20-regular graph using configuration model
        degree = 20
        stubs = [i for i in range(n) for _ in range(degree)]
        rng.shuffle(stubs)
        
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v or adj[u][v]:
                valid = False
                break
            adj[u][v] = 1
            adj[v][u] = 1
        
        if not valid:
            # Fallback: random graph with 400 edges
            all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
            rng.shuffle(all_pairs)
            for u, v in all_pairs[:400]:
                adj[u][v] = 1
                adj[v][u] = 1
        
        # Add 1 more edge to get 401
        existing_edges = set()
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    existing_edges.add((u, v))
        
        available = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in existing_edges]
        if available:
            u, v = rng.choice(available)
            adj[u][v] = 1
            adj[v][u] = 1
        
        return adj
    
    def local_search(adj, n=40, max_iterations=5000, seed=0):
        """Local search to minimize max book size."""
        rng = random.Random(seed)
        current_adj = adj.copy()
        current_book = compute_max_book(current_adj, n)
        best_adj = current_adj.copy()
        best_book = current_book
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Try swapping an edge: remove (u,v), add (x,y)
            # Find edges with high book contribution
            edge_list = []
            for u in range(n):
                for v in range(u+1, n):
                    if current_adj[u][v]:
                        edge_list.append((u, v))
            
            if not edge_list:
                break
            
            # Pick a random edge to remove
            u, v = rng.choice(edge_list)
            
            # Find a non-edge to add (prefer edges that reduce book)
            non_edges = []
            for x in range(n):
                for y in range(x+1, n):
                    if not current_adj[x][y]:
                        non_edges.append((x, y))
            
            if not non_edges:
                continue
            
            # Try a few random non-edges
            candidates = rng.sample(non_edges, min(10, len(non_edges)))
            best_candidate = None
            best_candidate_book = current_book
            
            for x, y in candidates:
                # Temporarily swap
                current_adj[u][v] = 0
                current_adj[v][u] = 0
                current_adj[x][y] = 1
                current_adj[y][x] = 1
                
                new_book = compute_max_book(current_adj, n)
                
                # Undo
                current_adj[u][v] = 1
                current_adj[v][u] = 1
                current_adj[x][y] = 0
                current_adj[y][x] = 0
                
                if new_book < best_candidate_book:
                    best_candidate_book = new_book
                    best_candidate = (x, y)
            
            if best_candidate is not None:
                x, y = best_candidate
                current_adj[u][v] = 0
                current_adj[v][u] = 0
                current_adj[x][y] = 1
                current_adj[y][x] = 1
                current_book = best_candidate_book
                
                if current_book < best_book:
                    best_book = current_book
                    best_adj = current_adj.copy()
        
        return best_adj, best_book
    
    best_edges = None
    best_book = float('inf')
    
    # Try multiple random starting points
    for seed in range(50):
        if time.time() > deadline - 10:
            break
        
        adj = random_quasi_regular(seed=seed)
        book = compute_max_book(adj)
        
        if book < best_book:
            best_book = book
            best_adj = adj.copy()
        
        # Run local search
        adj_ls, book_ls = local_search(adj, seed=seed)
        if book_ls < best_book:
            best_book = book_ls
            best_adj = adj_ls.copy()
        
        if time.time() > deadline - 20:
            break
    
    # Convert best adjacency matrix to edge list
    edges = []
    for u in range(40):
        for v in range(u+1, 40):
            if best_adj[u][v]:
                edges.append([u, v])
    
    # Verify we have exactly 401 edges
    if len(edges) != 401:
        # Fallback to seed solution
        edges = [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    
    return edges
# EVOLVE-BLOCK-END
