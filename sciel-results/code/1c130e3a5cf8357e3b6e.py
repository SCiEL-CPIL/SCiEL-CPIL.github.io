"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    N = 40
    E = 401
    
    def compute_books(edge_set, adj):
        """Compute book size for each edge. Book of edge (u,v) = number of w such that (u,w) and (v,w) are edges."""
        books = {}
        for (u, v) in edge_set:
            # common neighbors
            common = len(adj[u] & adj[v])
            books[(u, v)] = common
            books[(v, u)] = common
        return books
    
    def max_book(books):
        return max(books.values()) if books else 0
    
    def random_graph():
        """Generate a random graph with exactly E edges."""
        edges = set()
        while len(edges) < E:
            u = random.randint(0, N-1)
            v = random.randint(0, N-1)
            if u != v:
                edges.add((min(u,v), max(u,v)))
        return edges
    
    def build_adj(edges):
        adj = [set() for _ in range(N)]
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def local_search(edges, max_iter=5000):
        """Greedy local search: try to reduce max book by edge swaps."""
        adj = build_adj(edges)
        books = compute_books(edges, adj)
        current_max = max_book(books)
        
        improved = True
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            
            # Find edges with high book
            high_edges = [(e, b) for e, b in books.items() if e[0] < e[1] and b >= current_max]
            random.shuffle(high_edges)
            
            for (u, v), book_val in high_edges:
                if book_val < current_max:
                    continue
                    
                # Try removing (u,v) and adding a non-edge
                # A good non-edge to add: one where both endpoints have low book contribution
                # Try random non-edges
                for _ in range(20):
                    a = random.randint(0, N-1)
                    b = random.randint(0, N-1)
                    if a == b or (a, b) in edges or (b, a) in edges:
                        continue
                    if a > b:
                        a, b = b, a
                    new_edge = (a, b)
                    
                    # Check if adding new_edge creates too many triangles
                    # New book for new_edge = |adj[a] & adj[b]|
                    new_book = len(adj[a] & adj[b])
                    if new_book >= current_max:
                        continue
                    
                    # Perform swap
                    edges.remove((u, v))
                    adj[u].discard(v)
                    adj[v].discard(u)
                    edges.add(new_edge)
                    adj[a].add(b)
                    adj[b].add(a)
                    
                    # Recompute books (expensive but necessary)
                    books = compute_books(edges, adj)
                    new_max = max_book(books)
                    
                    if new_max < current_max:
                        current_max = new_max
                        improved = True
                        break
                    else:
                        # Revert
                        edges.remove(new_edge)
                        adj[a].discard(b)
                        adj[b].discard(a)
                        edges.add((u, v))
                        adj[u].add(v)
                        adj[v].add(u)
                        books = compute_books(edges, adj)
                
                if improved:
                    break
        
        return edges, current_max
    
    # Try multiple random starts with local search
    best_edges = None
    best_max = 999
    
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        edges = random_graph()
        edges, cur_max = local_search(edges, max_iter=3000)
        
        if cur_max < best_max:
            best_max = cur_max
            best_edges = edges.copy()
        
        # If we hit the theoretical lower bound, stop
        if best_max <= 7:
            break
    
    # Final local search refinement on best
    if best_edges is not None and time.time() < deadline - 5:
        best_edges, best_max = local_search(best_edges, max_iter=10000)
    
    if best_edges is None:
        best_edges = random_graph()
    
    # Convert to list of [u, v] pairs
    result = [[u, v] for (u, v) in best_edges]
    return result

# EVOLVE-BLOCK-END
