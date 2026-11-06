import random
import time
from collections import defaultdict

def construct():
    n = 40
    target_edges = 401
    
    # Start with a base structure: balanced bipartite + some internal edges
    # We'll use a randomized approach guided by the book-size minimization
    
    # Initialize with a random graph
    def random_graph():
        edges = set()
        # Start with a complete bipartite graph K20,20
        for a in range(20):
            for b in range(20, 40):
                edges.add((min(a, b), max(a, b)))
        # Add 1 internal edge to reach 401
        # Actually K20,20 has 400 edges, so we need exactly 1 more
        edges.add((0, 1))
        return edges
    
    def count_books(edges):
        """Count triangles for each edge. Return max book size."""
        # Build adjacency for fast triangle counting
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        edge_to_book = {}
        # For each edge, count common neighbors
        for u, v in edges:
            common = adj[u] & adj[v]
            edge_to_book[(u, v)] = len(common)
        
        return max(edge_to_book.values()) if edge_to_book else 0
    
    def get_largest_book_edges(edges):
        """Return the edge(s) with the largest book."""
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        max_book = -1
        worst_edges = []
        for u, v in edges:
            common = len(adj[u] & adj[v])
            if common > max_book:
                max_book = common
                worst_edges = [(u, v)]
            elif common == max_book:
                worst_edges.append((u, v))
        return worst_edges, max_book
    
    def try_improve(edges, iterations=5000):
        """Local search: try edge swaps to reduce max book."""
        current_edges = set(edges)
        current_max = count_books(current_edges)
        
        adj = defaultdict(set)
        for u, v in current_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        def book_size(edge):
            u, v = edge
            return len(adj[u] & adj[v])
        
        best_max = current_max
        
        for _ in range(iterations):
            # Find the worst edge
            worst_edges, worst_book = get_largest_book_edges(current_edges)
            
            # Try to break the worst edge by swapping it with a non-edge
            # that doesn't create a larger book
            for ue, ve in random.sample(worst_edges, min(1, len(worst_edges))):
                # Find potential replacement edges
                # We want to remove (ue, ve) and add (x, y) where
                # the new book sizes are smaller
                
                # Try adding edges that don't share many neighbors
                candidates = []
                for x in range(n):
                    for y in range(x + 1, n):
                        if (x, y) in current_edges or (y, x) in current_edges:
                            continue
                        if (min(x, y), max(x, y)) == (ue, ve):
                            continue
                        # Check if adding (x, y) would create a large book
                        # and removing (ue, ve) would help
                        new_edges = current_edges - {(ue, ve)}
                        new_edges.add((min(x, y), max(x, y)))
                        
                        # Quick estimate: book size of new edge
                        new_book = len((adj[x] - {ue, ve}) & (adj[y] - {ue, ve}))
                        # Also check if removing (ue, ve) reduces other books
                        
                        if new_book < worst_book:
                            candidates.append((x, y, new_book))
                
                if not candidates:
                    continue
                
                # Pick the candidate with smallest new book
                x, y, new_book = min(candidates, key=lambda c: c[2])
                
                # Apply the swap
                current_edges.remove((ue, ve))
                current_edges.add((x, y))
                
                # Update adjacency
                adj[ue].discard(ve)
                adj[ve].discard(ue)
                adj[x].add(y)
                adj[y].add(x)
                
                # Check new max book
                new_max = count_books(current_edges)
                
                if new_max <= best_max:
                    best_max = new_max
                else:
                    # Revert
                    current_edges.remove((x, y))
                    current_edges.add((ue, ve))
                    adj[ve].add(ue)
                    adj[ue].add(ve)
                    adj[y].discard(x)
                    adj[x].discard(y)
        
        return current_edges
    
    # Start from a good initial guess
    edges = random_graph()
    
    # Run local search with time limit
    start = time.time()
    timeout = 110  # seconds
    
    while time.time() - start < timeout:
        edges = try_improve(edges, iterations=1000)
        if time.time() - start > timeout:
            break
    
    return [list(e) for e in edges]
