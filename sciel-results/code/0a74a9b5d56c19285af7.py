import random
import time

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n = 40
    target_edges = 401
    
    def compute_books(adj):
        """Compute book size for each edge. adj is a list of sets."""
        max_book = 0
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    common = len(adj[u] & adj[v])
                    if common > max_book:
                        max_book = common
        return max_book
    
    def random_graph():
        edges = set()
        while len(edges) < target_edges:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                edges.add((min(u,v), max(u,v)))
        return edges
    
    def build_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    # Start with a random graph
    best_edges = random_graph()
    best_adj = build_adj(best_edges)
    best_score = compute_books(best_adj)
    
    # Local search: try edge swaps to reduce max book
    def try_improve(edges, adj, score):
        """Try to improve by removing one edge and adding another."""
        edge_list = list(edges)
        random.shuffle(edge_list)
        
        for i in range(min(200, len(edge_list))):
            if time.time() > deadline:
                return edges, adj, score
            
            u, v = edge_list[i]
            
            # Try removing (u,v) and adding a random non-edge
            # First, compute how much this edge contributes to book sizes
            # If we remove (u,v), the book size of (u,v) disappears, but 
            # books of edges (u,w) and (v,w) decrease by 1 for each common neighbor w
            
            # Try adding a random non-edge
            attempts = 0
            while attempts < 10:
                a = random.randint(0, n-1)
                b = random.randint(0, n-1)
                if a != b and (min(a,b), max(a,b)) not in edges:
                    na, nb = min(a,b), max(a,b)
                    # Check if this swap helps
                    # Remove (u,v), add (na,nb)
                    adj[u].discard(v)
                    adj[v].discard(u)
                    adj[na].add(nb)
                    adj[nb].add(na)
                    
                    new_score = compute_books(adj)
                    
                    if new_score < score:
                        # Accept
                        edges.discard((u,v))
                        edges.add((na,nb))
                        return edges, adj, new_score
                    else:
                        # Revert
                        adj[u].add(v)
                        adj[v].add(u)
                        adj[na].discard(nb)
                        adj[nb].discard(na)
                    attempts += 1
                    break  # Only try one candidate per removed edge
        
        return edges, adj, score
    
    # Run local search with time limit
    while time.time() < deadline:
        best_edges, best_adj, best_score = try_improve(best_edges, best_adj, best_score)
    
    # Convert to list of [u,v] pairs
    result = [[u, v] for u, v in sorted(best_edges)]
    return result
