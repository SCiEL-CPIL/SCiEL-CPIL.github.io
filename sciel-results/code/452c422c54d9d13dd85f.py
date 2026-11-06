import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 40
    m = 401
    
    # Build adjacency as list of sets
    # For speed, also maintain a set of edges
    
    def make_adj_from_edges(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def get_edges_from_adj(adj):
        edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.add((u,v))
        return edges
    
    def compute_max_book(adj):
        max_b = 0
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    b = len(adj[u] & adj[v])
                    if b > max_b:
                        max_b = b
        return max_b
    
    # Start with K_{20,20} + random within-part edge
    random.seed(123)
    
    best_adj = None
    best_score = float('inf')
    
    for trial in range(30):
        # Start with K_{20,20}
        adj = [set() for _ in range(n)]
        for i in range(20):
            for j in range(20, 40):
                adj[i].add(j)
                adj[j].add(i)
        
        # Add 1 edge within a part
        if random.random() < 0.5:
            u, v = random.sample(range(20), 2)
        else:
            u, v = random.sample(range(20, 40), 2)
        adj[u].add(v)
        adj[v].add(u)
        
        current_score = compute_max_book(adj)
        
        # Local search
        max_iters = 5000
        for it in range(max_iters):
            if time.time() - start_time > deadline:
                break
            
            # Find worst edge
            worst_edge = None
            worst_book = -1
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        b = len(adj[u] & adj[v])
                        if b > worst_book:
                            worst_book = b
                            worst_edge = (u, v)
            
            if worst_book <= 7:
                break
            
            # Try to swap worst_edge with a random non-edge
            a, b = worst_edge
            found_improvement = False
            
            # Try several random non-edges
            for _ in range(50):
                u, v = random.sample(range(n), 2)
                if v not in adj[u]:
                    # Simulate swap
                    new_book_worst = worst_book
                    # After removing (a,b), edges incident to a or b change
                    # After adding (u,v), edges incident to u or v change
                    
                    # Quick check: what's the book of (u,v) in current adj?
                    book_uv = len(adj[u] & adj[v])
                    
                    # After removing (a,b) and adding (u,v):
                    # The book of (u,v) might change if a or b is common neighbor
                    # Actually this is complex. Let's just compute fully for now.
                    
                    new_adj = [set(s) for s in adj]
                    new_adj[a].remove(b)
                    new_adj[b].remove(a)
                    new_adj[u].add(v)
                    new_adj[v].add(u)
                    
                    new_score = compute_max_book(new_adj)
                    
                    if new_score < current_score:
                        adj = new_adj
                        current_score = new_score
                        found_improvement = True
                        break
                    elif new_score == current_score and random.random() < 0.2:
                        adj = new_adj
                        found_improvement = True
                        break
            
            if not found_improvement and it > 100:
                # Try a different strategy: swap two non-worst edges
                pass
        
        if current_score < best_score:
            best_score = current_score
            best_adj = [set(s) for s in adj]
        
        if best_score <= 7:
            break
    
    # Convert to list
    result = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                result.append([u, v])
    
    return result
