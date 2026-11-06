import random
import time
from itertools import combinations

def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    n = 40
    target_edges = 401
    
    def compute_book_sizes(edge_set, adj):
        """Compute book size for each edge. Returns max book size and the dict."""
        book = {}
        for (u, v) in edge_set:
            if u > v:
                u, v = v, u
            # Count common neighbors
            common = len(adj[u] & adj[v])
            book[(u, v)] = common
        return max(book.values()) if book else 0, book
    
    def build_adj(edge_set, n):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def try_swap(edge_set, adj, u, v, x, y, current_max_book):
        """Try swapping: remove (u,v), add (x,y). Return new_max if improvement, else None."""
        # Check validity
        if (u, v) not in edge_set and (v, u) not in edge_set:
            return None
        if x == y:
            return None
        if (x, y) in edge_set or (y, x) in edge_set:
            return None
        if x == u or x == v or y == u or y == v:
            return None
        
        # Remove edge (u,v), add edge (x,y)
        # Update adjacency
        adj[u].discard(v)
        adj[v].discard(u)
        adj[x].add(y)
        adj[y].add(x)
        
        new_edge_set = set(edge_set)
        if (u, v) in new_edge_set:
            new_edge_set.discard((u, v))
        else:
            new_edge_set.discard((v, u))
        new_edge_set.add((min(x,y), max(x,y)))
        
        # Recompute max book
        new_max = 0
        for (a, b) in new_edge_set:
            common = len(adj[a] & adj[b])
            if common > new_max:
                new_max = common
        
        # Undo
        adj[u].add(v)
        adj[v].add(u)
        adj[x].discard(y)
        adj[y].discard(x)
        
        if new_max < current_max_book:
            return new_max, new_edge_set, (u, v), (x, y)
        return None
    
    # Initial construction: start with a Turán-like structure
    # Use 4 parts to keep degrees balanced
    # 40 vertices, 401 edges. Complete 4-partite with parts 10,10,10,10 has 6*100=600 edges. Too many.
    # Let's use a different approach: start with K_{20,20} minus some edges plus some intra edges
    # Better: start random and optimize
    
    random.seed(42)
    
    best_edge_set = None
    best_max_book = float('inf')
    
    # Try multiple random initializations
    for init_iter in range(30):
        if time.time() > deadline - 10:
            break
        
        # Create a random graph with 401 edges
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        edge_set = set(random.sample(all_possible, target_edges))
        
        adj = build_adj(edge_set, n)
        current_max, _ = compute_book_sizes(edge_set, adj)
        
        if current_max < best_max_book:
            best_max_book = current_max
            best_edge_set = set(edge_set)
        
        # Local search
        improved = True
        no_improve_count = 0
        while improved and time.time() < deadline - 10:
            improved = False
            no_improve_count += 1
            if no_improve_count > 50:
                break
            
            # Try random swaps
            edge_list = list(edge_set)
            random.shuffle(edge_list)
            
            for _ in range(2000):
                if time.time() > deadline - 10:
                    break
                if not edge_list:
                    break
                
                (u, v) = random.choice(edge_list)
                # Pick random non-edge
                for attempt in range(50):
                    x = random.randint(0, n-1)
                    y = random.randint(0, n-1)
                    if x == y or (x,y) in edge_set or (y,x) in edge_set:
                        continue
                    if x == u or x == v or y == u or y == v:
                        continue
                    break
                else:
                    continue
                
                result = try_swap(edge_set, adj, u, v, x, y, current_max)
                if result is not None:
                    new_max, new_edge_set, removed, added = result
                    # Apply the swap
                    edge_set = new_edge_set
                    adj = build_adj(edge_set, n)
                    current_max = new_max
                    improved = True
                    no_improve_count = 0
                    if current_max < best_max_book:
                        best_max_book = current_max
                        best_edge_set = set(edge_set)
                    if current_max <= 7:  # Theoretical lower bound
                        break
            
            if current_max <= 7:
                break
    
    # Convert to list of [u, v] pairs
    result = [[u, v] for u, v in best_edge_set]
    # Ensure exactly 401 distinct edges
    assert len(result) == 401, f"Expected 401 edges, got {len(result)}"
    # Ensure all vertices are in 0..39
    for u, v in result:
        assert 0 <= u <= 39 and 0 <= v <= 39
    return result
