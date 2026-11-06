import time
import random
import sys

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 115.0 # 5s safety margin
    
    # Adjacency matrix as a list of sets for fast intersection
    # Or bitsets for speed. Let's use sets first for simplicity, then optimize if needed.
    # With n=40, sets are fine.
    
    def get_book_size(adj):
        max_book = 0
        for u in range(n):
            for v in range(u + 1, n):
                if v in adj[u]:
                    common = len(adj[u] & adj[v])
                    if common > max_book:
                        max_book = common
        return max_book

    def random_graph():
        # Start with K20,20 + 1 random intra edge
        adj = [set() for _ in range(n)]
        # Bipartite
        for u in range(20):
            for v in range(20, 40):
                adj[u].add(v)
                adj[v].add(u)
        
        # Add 1 random intra edge
        # Try to pick vertices with low potential common neighbors? 
        # In K20,20, all have 20.
        # Just pick random
        a1, a2 = random.sample(range(20), 2)
        adj[a1].add(a2)
        adj[a2].add(a1)
        
        return adj

    def try_improve(adj):
        # Find worst edge
        worst_edge = None
        max_book = 0
        for u in range(n):
            for v in range(u + 1, n):
                if v in adj[u]:
                    common = len(adj[u] & adj[v])
                    if common > max_book:
                        max_book = common
                        worst_edge = (u, v)
        
        if max_book <= 7:
            return adj, max_book, True # Found optimal (likely)
            
        if not worst_edge:
            return adj, 0, True
            
        u, v = worst_edge
        # Triangles are (u, v, w) for w in common neighbors
        common_neighbors = list(adj[u] & adj[v])
        
        # Try to break a triangle by removing an edge (u, w) or (v, w)
        # and adding a random edge elsewhere.
        
        # To avoid bias, shuffle common neighbors
        random.shuffle(common_neighbors)
        
        # We also need a candidate edge to add.
        # A "good" edge to add is one with low common neighbors.
        # Let's find a few random non-edges with low book size.
        candidates_to_add = []
        for _ in range(10):
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if x != y and y not in adj[x]:
                # Estimate book size
                cs = len(adj[x] & adj[y])
                candidates_to_add.append((cs, x, y))
        
        # Sort by estimated book size
        candidates_to_add.sort(key=lambda item: item[0])
        
        best_new_adj = None
        best_new_score = max_book + 1 # We want to improve
        
        # Try removing one edge from the worst triangles
        # We can try removing (u, w) for a few w
        for w in common_neighbors[:5]: # Try top 5
            # Remove (u, w)
            adj[u].remove(w)
            adj[w].remove(u)
            
            # Try adding the best candidate
            for _, x, y in candidates_to_add[:3]:
                if x == y: continue
                if y in adj[x]: continue # It might have been affected? No, we only removed (u,w).
                # If (x,y) is (u,w) or (v,w) or something we just removed, skip?
                # Actually, if we removed (u,w), (u,w) is a non-edge now.
                
                adj[x].add(y)
                adj[y].add(x)
                
                # Calculate new max book
                # Optimization: Only need to check edges affected by removal/addition
                # But for n=40, full check is ~800 ops. 1000 iterations * 800 = 800k ops. Fast enough.
                new_score = get_book_size(adj)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    # Deep copy
                    best_new_adj = [s.copy() for s in adj]
                
                # Undo addition
                adj[x].remove(y)
                adj[y].remove(x)
            
            # Undo removal
            adj[u].add(w)
            adj[w].add(u)
            
            if best_new_score <= 7:
                break
        
        if best_new_adj is not None:
            return best_new_adj, best_new_score, False
        else:
            # If no improvement found, try a random restart or random move
            # To avoid getting stuck, sometimes accept a move with same score
            # Or just return current state
            return adj, max_book, False

    # Main loop
    best_adj = random_graph()
    best_score = get_book_size(best_adj)
    
    no_improve_count = 0
    
    while time.time() < deadline:
        current_adj, current_score, is_optimal = try_improve(best_adj)
        
        if is_optimal:
            best_adj = current_adj
            best_score = current_score
            break
            
        if current_score < best_score:
            best_adj = current_adj
            best_score = current_score
            no_improve_count = 0
        else:
            no_improve_count += 1
            # If stuck for a while, restart with a slightly different initial graph
            if no_improve_count > 100:
                # Perturb best_adj
                # Remove 2 random edges, add 2 random edges
                edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if v in best_adj[u]:
                            edges.append((u,v))
                
                # Remove 2
                rem = random.sample(edges, 2)
                for u, v in rem:
                    best_adj[u].remove(v)
                    best_adj[v].remove(u)
                    
                # Add 2
                added = 0
                while added < 2:
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u != v and v not in best_adj[u]:
                        best_adj[u].add(v)
                        best_adj[v].add(u)
                        added += 1
                        edges = [] # Rebuild edges list for next iteration if needed, but we are in restart logic
                        for uu in range(n):
                            for vv in range(uu+1, n):
                                if vv in best_adj[uu]:
                                    edges.append((uu,vv))
                
                best_score = get_book_size(best_adj)
                no_improve_count = 0
                continue

        # If we didn't break, loop continues with the same best_adj if no improvement
        # But try_improve returns the *new* adj if improved, or the *old* adj if not.
        # In the code above, if no improvement, it returns the old adj.
        # So we need to ensure we don't just spin.
        # The logic inside try_improve modifies a local copy? 
        # NO, `try_improve` modifies `adj` in place in the undo/redo steps, but returns `best_new_adj` (a copy) or `adj` (the original passed in).
        # Wait, if I pass `best_adj`, and it modifies it during the "try" phase, I must be careful.
        # In `try_improve`, I do `adj[u].remove(w)`. This modifies the input `adj`.
        # Then I undo it.
        # Then I return `adj` (the original, restored) if no improvement.
        # So `best_adj` remains unchanged if no improvement.
        # This leads to infinite loop if `try_improve` never finds an improvement.
        # The restart logic handles this by modifying `best_adj` directly.

    # Convert adj to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if v in best_adj[u]:
                edges.append([u, v])
                
    # Verify count
    if len(edges) != target_edges:
        # Fallback to seed if something went wrong
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return edges

# EVOLVE-BLOCK-START
def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 115.0 # 5s safety margin
    
    # Adjacency matrix as a list of sets for fast intersection
    def get_book_size(adj):
        max_book = 0
        for u in range(n):
            for v in range(u + 1, n):
                if v in adj[u]:
                    common = len(adj[u] & adj[v])
                    if common > max_book:
                        max_book = common
        return max_book

    def random_graph():
        adj = [set() for _ in range(n)]
        for u in range(20):
            for v in range(20, 40):
                adj[u].add(v)
                adj[v].add(u)
        
        a1, a2 = random.sample(range(20), 2)
        adj[a1].add(a2)
        adj[a2].add(a1)
        return adj

    def try_improve(adj):
        worst_edge = None
        max_book = 0
        for u in range(n):
            for v in range(u + 1, n):
                if v in adj[u]:
                    common = len(adj[u] & adj[v])
                    if common > max_book:
                        max_book = common
                        worst_edge = (u, v)
        
        if max_book <= 7:
            return adj, max_book, True
            
        if not worst_edge:
            return adj, 0, True
            
        u, v = worst_edge
        common_neighbors = list(adj[u] & adj[v])
        random.shuffle(common_neighbors)
        
        candidates_to_add = []
        for _ in range(10):
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if x != y and y not in adj[x]:
                cs = len(adj[x] & adj[y])
                candidates_to_add.append((cs, x, y))
        
        candidates_to_add.sort(key=lambda item: item[0])
        
        best_new_adj = None
        best_new_score = max_book + 1
        
        for w in common_neighbors[:5]:
            adj[u].remove(w)
            adj[w].remove(u)
            
            for _, x, y in candidates_to_add[:3]:
                if x == y: continue
                if y in adj[x]: continue
                
                adj[x].add(y)
                adj[y].add(x)
                
                new_score = get_book_size(adj)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_adj = [s.copy() for s in adj]
                
                adj[x].remove(y)
                adj[y].remove(x)
            
            adj[u].add(w)
            adj[w].add(u)
            
            if best_new_score <= 7:
                break
        
        if best_new_adj is not None:
            return best_new_adj, best_new_score, False
        else:
            return adj, max_book, False

    best_adj = random_graph()
    best_score = get_book_size(best_adj)
    no_improve_count = 0
    
    while time.time() < deadline:
        current_adj, current_score, is_optimal = try_improve(best_adj)
        
        if is_optimal:
            best_adj = current_adj
            best_score = current_score
            break
            
        if current_score < best_score:
            best_adj = current_adj
            best_score = current_score
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 100:
                edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if v in best_adj[u]:
                            edges.append((u,v))
                
                rem = random.sample(edges, 2)
                for u, v in rem:
                    best_adj[u].remove(v)
                    best_adj[v].remove(u)
                    
                added = 0
                while added < 2:
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u != v and v not in best_adj[u]:
                        best_adj[u].add(v)
                        best_adj[v].add(u)
                        added += 1
                        edges = []
                        for uu in range(n):
                            for vv in range(uu+1, n):
                                if vv in best_adj[uu]:
                                    edges.append((uu,vv))
                
                best_score = get_book_size(best_adj)
                no_improve_count = 0
                continue

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if v in best_adj[u]:
                edges.append([u, v])
                
    if len(edges) != target_edges:
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return edges
# EVOLVE-BLOCK-END
