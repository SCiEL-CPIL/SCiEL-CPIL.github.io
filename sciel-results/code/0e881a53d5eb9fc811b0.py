"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import heapq

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    n = 40
    target_edges = 401
    
    # Build adjacency structure for fast common-neighbor queries
    # adj[u] = set of neighbors of u
    # For each edge, book_size = |N(u) ∩ N(v)|
    
    def compute_books(adj, edge_list):
        """Compute book size for each edge."""
        books = {}
        for u, v in edge_list:
            common = len(adj[u] & adj[v])
            books[(u, v)] = common
            books[(v, u)] = common
        return books
    
    def max_book(books, edge_list):
        """Return the maximum book size."""
        mx = 0
        for u, v in edge_list:
            if (u, v) in books:
                mx = max(mx, books[(u, v)])
        return mx
    
    # Initial construction: use a near-bipartite graph but with more balanced structure
    # Start with K20,20 minus some edges plus different cross-edges to reduce books
    # Actually, let's start from a random graph with target edges and optimize
    
    random.seed(42)
    
    # Generate a random initial graph with 401 edges
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_possible_edges)
    edge_list = all_possible_edges[:target_edges]
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
    
    books = compute_books(adj, edge_list)
    current_max = max_book(books, edge_list)
    best_edges = list(edge_list)
    best_max = current_max
    
    # Set of current edges for O(1) lookup
    edge_set = set()
    for u, v in edge_list:
        edge_set.add((min(u,v), max(u,v)))
    
    # Local search: target the edge with maximum book, try to reduce it
    # Move types:
    # 1. For edge (u,v) with large book, pick a common neighbor w, 
    #    remove edge (u,w) or (v,w) and add a random other edge
    # 2. Swap: remove one edge from the book, add another edge elsewhere
    
    def try_improve(adj, edge_set, edge_list, books, current_max, max_iter=50000):
        best_edges = list(edge_list)
        best_max = current_max
        best_books = dict(books)
        
        for it in range(max_iter):
            if time.time() - start_time > DEADLINE:
                break
            
            # Find an edge with maximum book
            max_edge = None
            max_book_val = -1
            for u, v in edge_list:
                key = (min(u,v), max(u,v))
                b = books.get(key, 0)
                if b > max_book_val:
                    max_book_val = b
                    max_edge = (u, v)
            
            if max_book_val <= 0:
                break
            
            u, v = max_edge
            common = adj[u] & adj[v]
            if not common:
                continue
            
            # Try to break the book by removing an edge from a common neighbor
            w = random.choice(list(common))
            
            # Option 1: remove (u, w), add a random non-edge
            # Option 2: remove (v, w), add a random non-edge
            
            for remove_edge in [(u, w), (v, w)]:
                ru, rv = min(remove_edge), max(remove_edge)
                if (ru, rv) not in edge_set:
                    continue
                
                # Find a random edge to add (non-edge)
                # Try a few random candidates
                added = False
                for _ in range(20):
                    a = random.randint(0, n-1)
                    b = random.randint(0, n-1)
                    if a == b:
                        continue
                    ka = min(a, b)
                    kb = max(a, b)
                    if (ka, kb) in edge_set:
                        continue
                    # Check if this addition would create a worse max book
                    # Estimate: adding (a,b) creates book = |N(a)∩N(b)|
                    new_book_ab = len(adj[a] & adj[b])
                    # Removing (ru,rv) may reduce books of edges adjacent to ru or rv
                    # This is complex; just try it and check
                    if new_book_ab > max_book_val + 2:
                        continue
                    
                    # Apply the move
                    # Remove (ru, rv)
                    adj[ru].discard(rv)
                    adj[rv].discard(ru)
                    edge_set.discard((ru, rv))
                    
                    # Add (a, b)
                    adj[a].add(b)
                    adj[b].add(a)
                    edge_set.add((ka, kb))
                    
                    # Recompute books for affected edges
                    # Edges affected: those incident to ru, rv, a, b
                    affected = set()
                    for x in [ru, rv, a, b]:
                        for y in adj[x]:
                            affected.add((min(x,y), max(x,y)))
                    
                    new_max = 0
                    for e in affected:
                        eu, ev = e
                        bk = len(adj[eu] & adj[ev])
                        new_max = max(new_max, bk)
                    
                    # Also need to check global max - approximate by checking if new_max > current
                    # For speed, just check if this move improves
                    if new_max < max_book_val:
                        # Accept and update
                        # Update edge_list
                        edge_list = [e for e in edge_list if e != (ru,rv) and e != (rv,ru)]
                        edge_list.append([a, b])
                        
                        # Update books
                        for e in affected:
                            eu, ev = e
                            bk = len(adj[eu] & adj[ev])
                            books[e] = bk
                            books[(ev, eu)] = bk
                        
                        current_max = new_max
                        if new_max < best_max:
                            best_max = new_max
                            best_edges = list(edge_list)
                            best_books = dict(books)
                        added = True
                        break
                    else:
                        # Revert
                        adj[ru].add(rv)
                        adj[rv].add(ru)
                        edge_set.add((ru, rv))
                        
                        adj[a].discard(b)
                        adj[b].discard(a)
                        edge_set.discard((ka, kb))
                
                if added:
                    break
        
        return best_edges, best_max, best_books
    
    # Run local search
    edge_list, best_max, best_books = try_improve(adj, edge_set, edge_list, books, current_max)
    
    if time.time() - start_time > DEADLINE:
        # Return best found
        return [[u, v] for u, v in best_edges]
    
    # Second phase: more targeted search on the best solution
    # Rebuild from best_edges
    adj2 = [set() for _ in range(n)]
    for u, v in best_edges:
        adj2[u].add(v)
        adj2[v].add(u)
    
    edge_set2 = set()
    for u, v in best_edges:
        edge_set2.add((min(u,v), max(u,v)))
    
    books2 = compute_books(adj2, best_edges)
    current_max2 = max_book(books2, best_edges)
    
    # More aggressive: try all possible single-edge modifications for the worst edge
    for phase in range(3):
        if time.time() - start_time > DEADLINE:
            break
        
        for it in range(20000):
            if time.time() - start_time > DEADLINE:
                break
            
            # Find worst edge
            max_edge = None
            max_book_val = -1
            for u, v in best_edges:
                key = (min(u,v), max(u,v))
                b = books2.get(key, 0)
                if b > max_book_val:
                    max_book_val = b
                    max_edge = (u, v)
            
            if max_book_val <= 0:
                break
            
            u, v = max_edge
            common = adj2[u] & adj2[v]
            if not common:
                continue
            
            # Try removing each common neighbor edge
            improved = False
            for w in random.sample(list(common), min(len(common), 10)):
                for remove_edge in [(u, w), (v, w)]:
                    ru, rv = min(remove_edge), max(remove_edge)
                    if (ru, rv) not in edge_set2:
                        continue
                    
                    # Find best edge to add
                    best_add = None
                    best_new_max = max_book_val + 1
                    
                    # Sample candidate edges to add
                    candidates = []
                    for _ in range(50):
                        a = random.randint(0, n-1)
                        b = random.randint(0, n-1)
                        if a == b:
                            continue
                        ka = min(a, b)
                        kb = max(a, b)
                        if (ka, kb) in edge_set2:
                            continue
                        candidates.append((a, b))
                    
                    for a, b in candidates:
                        ka = min(a, b)
                        kb = max(a, b)
                        new_book_ab = len(adj2[a] & adj2[b])
                        
                        # Apply tentatively
                        adj2[ru].discard(rv)
                        adj2[rv].discard(ru)
                        edge_set2.discard((ru, rv))
                        adj2[a].add(b)
                        adj2[b].add(a)
                        edge_set2.add((ka, kb))
                        
                        # Check max book on affected edges
                        affected = set()
                        for x in [ru, rv, a, b]:
                            for y in adj2[x]:
                                affected.add((min(x,y), max(x,y)))
                        
                        new_max = 0
                        for e in affected:
                            eu, ev = e
                            bk = len(adj2[eu] & adj2[ev])
                            new_max = max(new_max, bk)
                        
                        # Revert
                        adj2[ru].add(rv)
                        adj2[rv].add(ru)
                        edge_set2.add((ru, rv))
                        adj2[a].discard(b)
                        adj2[b].discard(a)
                        edge_set2.discard((ka, kb))
                        
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_add = (a, b)
                    
                    if best_add and best_new_max < max_book_val:
                        a, b = best_add
                        ka = min(a, b)
                        kb = max(a, b)
                        
                        # Apply for real
                        adj2[ru].discard(rv)
                        adj2[rv].discard(ru)
                        edge_set2.discard((ru, rv))
                        adj2[a].add(b)
                        adj2[b].add(a)
                        edge_set2.add((ka, kb))
                        
                        # Update edge list
                        best_edges = [e for e in best_edges if e != (ru,rv) and e != (rv,ru)]
                        best_edges.append([a, b])
                        
                        # Update books
                        affected = set()
                        for x in [ru, rv, a, b]:
                            for y in adj2[x]:
                                affected.add((min(x,y), max(x,y)))
                        for e in affected:
                            eu, ev = e
                            bk = len(adj2[eu] & adj2[ev])
                            books2[e] = bk
                            books2[(ev, eu)] = bk
                        
                        current_max2 = best_new_max
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
    
    if time.time() - start_time > DEADLINE:
        return [[u, v] for u, v in best_edges]
    
    return [[u, v] for u, v in best_edges]
# EVOLVE-BLOCK-END
