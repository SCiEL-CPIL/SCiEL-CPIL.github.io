import random
import time

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 110
    
    A = list(range(20))
    B = list(range(20, 40))
    
    # State:
    # edges: set of tuples
    # adj: list of sets
    # book: dict (u,v) -> count
    
    def init_state(edges_set):
        adj = [set() for _ in range(n)]
        book = {}
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
            book[(u, v)] = 0
        # Calculate initial books
        for u, v in edges_set:
            book[(u, v)] = len(adj[u] & adj[v])
        return adj, book

    def get_max_book(book):
        return max(book.values()) if book else 0

    def generate_initial_graph(k_swaps):
        edges = set()
        for a in A:
            for b in B:
                edges.add((a, b))
        
        cross_edges = list(edges)
        random.shuffle(cross_edges)
        removed = cross_edges[:k_swaps]
        for e in removed:
            edges.remove(e)
            
        possible_internal = []
        for i in range(20):
            for j in range(i+1, 20):
                possible_internal.append((i, j))
        for i in range(20):
            for j in range(i+1, 20):
                possible_internal.append((i+20, j+20))
                
        random.shuffle(possible_internal)
        added = possible_internal[:k_swaps]
        for e in added:
            if e not in edges:
                edges.add(e)
        return edges

    # Find best init
    best_edges = None
    best_score = float('inf')
    
    for k in range(1, 11):
        if time.time() > deadline - 10: break
        init_edges = generate_initial_graph(k)
        adj, book = init_state(init_edges)
        score = get_max_book(book)
        if score < best_score:
            best_score = score
            best_edges = init_edges
            best_adj = adj
            best_book = book
            
    if best_edges is None:
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]

    current_edges = best_edges
    current_adj = best_adj
    current_book = best_book
    current_score = best_score
    
    all_possible_edges = set()
    for i in range(n):
        for j in range(i+1, n):
            all_possible_edges.add((i, j))
            
    no_improve_count = 0
    
    while time.time() < deadline:
        # Generate move
        # 1. Choose edge to remove
        if random.random() < 0.7:
            # Pick edge with max book
            max_b = 0
            worst_edge = None
            for e, b in current_book.items():
                if b > max_b:
                    max_b = b
                    worst_edge = e
            if worst_edge:
                u, v = worst_edge
            else:
                u, v = random.choice(list(current_edges))
        else:
            u, v = random.choice(list(current_edges))
            
        # 2. Choose edge to add
        # Pick random non-edge
        # To speed up, we can pick a random pair and check
        # Or precompute non-edges? Non-edges change.
        # Let's just pick random u2, v2 and check if not in edges.
        # Expected trials is small if graph is sparse (401/780 ~ 50% density).
        while True:
            u2 = random.randint(0, n-1)
            v2 = random.randint(0, n-1)
            if u2 == v2: continue
            if u2 > v2: u2, v2 = v2, u2
            if (u2, v2) not in current_edges:
                break
                
        # Apply move tentatively
        # Remove (u,v)
        # Add (u2,v2)
        
        # Affected edges:
        # Removing (u,v) decreases book size of neighbors of u and v that are connected to the other.
        # Specifically, for any x in adj[u] (excluding v), book[(u,x)] decreases by 1 if v in adj[x].
        # Wait, book[(u,x)] is number of common neighbors.
        # If we remove (u,v), v is no longer a neighbor of u.
        # So for any x in adj[u] \ {v}, if v in adj[x], then (u,x) loses common neighbor v.
        # Similarly for y in adj[v] \ {u}, if u in adj[y], (v,y) loses common neighbor u.
        
        # Adding (u2,v2) increases book size of neighbors of u2 and v2.
        # For any x in adj[u2], if v2 in adj[x], (u2,x) gains common neighbor v2.
        # For any y in adj[v2], if u2 in adj[y], (v2,y) gains common neighbor u2.
        
        # We need to update `current_book` and `current_adj` and `current_edges`.
        # To allow rollback, we should copy or store changes.
        # Given small size, copying `current_book` (401 entries) is cheap.
        
        new_book = dict(current_book)
        new_adj = [set(s) for s in current_adj] # Copy of sets
        new_edges = set(current_edges)
        
        # Remove (u,v)
        new_edges.remove((u,v))
        new_adj[u].remove(v)
        new_adj[v].remove(u)
        
        # Update books for edges incident to u or v
        # Edges (u, x) where x in new_adj[u]
        for x in list(new_adj[u]):
            if x == v: continue # v is removed
            # Check if v was a common neighbor of u and x BEFORE removal
            # i.e., is v in old_adj[x]?
            if v in current_adj[x]:
                new_book[(u, x) if u < x else (x, u)] -= 1
        for x in list(new_adj[v]):
            if x == u: continue
            if u in current_adj[x]:
                new_book[(v, x) if v < x else (x, v)] -= 1
                
        # Add (u2,v2)
        new_edges.add((u2,v2))
        new_adj[u2].add(v2)
        new_adj[v2].add(u2)
        
        # Update books for edges incident to u2 or v2
        # Note: (u2,v2) itself is a new edge, its book size is 0 initially?
        # No, book[(u2,v2)] = |adj[u2] & adj[v2]|.
        # But we are updating existing edges first.
        # Let's calculate book for (u2,v2) directly.
        common = len(new_adj[u2] & new_adj[v2])
        new_book[(u2, v2) if u2 < v2 else (v2, u2)] = common
        
        for x in list(new_adj[u2]):
            if x == v2: continue
            if v2 in new_adj[x]: # v2 is a common neighbor of u2 and x now?
                # Wait, we added (u2,v2). So v2 is neighbor of u2.
                # Is v2 neighbor of x?
                # If yes, then v2 is a common neighbor of u2 and x.
                # Did it exist before?
                # Before, (u2,v2) was not an edge, so v2 was not neighbor of u2.
                # So v2 was NOT a common neighbor.
                # So we ADD 1.
                key = (u2, x) if u2 < x else (x, u2)
                if key in new_book:
                    new_book[key] += 1
        for x in list(new_adj[v2]):
            if x == u2: continue
            if u2 in new_adj[x]:
                key = (v2, x) if v2 < x else (x, v2)
                if key in new_book:
                    new_book[key] += 1
                    
        # Get new score
        new_score = get_max_book(new_book)
        
        # Acceptance
        if new_score < current_score:
            current_book = new_book
            current_adj = new_adj
            current_edges = new_edges
            current_score = new_score
            no_improve_count = 0
        else:
            if random.random() < 0.05 and new_score <= current_score + 1:
                current_book = new_book
                current_adj = new_adj
                current_edges = new_edges
                current_score = new_score
            no_improve_count += 1
            if no_improve_count > 2000:
                # Restart
                k = random.randint(1, 15)
                init_edges = generate_initial_graph(k)
                adj, book = init_state(init_edges)
                score = get_max_book(book)
                if score < current_score:
                    current_edges = init_edges
                    current_adj = adj
                    current_book = book
                    current_score = score
                no_improve_count = 0

    return [[u, v] for u, v in current_edges]
