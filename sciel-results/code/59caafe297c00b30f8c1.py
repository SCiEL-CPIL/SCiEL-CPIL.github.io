import time
import random
import numpy as np

def construct():
    deadline = time.time() + 115
    n = 40
    m = 401
    
    # Bitmask representation
    # adj[i] is an integer where bit j is set if edge (i, j) exists
    def popcount(x):
        return bin(x).count('1')
    
    def get_books(adj, edge_set):
        books = {}
        for u, v in edge_set:
            books[(u, v)] = popcount(adj[u] & adj[v])
        return books
    
    def get_max_book(books):
        if not books:
            return 0
        return max(books.values())
    
    def update_books_for_swap(adj, edge_set, u, v, x, y, books):
        # Remove edge (u, v)
        # Identify affected edges: incident to u or v
        # We need to update books for:
        # 1. Edges incident to u or v that remain in the graph
        # 2. Edges incident to x or y (which might be new or existing)
        
        # To be safe and simple, we can just recompute books for all edges incident to u, v, x, y
        # But we must be careful not to double count or miss.
        
        # Let's collect all candidate edges to update
        candidates = set()
        
        # Edges incident to u
        for w in range(n):
            if w != u and (u, w) in edge_set or (w, u) in edge_set:
                candidates.add((min(u, w), max(u, w)))
        # Edges incident to v
        for w in range(n):
            if w != v and (v, w) in edge_set or (w, v) in edge_set:
                candidates.add((min(v, w), max(v, w)))
        # Edges incident to x
        for w in range(n):
            if w != x and (x, w) in edge_set or (w, x) in edge_set:
                candidates.add((min(x, w), max(x, w)))
        # Edges incident to y
        for w in range(n):
            if w != y and (y, w) in edge_set or (w, y) in edge_set:
                candidates.add((min(y, w), max(y, w)))
        
        # Also, the new edge (x, y) itself might be in the set (if we are adding it)
        # And the old edge (u, v) was removed.
        
        # Wait, the swap is: remove (u,v), add (x,y).
        # So (u,v) is NOT in edge_set anymore.
        # (x,y) IS in edge_set.
        
        # We need to update books for all edges in `candidates`.
        # Note: If an edge is not in edge_set, it shouldn't have a book entry.
        # We should remove entries for edges no longer in edge_set?
        # Actually, `books` is a dict. We can just update the values.
        # If an edge is removed, we should remove it from books.
        # If an edge is added, we add it to books.
        
        # Let's just recompute the book size for every edge in `candidates` that is in `edge_set`.
        # And remove any candidates that are NOT in `edge_set` from `books`.
        
        for e in candidates:
            if e in edge_set:
                a, b = e
                books[e] = popcount(adj[a] & adj[b])
            else:
                if e in books:
                    del books[e]
        
        # What about edges that were affected but are NOT in candidates?
        # Book size of edge (a,b) depends on N(a) & N(b).
        # If we change adj[u] and adj[v] (removing edge u-v), does it affect edge (a,b) where a,b not in {u,v,x,y}?
        # No, because N(a) and N(b) don't change unless a or b is u, v, x, or y.
        # So `candidates` covers all affected edges.
        
        # However, we must ensure that if we add (x,y), and (x,y) was not in candidates?
        # (x,y) is incident to x and y, so it IS in candidates.
        
        # One edge case: what if (u,v) was the only edge with a certain book size?
        # The dictionary handles this naturally.
        
        return books

    # Initial State: Random
    random.seed(42)
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    init_edges = all_edges[:m]
    
    adj = [0] * n
    edge_set = set()
    for u, v in init_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((u, v))
        
    books = get_books(adj, edge_set)
    current_max = get_max_book(books)
    
    best_adj = adj[:]
    best_edge_set = set(edge_set)
    best_max = current_max
    best_books = dict(books)
    
    # Simulated Annealing
    temp = 50.0
    cooling = 0.9995
    min_temp = 0.5
    
    steps_since_improvement = 0
    max_steps_without_improvement = 20000
    
    # Precompute list of edges for random selection
    # We need to pick a random edge to remove and a random non-edge to add.
    # Maintaining a list of non-edges is expensive if we update it every step.
    # Instead, we can just sample random pairs and check.
    
    while time.time() < deadline:
        if steps_since_improvement > max_steps_without_improvement:
            # Restart with a new random state or kick
            # Let's try a kick: random 10 swaps
            for _ in range(10):
                u, v = random.choice(list(edge_set))
                found = False
                for _ in range(50):
                    x = random.randint(0, n-1)
                    y = random.randint(0, n-1)
                    if x == y: continue
                    if (x, y) in edge_set or (y, x) in edge_set: continue
                    if (x, y) == (u, v) or (y, x) == (u, v): continue
                    found = True
                    break
                if found:
                    # Apply swap
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    adj[x] |= (1 << y)
                    adj[y] |= (1 << x)
                    edge_set.discard((u, v))
                    edge_set.discard((v, u))
                    if x < y: edge_set.add((x, y))
                    else: edge_set.add((y, x))
                    books = update_books_for_swap(adj, edge_set, u, v, x, y, books)
            current_max = get_max_book(books)
            steps_since_improvement = 0
            temp = max(temp, 10.0)
            
        # Pick edge to remove
        # Bias towards high book edges
        if random.random() < 0.7:
            # Pick from top 20% book sizes
            # This is expensive to sort every time.
            # Approximation: pick random, if book < current_max/2, retry?
            # Or just pick random.
            u, v = random.choice(list(edge_set))
        else:
            u, v = random.choice(list(edge_set))
            
        # Pick non-edge to add
        found = False
        for _ in range(100):
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y: continue
            if (x, y) in edge_set or (y, x) in edge_set: continue
            if (x, y) == (u, v) or (y, x) == (u, v): continue
            found = True
            break
            
        if not found:
            temp *= cooling
            continue
            
        # Save state
        old_u = adj[u]
        old_v = adj[v]
        old_x = adj[x]
        old_y = adj[y]
        old_books = dict(books)
        
        # Apply move
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        adj[x] |= (1 << y)
        adj[y] |= (1 << x)
        
        edge_set.discard((u, v))
        edge_set.discard((v, u))
        if x < y: 
            new_edge = (x, y)
            edge_set.add(new_edge)
        else: 
            new_edge = (y, x)
            edge_set.add(new_edge)
            
        # Update books
        books = update_books_for_swap(adj, edge_set, u, v, x, y, books)
        new_max = get_max_book(books)
        
        # Accept/Reject
        delta = new_max - current_max
        if delta <= 0 or random.random() < np.exp(-delta / temp):
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_adj = adj[:]
                best_edge_set = set(edge_set)
                best_books = dict(books)
                steps_since_improvement = 0
            # If no improvement in best, but accepted move, we still count it as "activity"
            # but we don't reset steps_since_improvement unless best improves?
            # Usually SA resets on improvement of current state, but for finding global best,
            # we might want to keep searching.
            # Let's reset on any improvement of current_max as well, to encourage exploration of new basins.
            # Actually, standard SA doesn't reset "steps since global best".
            # Let's just increment steps_since_improvement if best_max didn't improve.
            if new_max >= best_max:
                steps_since_improvement += 1
        else:
            # Reject: Undo
            adj[u] = old_u
            adj[v] = old_v
            adj[x] = old_x
            adj[y] = old_y
            edge_set.add((u, v))
            edge_set.discard(new_edge)
            books = old_books
            steps_since_improvement += 1
            
        temp *= cooling
        if temp < min_temp:
            temp = min_temp
            
    # Return best
    result = [[u, v] for u, v in best_edge_set]
    assert len(result) == m
    return result
