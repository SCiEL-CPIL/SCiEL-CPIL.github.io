import random

def construct():
    n = 40
    m = 401
    
    adj = [0] * n
    books = {}
    edge_set = set()
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def add_edge(u, v):
        if u > v: u, v = v, u
        edge_set.add((u, v))
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        common_mask = adj[u] & adj[v]
        c = common_mask.bit_count()
        books[(u, v)] = c
        
        w = 0
        cm = common_mask
        while cm:
            if cm & 1:
                e1 = (u, w) if u < w else (w, u)
                e2 = (v, w) if v < w else (w, v)
                if e1 in books:
                    books[e1] += 1
                if e2 in books:
                    books[e2] += 1
            cm >>= 1
            w += 1

    def remove_edge(u, v):
        if u > v: u, v = v, u
        edge_set.discard((u, v))
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        # Remove book for (u,v)
        if (u, v) in books:
            del books[(u, v)]
            
        # Decrease books for neighbors w in common
        common_mask = adj[u] & adj[v]
        w = 0
        cm = common_mask
        while cm:
            if cm & 1:
                e1 = (u, w) if u < w else (w, u)
                e2 = (v, w) if v < w else (w, v)
                if e1 in books:
                    books[e1] -= 1
                if e2 in books:
                    books[e2] -= 1
            cm >>= 1
            w += 1

    # Greedy Construction
    current_max_book = 0
    
    for _ in range(m):
        best_cost = (float('inf'), float('inf'), float('inf'))
        best_edge = None
        
        for u, v in all_pairs:
            if (u, v) in edge_set:
                continue
                
            common_mask = adj[u] & adj[v]
            c = common_mask.bit_count()
            
            max_affected = 0
            cm = common_mask
            w = 0
            while cm:
                if cm & 1:
                    e1 = (u, w) if u < w else (w, u)
                    e2 = (v, w) if v < w else (w, v)
                    b1 = books.get(e1, 0) + 1
                    b2 = books.get(e2, 0) + 1
                    if b1 > max_affected: max_affected = b1
                    if b2 > max_affected: max_affected = b2
                cm >>= 1
                w += 1
                
            new_max = max(current_max_book, c, max_affected)
            cost = (new_max, c, max_affected)
            
            if cost < best_cost:
                best_cost = cost
                best_edge = (u, v)
                
        if best_edge is None:
            break
            
        u, v = best_edge
        add_edge(u, v)
        current_max_book = best_cost[0]
        
    # Local Search (Hill Climbing)
    # Calculate current max
    max_book = max(books.values()) if books else 0
    sum_books = sum(books.values())
    
    improved = True
    while improved and max_book > 7:
        improved = False
        # Try all swaps
        # To speed up, we can iterate edges to remove and random non-edges to add?
        # Or all pairs? 780*400 = 312k swaps. Each swap O(1) update? 
        # Update is O(deg) ~ 20. So 6M ops. 1 iteration.
        # If we need multiple iterations, it might be slow.
        # Let's try a limited number of iterations.
        
        for _ in range(100): # Limit iterations
            if max_book <= 7: break
            found = False
            # Shuffle edges to avoid bias
            edges_list = list(edge_set)
            random.shuffle(edges_list)
            
            for u, v in edges_list:
                # Try to remove (u,v) and add (x,y)
                # Pick random non-edge
                while True:
                    x, y = random.choice(all_pairs)
                    if (x, y) not in edge_set:
                        break
                        
                # Calculate delta
                # This is expensive to do for all x,y.
                # Let's just do a few random swaps per edge.
                for _ in range(10):
                    while True:
                        x, y = random.choice(all_pairs)
                        if (x, y) not in edge_set:
                            break
                            
                    # Perform swap
                    remove_edge(u, v)
                    add_edge(x, y)
                    
                    new_max = max(books.values())
                    new_sum = sum(books.values())
                    
                    if new_max < max_book or (new_max == max_book and new_sum < sum_books):
                        max_book = new_max
                        sum_books = new_sum
                        improved = True
                        found = True
                        break
                    else:
                        # Revert
                        remove_edge(x, y)
                        add_edge(u, v)
                        
                if found:
                    break
                    
            if max_book <= 7:
                break
            if not found:
                break
                
    return [[u, v] for u, v in edge_set]
