import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    m = 401
    random.seed(42)
    
    # Bitmask representation
    adj_mask = [0] * n
    
    def compute_books(adj_mask):
        books = {}
        for u in range(n):
            mask = adj_mask[u]
            while mask:
                v = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if u < v:
                    common = adj_mask[u] & adj_mask[v]
                    books[(u, v)] = common.bit_count()
        return books

    def get_max_book(books):
        return max(books.values()) if books else 0

    def get_sum_sq_books(books):
        return sum(b*b for b in books.values())

    # Initialization: Random graph balanced to near-regular degrees
    # Target degrees: 20 or 21. Sum = 802.
    # 38 vertices degree 20, 2 vertices degree 21.
    
    edge_set = set()
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # 1. Randomly pick m edges
    random.shuffle(all_possible)
    for u, v in all_possible[:m]:
        edge_set.add((u, v))
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # 2. Balance degrees
    # Current degrees
    degrees = [adj_mask[i].bit_count() for i in range(n)]
    
    # We want to minimize variance of degrees.
    # Perform swaps: remove (u, v), add (x, y) such that degrees improve.
    # Or simpler: remove edge (u, v) where deg(u) > 20 and deg(v) > 20? No, that reduces edges.
    # We need to maintain m=401.
    # Move an edge: remove (u, v), add (u, w) or (v, w)?
    # If we remove (u, v) and add (u, w):
    # deg(u) unchanged, deg(v) -1, deg(w) +1.
    # This is useful if deg(v) is high and deg(w) is low.
    
    for _ in range(1000):
        if time.time() > deadline - 10: break
        # Find high and low degree vertices
        # Sort indices by degree
        sorted_verts = sorted(range(n), key=lambda i: degrees[i])
        low_idx = sorted_verts[0]
        high_idx = sorted_verts[-1]
        
        if degrees[high_idx] - degrees[low_idx] < 2:
            break
            
        h = high_idx
        l = low_idx
        
        # Try to move an edge from h to l.
        # Remove (h, u), Add (l, u) ? No, that changes deg(h)-1, deg(l)+1, deg(u) unchanged.
        # This is valid if (h, u) is edge and (l, u) is not.
        
        # Find a neighbor u of h such that (l, u) is not an edge.
        moved = False
        # Iterate neighbors of h
        mask_h = adj_mask[h]
        candidates = []
        mask = mask_h
        while mask:
            u = (mask & -mask).bit_length() - 1
            mask &= mask - 1
            if u == l: continue
            if not (adj_mask[l] & (1 << u)):
                candidates.append(u)
                
        if candidates:
            # Pick one randomly
            u = random.choice(candidates)
            # Remove (h, u)
            adj_mask[h] &= ~(1 << u)
            adj_mask[u] &= ~(1 << h)
            edge_set.discard((min(h, u), max(h, u)))
            
            # Add (l, u)
            adj_mask[l] |= (1 << u)
            adj_mask[u] |= (1 << l)
            edge_set.add((min(l, u), max(l, u)))
            
            degrees[h] -= 1
            degrees[l] += 1
            # degrees[u] unchanged
            
    # Check if we have exactly 401 edges
    # The swaps above preserve edge count.
    # Initial pick was 401.
    
    books = compute_books(adj_mask)
    current_max_book = get_max_book(books)
    current_sum_sq = get_sum_sq_books(books)
    
    best_max_book = current_max_book
    best_sum_sq = current_sum_sq
    best_edge_set = set(edge_set)
    best_adj_mask = list(adj_mask)
    
    T = 2.0
    T_min = 0.01
    cooling = 0.999
    
    # Precompute list of all possible edges for non-edge sampling
    # We need a way to quickly sample non-edges.
    # With n=40, total possible 780. Edges 401. Non-edges 379.
    # List of non-edges is small enough to rebuild or filter.
    
    while time.time() < deadline and T > T_min:
        # Select edge to remove
        # Bias towards high book size
        if random.random() < 0.8:
            # Get top 10% edges by book size
            sorted_books = sorted(books.items(), key=lambda x: -x[1])
            top_k = max(1, len(sorted_books) // 10)
            e_remove = random.choice(sorted_books[:top_k])[0]
        else:
            e_remove = random.choice(list(edge_set))
            
        u, v = e_remove
        
        # Select edge to add
        # We want to add an edge that doesn't create large books.
        # Sample random non-edges.
        # To be fast, we can just try a few random pairs.
        e_add = None
        for _ in range(20):
            a = random.randint(0, n-1)
            b = random.randint(0, n-1)
            if a == b: continue
            if a > b: a, b = b, a
            if (a, b) in edge_set: continue
            if (a, b) == e_remove: continue
            
            # Check common neighbors
            common = (adj_mask[a] & adj_mask[b]).bit_count()
            # We prefer low common.
            # But we also need to consider the effect on the rest of the graph?
            # Adding (a, b) increases book size of (a, b) to 'common'.
            # It also increases book size of neighbors?
            # Book size of (x, y) is |N(x) \cap N(y)|.
            # If we add (a, b), does it change book size of (x, y) for x!=a,b?
            # Only if x or y is a or b.
            # So it only affects edges incident to a or b.
            
            # Heuristic: Pick the one with lowest 'common'
            if e_add is None:
                e_add = (a, b)
                best_add_score = common
            else:
                if common < best_add_score:
                    e_add = (a, b)
                    best_add_score = common
                    
        if e_add is None:
            T *= cooling
            continue
            
        a, b = e_add
        
        # Apply move
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        adj_mask[a] |= (1 << b)
        adj_mask[b] |= (1 << a)
        
        edge_set.remove(e_remove)
        edge_set.add(e_add)
        
        # Recompute books
        new_books = compute_books(adj_mask)
        new_max_book = get_max_book(new_books)
        new_sum_sq = get_sum_sq_books(new_books)
        
        # Acceptance
        accept = False
        if new_max_book < current_max_book:
            accept = True
        elif new_max_book == current_max_book:
            if new_sum_sq < current_sum_sq:
                accept = True
            else:
                delta_sq = new_sum_sq - current_sum_sq
                if delta_sq > 0:
                    prob = math.exp(-delta_sq / (T * 50)) # Scale factor for sum_sq
                    if random.random() < prob:
                        accept = True
                else:
                    accept = True
        else:
            delta_max = new_max_book - current_max_book
            prob = math.exp(-delta_max / T)
            if random.random() < prob:
                accept = True
                
        if accept:
            current_max_book = new_max_book
            current_sum_sq = new_sum_sq
            books = new_books
            
            if current_max_book < best_max_book or (current_max_book == best_max_book and current_sum_sq < best_sum_sq):
                best_max_book = current_max_book
                best_sum_sq = current_sum_sq
                best_edge_set = set(edge_set)
                best_adj_mask = list(adj_mask)
        else:
            # Revert
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            adj_mask[a] &= ~(1 << b)
            adj_mask[b] &= ~(1 << a)
            
            edge_set.remove(e_add)
            edge_set.add(e_remove)
            
        T *= cooling
        
    return [list(e) for e in best_edge_set]
