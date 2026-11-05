import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    random.seed(42)
    
    # Bitmask representation for fast intersection
    # adj_mask[i] is a 40-bit integer where bit j is set if edge (i,j) exists
    
    def random_graph_with_m_edges(m):
        """Generate a random graph with exactly m edges."""
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        chosen = all_edges[:m]
        edge_set = set()
        adj_mask = [0] * n
        for u, v in chosen:
            edge_set.add((u, v))
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        return edge_set, adj_mask

    def compute_all_books(edge_set, adj_mask):
        """Compute book size for each edge using bitmask intersection."""
        books = {}
        for (u, v) in edge_set:
            common = adj_mask[u] & adj_mask[v]
            books[(u, v)] = bin(common).count('1')
        return books

    def get_max_book(books):
        return max(books.values()) if books else 0

    # Initialize with a random graph
    edge_set, adj_mask = random_graph_with_m_edges(401)
    books = compute_all_books(edge_set, adj_mask)
    
    # Maintain frequency of book sizes for O(1) max lookup
    max_possible_book = n - 2
    book_freq = [0] * (max_possible_book + 1)
    current_max_book = 0
    
    def update_freq(old_val, new_val):
        nonlocal current_max_book
        if old_val is not None:
            book_freq[old_val] -= 1
            if book_freq[old_val] == 0 and old_val == current_max_book:
                # Need to find new max
                while current_max_book > 0 and book_freq[current_max_book] == 0:
                    current_max_book -= 1
        if new_val is not None:
            book_freq[new_val] += 1
            if new_val > current_max_book:
                current_max_book = new_val

    # Initialize frequency table
    for val in books.values():
        book_freq[val] += 1
    current_max_book = max(books.values()) if books else 0
    
    best_book = current_max_book
    best_edge_set = set(edge_set)
    best_adj_mask = list(adj_mask)
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    all_edges_set = set(all_edges)
    
    def local_search(edge_set, adj_mask, books, book_freq, current_max_book):
        """
        Perform local search to minimize max book.
        Returns updated state.
        """
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to find an improving move
            # We look for a swap (remove e1, add e2) that reduces max book
            # Or at least doesn't increase it, and if max is reduced, accept.
            
            # To be efficient, we only consider removing edges that are "high" (part of max book)
            # or adding edges that create low book sizes.
            
            # Get edges with current max book
            max_edges = [e for e, b in books.items() if b == current_max_book]
            
            if not max_edges:
                break
                
            random.shuffle(max_edges)
            
            # Try removing one of the max edges and adding a random other edge
            for e_remove in max_edges:
                if time.time() > deadline:
                    break
                
                u, v = e_remove
                old_book_rm = books[e_remove]
                
                # Find candidate edges to add
                # We want an edge e_add such that the new max book is < current_max_book
                # The new max book will be max(
                #   max book of unaffected edges (which is <= current_max_book),
                #   max book of affected edges after swap
                # )
                
                # If we remove an edge with book == current_max_book, and no other edge has book == current_max_book,
                # then the new max will be determined by affected edges.
                # If there are other edges with book == current_max_book, we must ensure affected edges don't exceed that,
                # and ideally we reduce the count of current_max_book.
                
                count_max = book_freq[current_max_book]
                
                # If count_max == 1, removing this edge drops the potential max to current_max_book - 1 (unless affected edges go up)
                # If count_max > 1, we just need to ensure we don't create a book > current_max_book.
                
                candidates = []
                # Sample some random edges to add
                # Instead of checking all, we can check a subset
                # But for n=40, total edges ~800. Checking all is fast enough?
                # 800 swaps * 401 edges * cost of swap.
                # Cost of swap is O(1) if we just compute affected books.
                # Let's try checking all non-edges.
                
                non_edges = [e for e in all_edges if e not in edge_set]
                random.shuffle(non_edges)
                
                # To speed up, we can just try a few random non-edges first
                # Or iterate through all if time permits
                found_improvement = False
                
                for e_add in non_edges:
                    if e_add == e_remove:
                        continue
                    
                    a, b = e_add
                    
                    # Calculate delta for affected edges
                    # Affected vertices: u, v, a, b
                    # Affected edges: all edges incident to u, v, a, b
                    
                    # We need to calculate new book sizes for affected edges.
                    # Let's do a dry run.
                    
                    # Save old books for affected edges
                    affected_edges = set()
                    for x in [u, v, a, b]:
                        mask = adj_mask[x]
                        m = mask
                        while m:
                            y = (m & -m).bit_length() - 1
                            m &= m - 1
                            e = (min(x, y), max(x, y))
                            if e in edge_set or e == e_add:
                                affected_edges.add(e)
                    
                    # Apply move to temporary masks?
                    # Or just calculate new books directly using current masks + modifications?
                    # Direct calculation is safer.
                    
                    # New adj_mask logic:
                    # u-v removed, a-b added
                    
                    # For an edge (x,y) in affected_edges:
                    # new_common = (adj_mask[x] & adj_mask[y]) 
                    #   XOR (bit v if x==u or y==u) 
                    #   XOR (bit u if x==v or y==v)
                    #   XOR (bit b if x==a or y==a)
                    #   XOR (bit a if x==b or y==b)
                    
                    # This is getting complex. Let's just modify masks, compute, revert.
                    
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
                    adj_mask[a] |= (1 << b)
                    adj_mask[b] |= (1 << a)
                    
                    new_max_affected = 0
                    new_books_map = {}
                    
                    for e in affected_edges:
                        if e == e_remove:
                            continue
                        x, y = e
                        if e == e_add or e in edge_set:
                            common = adj_mask[x] & adj_mask[y]
                            nb = bin(common).count('1')
                            new_books_map[e] = nb
                            if nb > new_max_affected:
                                new_max_affected = nb
                    
                    # Revert masks
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    adj_mask[a] &= ~(1 << b)
                    adj_mask[b] &= ~(1 << a)
                    
                    # Determine new global max
                    # Unaffected edges have book <= current_max_book
                    # If we removed the *only* edge with book == current_max_book, 
                    # and new_max_affected < current_max_book, then new global max < current_max_book.
                    
                    # Let's check if this move reduces the global max.
                    # Current global max is current_max_book.
                    # New global max is max(
                    #   max book of unaffected edges,
                    #   new_max_affected
                    # )
                    
                    # Max book of unaffected edges:
                    # If count_max == 1 and e_remove was the only one, then max_unaffected = current_max_book - 1 (or lower).
                    # If count_max > 1, max_unaffected = current_max_book.
                    
                    if count_max == 1:
                        # Potential for improvement
                        if new_max_affected < current_max_book:
                            # This is an improvement!
                            # Apply move
                            edge_set.remove(e_remove)
                            edge_set.add(e_add)
                            
                            # Update books and freq
                            for e in affected_edges:
                                if e == e_remove:
                                    if e in books:
                                        update_freq(books[e], None)
                                        del books[e]
                                elif e in new_books_map:
                                    old_val = books.get(e)
                                    new_val = new_books_map[e]
                                    if old_val != new_val:
                                        update_freq(old_val, new_val)
                                    books[e] = new_val
                                else:
                                    # Edge exists but not in new_books_map? Should not happen for affected
                                    # But if e is in edge_set and not in affected_edges? 
                                    # affected_edges includes all incident to u,v,a,b.
                                    # So if e is incident, it's in affected_edges.
                                    pass
                            
                            if e_add not in books:
                                books[e_add] = new_books_map.get(e_add, 0)
                                update_freq(None, books[e_add])
                            else:
                                # Already handled in loop if e_add was in affected_edges
                                pass
                            
                            current_max_book = max(books.values()) if books else 0
                            # Actually, we can update freq properly
                            # Let's just recalc current_max_book from freq to be safe
                            while current_max_book > 0 and book_freq[current_max_book] == 0:
                                current_max_book -= 1
                            
                            improved = True
                            found_improvement = True
                            break
                    else:
                        # count_max > 1
                        # We can only improve if we reduce the count of current_max_book
                        # and don't create any new current_max_book or higher.
                        # If new_max_affected < current_max_book, we reduce count by 1.
                        # This doesn't reduce the MAX value, but reduces the number of max edges.
                        # This is a neutral move for the objective (max value), but might help later.
                        # However, for "steepest descent" on MAX VALUE, this is not an improvement.
                        # We can accept it with some probability or just ignore for now.
                        # Let's ignore for now to keep it pure local search on the value.
                        pass
                
                if found_improvement:
                    break
            
            if not improved:
                # No single swap improves the max book value.
                # Try swaps that keep max book value same but reduce count of max edges?
                # Or just stop local search and perturb.
                break
        
        return edge_set, adj_mask, books, book_freq, current_max_book

    # ILS Loop
    num_restarts = 0
    max_restarts = 100
    
    while time.time() < deadline and num_restarts < max_restarts:
        # Local Search
        edge_set, adj_mask, books, book_freq, current_max_book = local_search(
            edge_set, adj_mask, books, book_freq, current_max_book
        )
        
        if current_max_book < best_book:
            best_book = current_max_book
            best_edge_set = set(edge_set)
            best_adj_mask = list(adj_mask)
        
        if time.time() > deadline:
            break
        
        # Perturbation
        # Randomly swap K edges
        K = 5
        for _ in range(K):
            # Pick a random edge to remove
            e_remove = random.choice(list(edge_set))
            # Pick a random edge to add
            e_add = random.choice(all_edges)
            if e_add in edge_set or e_add == e_remove:
                continue
            
            u, v = e_remove
            a, b = e_add
            
            # Apply
            edge_set.remove(e_remove)
            edge_set.add(e_add)
            
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            adj_mask[a] |= (1 << b)
            adj_mask[b] |= (1 << a)
            
            # Update books
            affected_edges = set()
            for x in [u, v, a, b]:
                mask = adj_mask[x]
                m = mask
                while m:
                    y = (m & -m).bit_length() - 1
                    m &= m - 1
                    e = (min(x, y), max(x, y))
                    if e in edge_set or e == e_add:
                        affected_edges.add(e)
            
            for e in affected_edges:
                if e == e_remove:
                    if e in books:
                        update_freq(books[e], None)
                        del books[e]
                else:
                    x, y = e
                    if e in edge_set:
                        new_val = bin(adj_mask[x] & adj_mask[y]).count('1')
                        old_val = books.get(e)
                        if old_val != new_val:
                            update_freq(old_val, new_val)
                        books[e] = new_val
            
            # Update current_max_book
            while current_max_book > 0 and book_freq[current_max_book] == 0:
                current_max_book -= 1
        
        num_restarts += 1
    
    return [list(e) for e in best_edge_set]
