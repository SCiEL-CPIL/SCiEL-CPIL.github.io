import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n = 40
    m = 401
    random.seed(42)
    
    adj_mask = [0] * n
    
    def compute_books(adj_mask):
        """Compute book size for each edge using bitmask intersection."""
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

    def structured_init(k):
        """
        Construct graph: K20,20 minus k edges, plus k+1 internal edges.
        Try to minimize max book by aligning deficits.
        """
        adj_mask = [0] * n
        edge_set = set()
        
        set_a = list(range(20))
        set_b = list(range(20, 40))
        
        # Start with K20,20
        for u in set_a:
            for v in set_b:
                edge_set.add((u, v))
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        
        # Remove k edges
        removed = []
        for _ in range(k):
            u = random.choice(set_a)
            v = random.choice(set_b)
            if (u, v) in edge_set:
                edge_set.remove((u, v))
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                removed.append((u, v))
        
        # Add k+1 internal edges
        # To minimize book size, internal edges should involve vertices with missing neighbors
        # Simple heuristic: pick pairs from A and B that include the removed vertices
        # For now, random selection, but prefer vertices in 'removed'
        
        internal_edges = []
        # We need to add k+1 edges.
        # Let's pick them randomly but ensure we use the 'deficit' vertices if possible.
        
        # Shuffle sets to randomize
        a_candidates = set_a[:]
        b_candidates = set_b[:]
        
        # Prioritize vertices involved in removed edges
        a_deficit = [u for u, v in removed]
        b_deficit = [v for u, v in removed]
        
        random.shuffle(a_deficit)
        random.shuffle(b_deficit)
        
        added = 0
        edges_to_add = k + 1
        
        # Add edges involving deficit vertices first
        while added < edges_to_add:
            if a_deficit and len(a_deficit) >= 2:
                u, v = a_deficit.pop(0), a_deficit.pop(0)
                if (u, v) not in edge_set and u != v:
                    edge_set.add((min(u,v), max(u,v)))
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    added += 1
            elif b_deficit and len(b_deficit) >= 2:
                u, v = b_deficit.pop(0), b_deficit.pop(0)
                if (u, v) not in edge_set and u != v:
                    edge_set.add((min(u,v), max(u,v)))
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    added += 1
            else:
                # Fallback to random
                if a_deficit:
                    u = a_deficit.pop(0)
                    v = random.choice(set_a)
                    if (u, v) not in edge_set and u != v:
                        edge_set.add((min(u,v), max(u,v)))
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        added += 1
                elif b_deficit:
                    u = b_deficit.pop(0)
                    v = random.choice(set_b)
                    if (u, v) not in edge_set and u != v:
                        edge_set.add((min(u,v), max(u,v)))
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        added += 1
                else:
                    # No deficits left, just random
                    if random.random() < 0.5:
                        u, v = random.sample(set_a, 2)
                    else:
                        u, v = random.sample(set_b, 2)
                    u, v = min(u, v), max(u, v)
                    if (u, v) not in edge_set:
                        edge_set.add((u, v))
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        added += 1
                    else:
                        # Collision, try again
                        pass
        
        return edge_set, adj_mask

    # Try multiple initializations
    best_edge_set = None
    best_adj_mask = None
    best_max_book = float('inf')
    best_sum_sq = float('inf')
    
    # Try structured inits with different k
    for k in range(0, 10):
        for attempt in range(5):
            edge_set, adj_mask = structured_init(k)
            books = compute_books(adj_mask)
            current_max_book = get_max_book(books)
            current_sum_sq = get_sum_sq_books(books)
            
            if current_max_book < best_max_book or (current_max_book == best_max_book and current_sum_sq < best_sum_sq):
                best_max_book = current_max_book
                best_sum_sq = current_sum_sq
                best_edge_set = set(edge_set)
                best_adj_mask = list(adj_mask)
                
                if best_max_book <= 7:
                    return [list(e) for e in best_edge_set]

    edge_set = best_edge_set
    adj_mask = best_adj_mask
    books = compute_books(adj_mask)
    current_max_book = best_max_book
    current_sum_sq = best_sum_sq
    
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Local Search
    # Use incremental book updates for speed
    # For simplicity and safety within time limits, we stick to full recompute but limit iterations
    # Or implement a fast check for the swap
    
    def try_swap(e1, e2):
        """
        Try swapping edge e1 (remove) and e2 (add).
        Returns (new_max_book, new_sum_sq) if valid and better, else None.
        """
        u1, v1 = e1
        u2, v2 = e2
        
        if (u2, v2) in edge_set or (v2, u2) in edge_set:
            return None
        if e1 == e2:
            return None
            
        # Save old masks
        om_u1 = adj_mask[u1]
        om_v1 = adj_mask[v1]
        om_u2 = adj_mask[u2]
        om_v2 = adj_mask[v2]
        
        # Apply
        adj_mask[u1] &= ~(1 << v1)
        adj_mask[v1] &= ~(1 << u1)
        adj_mask[u2] |= (1 << v2)
        adj_mask[v2] |= (1 << u2)
        
        edge_set.remove(e1)
        edge_set.add((u2, v2))
        
        # Recompute books (expensive)
        new_books = compute_books(adj_mask)
        new_max = get_max_book(new_books)
        new_sq = get_sum_sq_books(new_books)
        
        # Revert if not better (or always revert here and let caller decide? 
        # Better to revert here if we don't accept, but we need to compare first)
        # We will revert in the main loop if rejected.
        # For now, just return metrics.
        
        return new_max, new_sq, new_books

    # Simulated Annealing
    T = 5.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline and T > T_min:
        # Pick an edge to remove
        # Bias towards high book edges
        sorted_books = sorted(books.items(), key=lambda x: -x[1])
        if random.random() < 0.7:
            top_k = max(1, len(sorted_books) // 5)
            e_remove = random.choice(sorted_books[:top_k])[0]
        else:
            e_remove = random.choice(list(edge_set))
            
        # Pick an edge to add
        # Sample candidates
        candidates = []
        for _ in range(20):
            a, b = random.choice(all_possible)
            a, b = min(a, b), max(a, b)
            if (a, b) not in edge_set and (a, b) != e_remove:
                candidates.append((a, b))
        
        if not candidates:
            T *= cooling
            continue
            
        # Pick best candidate (min common neighbors)
        best_cand = None
        min_common = 100
        for a, b in candidates:
            common = (adj_mask[a] & adj_mask[b]).bit_count()
            if common < min_common:
                min_common = common
                best_cand = (a, b)
                
        if not best_cand:
            T *= cooling
            continue
            
        # Perform swap
        u1, v1 = e_remove
        u2, v2 = best_cand
        
        # Save state
        old_books = dict(books)
        
        adj_mask[u1] &= ~(1 << v1)
        adj_mask[v1] &= ~(1 << u1)
        adj_mask[u2] |= (1 << v2)
        adj_mask[v2] |= (1 << u2)
        
        edge_set.remove(e_remove)
        edge_set.add(best_cand)
        
        new_books = compute_books(adj_mask)
        new_max = get_max_book(new_books)
        new_sq = get_sum_sq_books(new_books)
        
        # Acceptance
        accept = False
        if new_max < current_max_book:
            accept = True
        elif new_max == current_max_book:
            if new_sq < current_sum_sq:
                accept = True
            else:
                if random.random() < math.exp(-(new_sq - current_sum_sq) / (T * 100)):
                    accept = True
        else:
            if random.random() < math.exp(-(new_max - current_max_book) / T):
                accept = True
                
        if accept:
            books = new_books
            current_max_book = new_max
            current_sum_sq = new_sq
            
            if current_max_book < best_max_book or (current_max_book == best_max_book and current_sum_sq < best_sum_sq):
                best_max_book = current_max_book
                best_sum_sq = current_sum_sq
                best_edge_set = set(edge_set)
                best_adj_mask = list(adj_mask)
                
                if best_max_book <= 7:
                    return [list(e) for e in best_edge_set]
        else:
            # Revert
            adj_mask[u1] |= (1 << v1)
            adj_mask[v1] |= (1 << u1)
            adj_mask[u2] &= ~(1 << v2)
            adj_mask[v2] &= ~(1 << u2)
            
            edge_set.remove(best_cand)
            edge_set.add(e_remove)
            books = old_books
            
        T *= cooling
        
    return [list(e) for e in best_edge_set]
