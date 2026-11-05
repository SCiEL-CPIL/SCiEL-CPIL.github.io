import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 40
    m = 401
    
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
    
    def near_regular_init(n, m, seed):
        random.seed(seed)
        adj_mask = [0] * n
        edge_set = set()
        
        # Start with a random graph and balance degrees
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        
        for u, v in all_possible[:m]:
            edge_set.add((u, v))
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Balance degrees via 2-opt swaps
        degrees = [adj_mask[i].bit_count() for i in range(n)]
        target_deg = m * 2 // n  # ~20
        
        for _ in range(500):
            max_idx = max(range(n), key=lambda i: degrees[i])
            min_idx = min(range(n), key=lambda i: degrees[i])
            
            if degrees[max_idx] - degrees[min_idx] < 2:
                break
            
            moved = False
            neigh_max = [j for j in range(n) if (adj_mask[max_idx] >> j) & 1]
            random.shuffle(neigh_max)
            
            for u in neigh_max:
                if u == min_idx:
                    continue
                if not ((adj_mask[min_idx] >> u) & 1):
                    adj_mask[max_idx] &= ~(1 << u)
                    adj_mask[u] &= ~(1 << max_idx)
                    edge_set.discard((min(max_idx, u), max(max_idx, u)))
                    
                    adj_mask[min_idx] |= (1 << u)
                    adj_mask[u] |= (1 << min_idx)
                    edge_set.add((min(min_idx, u), max(min_idx, u)))
                    
                    degrees[max_idx] -= 1
                    degrees[min_idx] += 1
                    moved = True
                    break
            
            if not moved:
                break
        
        return edge_set, adj_mask
    
    def bipartite_init(n, m, seed):
        random.seed(seed)
        adj_mask = [0] * n
        edge_set = set()
        
        set_a = list(range(20))
        set_b = list(range(20, 40))
        
        for u in set_a:
            for v in set_b:
                edge_set.add((u, v))
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        
        # Add 1 internal edge - pick one that minimizes impact
        # Any internal edge in A or B gives book size 20
        # Let's try adding it and see if we can fix it
        u, v = random.sample(set_a, 2)
        u, v = min(u, v), max(u, v)
        edge_set.add((u, v))
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
        return edge_set, adj_mask
    
    def local_search(edge_set, adj_mask, books, max_book, sum_sq, time_limit):
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        best_max = max_book
        best_sum_sq = sum_sq
        best_edge_set = set(edge_set)
        best_adj_mask = list(adj_mask)
        
        T = 5.0
        T_min = 0.01
        cooling = 0.99995
        
        while time.time() < time_limit and T > T_min:
            sorted_books = sorted(books.items(), key=lambda x: -x[1])
            
            # Pick edge to remove: weighted toward high-book edges
            if random.random() < 0.7:
                top_k = max(1, len(sorted_books) // 5)
                e_remove = random.choice(sorted_books[:top_k])[0]
            elif random.random() < 0.5:
                e_remove = random.choice(sorted_books)
            else:
                e_remove = random.choice(list(edge_set))
            
            u, v = e_remove
            u, v = min(u, v), max(u, v)
            
            # Find candidate edges to add
            candidates = []
            for _ in range(30):
                a, b = random.choice(all_possible)
                a, b = min(a, b), max(a, b)
                if (a, b) not in edge_set and (a, b) != e_remove:
                    common = (adj_mask[a] & adj_mask[b]).bit_count()
                    candidates.append((common, (a, b)))
            
            if not candidates:
                T *= cooling
                continue
            
            candidates.sort(key=lambda x: x[0])
            # Pick from top candidates with some randomness
            rcl_size = max(1, min(5, len(candidates)))
            best_cand = random.choice(candidates[:rcl_size])
            a, b = best_cand[1]
            
            # Compute delta in max book and sum_sq
            affected = {u, v, a, b}
            
            # Save old books for affected edges
            old_affected_books = {}
            for e in edge_set:
                x, y = e
                if x in affected or y in affected:
                    old_affected_books[e] = books[e]
            
            # Apply move
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            adj_mask[a] |= (1 << b)
            adj_mask[b] |= (1 << a)
            
            edge_set.discard(e_remove)
            edge_set.add((a, b))
            
            # Update books for affected edges
            new_affected_books = {}
            for e in edge_set:
                x, y = e
                if x in affected or y in affected:
                    common = adj_mask[x] & adj_mask[y]
                    new_affected_books[e] = common.bit_count()
            
            # Update global books
            for e in old_affected_books:
                if e in new_affected_books:
                    books[e] = new_affected_books[e]
                else:
                    del books[e]
            for e in new_affected_books:
                if e not in old_affected_books:
                    books[e] = new_affected_books[e]
            
            new_max = get_max_book(books)
            new_sum_sq = get_sum_sq_books(books)
            
            # Acceptance
            accept = False
            if new_max < max_book:
                accept = True
            elif new_max == max_book:
                if new_sum_sq < sum_sq:
                    accept = True
                else:
                    if random.random() < math.exp(-(new_sum_sq - sum_sq) / (T * 500)):
                        accept = True
            else:
                if random.random() < math.exp(-(new_max - max_book) / T):
                    accept = True
            
            if accept:
                max_book = new_max
                sum_sq = new_sum_sq
                
                if max_book < best_max or (max_book == best_max and sum_sq < best_sum_sq):
                    best_max = max_book
                    best_sum_sq = sum_sq
                    best_edge_set = set(edge_set)
                    best_adj_mask = list(adj_mask)
                    
                    if best_max <= 7:
                        return best_edge_set, best_adj_mask, best_max, best_sum_sq
            else:
                # Revert
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                adj_mask[a] &= ~(1 << b)
                adj_mask[b] &= ~(1 << a)
                
                edge_set.discard((a, b))
                edge_set.add(e_remove)
                
                for e in new_affected_books:
                    if e in old_affected_books:
                        books[e] = old_affected_books[e]
                    else:
                        del books[e]
                for e in old_affected_books:
                    if e not in new_affected_books:
                        books[e] = old_affected_books[e]
            
            T *= cooling
        
        return best_edge_set, best_adj_mask, best_max, best_sum_sq
    
    # Try multiple initializations
    best_overall_max = float('inf')
    best_overall_sum_sq = float('inf')
    best_overall_edges = None
    
    init_configs = [
        ('bipartite', 42),
        ('bipartite', 123),
        ('near_reg', 42),
        ('near_reg', 777),
        ('near_reg', 2024),
    ]
    
    for init_type, seed in init_configs:
        if time.time() > deadline - 30:
            break
        
        if init_type == 'bipartite':
            edge_set, adj_mask = bipartite_init(n, m, seed)
        else:
            edge_set, adj_mask = near_regular_init(n, m, seed)
        
        books = compute_books(adj_mask)
        max_book = get_max_book(books)
        sum_sq = get_sum_sq_books(books)
        
        # Run local search with remaining time
        remaining = deadline - time.time() - 5
        if remaining < 10:
            if max_book < best_overall_max or (max_book == best_overall_max and sum_sq < best_overall_sum_sq):
                best_overall_max = max_book
                best_overall_sum_sq = sum_sq
                best_overall_edges = set(edge_set)
            continue
        
        edge_set, adj_mask, max_book, sum_sq = local_search(
            edge_set, adj_mask, books, max_book, sum_sq, deadline - 5
        )
        
        if max_book < best_overall_max or (max_book == best_overall_max and sum_sq < best_overall_sum_sq):
            best_overall_max = max_book
            best_overall_sum_sq = sum_sq
            best_overall_edges = set(edge_set)
    
    return [list(e) for e in best_overall_edges]
