import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 40
    m = 401
    
    # Generate Paley graph on 41 vertices
    # Vertices 0..40
    # Edge (u, v) if (v-u)^2 is a quadratic residue mod 41
    # Quadratic residues mod 41:
    # 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=25, 6^2=36, 7^2=49=8, 8^2=64=23, 9^2=81=40, 10^2=100=18, 11^2=121=39, 12^2=144=21, 13^2=169=5, 14^2=196=32, 15^2=225=20, 16^2=256=10, 17^2=289=2, 18^2=324=37, 19^2=361=33, 20^2=400=31
    # Residues: {1, 2, 4, 5, 8, 9, 10, 16, 18, 20, 21, 23, 25, 31, 32, 33, 36, 37, 39, 40}
    residues = {1, 2, 4, 5, 8, 9, 10, 16, 18, 20, 21, 23, 25, 31, 32, 33, 36, 37, 39, 40}
    
    # Build adjacency for 41 vertices
    adj_41 = [[False]*41 for _ in range(41)]
    for u in range(41):
        for v in range(u+1, 41):
            diff = (v - u) % 41
            if diff in residues:
                adj_41[u][v] = True
                adj_41[v][u] = True
    
    # Remove vertex 0 to get 40 vertices (0..39 in new indexing correspond to 1..40 in old)
    # New vertex i corresponds to old vertex i+1
    adj_mask = [0] * n
    edge_set = set()
    
    for i in range(n):
        old_u = i + 1
        for j in range(i+1, n):
            old_v = j + 1
            if adj_41[old_u][old_v]:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                edge_set.add((i, j))
                
    # Current edges: 390
    # Need 401 edges, so add 11 edges
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def compute_books(adj_mask):
        books = {}
        for u in range(n):
            mask = adj_mask[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                mask ^= lsb
                if u < v:
                    books[(u, v)] = (adj_mask[u] & adj_mask[v]).bit_count()
        return books
        
    def get_metrics(books):
        if not books:
            return 0, 0
        max_book = max(books.values())
        sum_sq = sum(b * b for b in books.values())
        return max_book, sum_sq

    books = compute_books(adj_mask)
    current_max_book, current_sum_sq = get_metrics(books)
    
    best_max_book = current_max_book
    best_sum_sq = current_sum_sq
    best_edge_set = set(edge_set)
    best_adj_mask = list(adj_mask)
    
    non_edges = [e for e in all_edges if e not in edge_set]
    
    # Add 11 edges greedily to minimize increase in max book
    # Sort non-edges by common neighbors count
    def common_neighbors_count(e, adj_mask):
        u, v = e
        return (adj_mask[u] & adj_mask[v]).bit_count()
        
    for _ in range(11):
        if not non_edges:
            break
        # Find non-edge with min common neighbors
        best_e = None
        best_c = float('inf')
        # Sample a subset for speed if non_edges is large, but it's small (795-390=405)
        for e in non_edges:
            c = common_neighbors_count(e, adj_mask)
            if c < best_c:
                best_c = c
                best_e = e
                
        if best_e:
            u, v = best_e
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            edge_set.add(best_e)
            non_edges.remove(best_e)
            
    # Now we have 401 edges
    books = compute_books(adj_mask)
    current_max_book, current_sum_sq = get_metrics(books)
    
    best_max_book = current_max_book
    best_sum_sq = current_sum_sq
    best_edge_set = set(edge_set)
    best_adj_mask = list(adj_mask)
    
    # Simulated Annealing
    T = 2.0
    T_min = 0.01
    cooling = 0.9998
    
    def get_top_edges(books, k):
        if k >= len(books):
            return list(books.keys())
        sorted_books = sorted(books.items(), key=lambda x: -x[1])
        return [e for e, b in sorted_books[:k]]
        
    while time.time() < deadline and T > T_min:
        # Select edge to remove
        if random.random() < 0.7:
            top_k = max(1, len(books) // 5)
            top_edges = get_top_edges(books, top_k)
            e_remove = random.choice(top_edges)
        else:
            e_remove = random.choice(list(edge_set))
            
        u, v = e_remove
        
        if not non_edges:
            break
            
        # Select edge to add
        # Prefer edges with low common neighbors
        sample_size = min(50, len(non_edges))
        candidates = random.sample(non_edges, sample_size)
        
        best_add = None
        best_add_score = float('inf')
        
        for e in candidates:
            a, b = e
            score = (adj_mask[a] & adj_mask[b]).bit_count()
            if score < best_add_score:
                best_add_score = score
                best_add = e
                
        e_add = best_add
        
        if e_add is None or e_add == e_remove or e_add in edge_set:
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
        
        non_edges.remove(e_add)
        non_edges.append(e_remove)
        
        new_books = compute_books(adj_mask)
        new_max_book, new_sum_sq = get_metrics(new_books)
        
        accept = False
        if new_max_book < current_max_book:
            accept = True
        elif new_max_book == current_max_book:
            if new_sum_sq <= current_sum_sq:
                accept = True
            else:
                delta_sq = new_sum_sq - current_sum_sq
                prob = math.exp(-delta_sq / (T * 50))
                if random.random() < prob:
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
            
            non_edges.remove(e_remove)
            non_edges.append(e_add)
            
        T *= cooling
        
    return [list(e) for e in best_edge_set]
