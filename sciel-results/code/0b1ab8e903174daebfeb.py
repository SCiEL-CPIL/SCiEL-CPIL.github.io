import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 40
    target_edges = 401
    
    # Structural Initialization: K_{9,31} + sparse intra-B edges
    A = list(range(9))
    B = list(range(9, 40))
    B1 = list(range(9, 24))   # 15 vertices
    B2 = list(range(24, 40))   # 16 vertices
    
    edges = set()
    
    # 1. Add all edges between A and B (279 edges)
    for a in A:
        for b in B:
            edges.add((a, b))
    
    # 2. Add 122 edges within B using greedy construction to minimize book size
    needed = 122
    added = 0
    
    # Bitmask adjacency for quick common neighbor calculation
    adj_masks = [0] * n
    
    def update_adj(u, v, add):
        if add:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        else:
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
    
    for e in edges:
        update_adj(e[0], e[1], True)
    
    def get_book_size(u, v):
        return (adj_masks[u] & adj_masks[v]).bit_count()
    
    # Greedy: add intra-B edges one by one, choosing the pair that minimizes
    # the maximum book size increase
    # Generate candidate pairs between B1 and B2
    pairs = [(u, v) for u in B1 for v in B2]
    random.shuffle(pairs)
    
    # Use a priority approach: try to add edges that create small books
    # Track degree within B to keep it balanced
    deg_b = {v: 0 for v in B}
    
    # Sort pairs by a heuristic: prefer pairs where both endpoints have low degree
    # and low current common neighbors
    def pair_score(u, v):
        # Estimate book size if we add (u,v)
        # Common neighbors = common in A (always 9) + common in B
        common_b = (adj_masks[u] & adj_masks[v]).bit_count()
        return common_b + deg_b[u] + deg_b[v]
    
    # Greedy selection: repeatedly pick the best available pair
    remaining = list(pairs)
    while added < needed and remaining:
        # Find the pair with minimum estimated book size
        best_idx = 0
        best_score = float('inf')
        # Sample a subset for speed if list is large
        sample_size = min(len(remaining), 100)
        sample_indices = random.sample(range(len(remaining)), sample_size) if len(remaining) > sample_size else range(len(remaining))
        
        for idx in sample_indices:
            u, v = remaining[idx]
            if deg_b[u] >= 7 or deg_b[v] >= 7:
                continue
            s = pair_score(u, v)
            if s < best_score:
                best_score = s
                best_idx = idx
        
        u, v = remaining.pop(best_idx)
        edges.add((u, v))
        update_adj(u, v, True)
        deg_b[u] += 1
        deg_b[v] += 1
        added += 1
    
    # Fallback if we didn't reach target
    if added < needed:
        all_pairs = [(u, v) for u in B1 for v in B2 if (u, v) not in edges]
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if added >= needed:
                break
            edges.add((u, v))
            update_adj(u, v, True)
            added += 1
    
    # Local Search Refinement
    def get_max_book_info():
        max_b = -1
        max_edges = []
        for u, v in edges:
            b = get_book_size(u, v)
            if b > max_b:
                max_b = b
                max_edges = [(u, v)]
            elif b == max_b:
                max_edges.append((u, v))
        return max_b, max_edges
    
    current_max_book, current_max_edges = get_max_book_info()
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if current_max_book <= 7:
            break
        if not current_max_edges:
            break
        if current_max_book == 8 and no_improve_count > 200:
            break
        
        best_swap = None
        
        # Try multiple candidate edges to remove
        num_candidates = min(5, len(current_max_edges))
        candidates = random.sample(current_max_edges, num_candidates)
        
        for (u, v) in candidates:
            # Try to find a good non-edge to add
            best_new_max = float('inf')
            best_x, best_y = -1, -1
            best_new_max_edges = []
            
            # Sample non-edges for speed
            non_edges_sampled = []
            for _ in range(200):
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x >= y:
                    continue
                if (x, y) in edges:
                    continue
                non_edges_sampled.append((x, y))
            
            for x, y in non_edges_sampled:
                # Skip if adding (x,y) would create a triangle with removed edge (u,v)
                # i.e., if x-u and y-v or x-v and y-u exist
                if ((1 << u) & adj_masks[x]) and ((1 << v) & adj_masks[y]):
                    continue
                if ((1 << v) & adj_masks[x]) and ((1 << u) & adj_masks[y]):
                    continue
                
                # Perform trial swap
                edges.remove((u, v))
                update_adj(u, v, False)
                edges.add((x, y))
                update_adj(x, y, True)
                
                new_max, new_max_edges = get_max_book_info()
                
                # Rollback
                edges.add((u, v))
                update_adj(u, v, True)
                edges.remove((x, y))
                update_adj(x, y, False)
                
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_x, best_y = x, y
                    best_new_max_edges = new_max_edges
                elif new_max == best_new_max and len(new_max_edges) < len(best_new_max_edges):
                    best_x, best_y = x, y
                    best_new_max_edges = new_max_edges
            
            if best_x == -1:
                continue
            
            if best_swap is None:
                best_swap = (best_new_max, best_new_max_edges, u, v, best_x, best_y)
            else:
                if best_new_max < best_swap[0]:
                    best_swap = (best_new_max, best_new_max_edges, u, v, best_x, best_y)
                elif best_new_max == best_swap[0] and len(best_new_max_edges) < len(best_swap[1]):
                    best_swap = (best_new_max, best_new_max_edges, u, v, best_x, best_y)
        
        if best_swap is None:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
            continue
        
        new_max, new_max_edges, u, v, best_x, best_y = best_swap
        old_max = current_max_book
        old_max_count = len(current_max_edges)
        
        # Acceptance
        accepted = False
        if new_max < old_max:
            accepted = True
            no_improve_count = 0
        elif new_max == old_max:
            if len(new_max_edges) < old_max_count:
                accepted = True
                no_improve_count = 0
            elif random.random() < 0.05:
                accepted = True
        else:
            accepted = False
        
        if accepted:
            edges.remove((u, v))
            update_adj(u, v, False)
            edges.add((best_x, best_y))
            update_adj(best_x, best_y, True)
            current_max_book = new_max
            current_max_edges = new_max_edges
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    result = [[u, v] for u, v in edges]
    return result
