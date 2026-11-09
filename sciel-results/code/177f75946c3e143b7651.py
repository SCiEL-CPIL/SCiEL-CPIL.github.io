import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin from 150s limit
    
    n = 40
    target_edges = 401
    
    def build_graph(seed_val):
        """
        Builds a graph with A(9) + B(31), complete bipartite A-B,
        and 122 edges within B chosen to minimize max book.
        Uses a greedy approach with degree constraints and triangle avoidance.
        """
        random.seed(seed_val)
        edges = set()
        
        A = list(range(9))
        B = list(range(9, 40))
        
        # 1. Add all edges between A and B (279 edges)
        for a in A:
            for b in B:
                edges.add((a, b))
                
        # 2. Add 122 edges within B
        # Split B into B1 (15) and B2 (16) to enable bipartite-like structure
        B1 = list(range(9, 24))   # 15 vertices
        B2 = list(range(24, 40))  # 16 vertices
        
        needed = 122
        added = 0
        
        # Track degrees within B to control common neighbors
        deg_in_b = {v: 0 for v in B}
        
        # Generate candidate pairs: prefer B1-B2 (no triangles within B), 
        # then B1-B1, B2-B2 with triangle check
        b1_b2_pairs = [(u, v) for u in B1 for v in B2]
        b1_b1_pairs = [(u, v) for u in B1 for i, v in enumerate(B1) if i > 0]
        b2_b2_pairs = [(u, v) for u in B2 for i, v in enumerate(B2) if i > 0]
        
        random.shuffle(b1_b2_pairs)
        random.shuffle(b1_b1_pairs)
        random.shuffle(b2_b2_pairs)
        
        # Adjacency set for quick lookup
        adj = set()
        for u, v in edges:
            adj.add((u, v))
            adj.add((v, u))
        
        def has_triangle(u, v):
            """Check if adding edge (u,v) creates a triangle."""
            for w in range(n):
                if w == u or w == v:
                    continue
                if (u, w) in adj and (v, w) in adj:
                    return True
            return False
        
        max_deg = 8
        
        # Phase 1: Add B1-B2 edges (triangle-free within B)
        for u, v in b1_b2_pairs:
            if added == needed:
                break
            if deg_in_b[u] < max_deg and deg_in_b[v] < max_deg:
                edges.add((u, v))
                adj.add((u, v))
                adj.add((v, u))
                deg_in_b[u] += 1
                deg_in_b[v] += 1
                added += 1
        
        # Phase 2: If still need more, add B1-B1 or B2-B2 edges with triangle check
        if added < needed:
            remaining_pairs = b1_b1_pairs + b2_b2_pairs
            random.shuffle(remaining_pairs)
            for u, v in remaining_pairs:
                if added == needed:
                    break
                if deg_in_b[u] < max_deg and deg_in_b[v] < max_deg:
                    if not has_triangle(u, v):
                        edges.add((u, v))
                        adj.add((u, v))
                        adj.add((v, u))
                        deg_in_b[u] += 1
                        deg_in_b[v] += 1
                        added += 1
        
        # Fallback: fill remaining without constraints
        if added < needed:
            all_pairs = b1_b2_pairs + b1_b1_pairs + b2_b2_pairs
            random.shuffle(all_pairs)
            for u, v in all_pairs:
                if added == needed:
                    break
                if (u, v) not in edges and (v, u) not in edges:
                    edges.add((u, v))
                    adj.add((u, v))
                    adj.add((v, u))
                    added += 1

        # Build adjacency masks
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            
        # Calculate max book
        max_b = -1
        max_edges = []
        for u, v in edges:
            b = (adj_masks[u] & adj_masks[v]).bit_count()
            if b > max_b:
                max_b = b
                max_edges = [(u, v)]
            elif b == max_b:
                max_edges.append((u, v))
                
        return edges, adj_masks, max_b, max_edges

    # --- Phase 1: Multi-start Initialization ---
    best_state = None
    best_max_book = float('inf')
    
    seeds = [42, 123, 999, 555, 777, 111, 222, 333, 444, 555]
    for s in seeds:
        edges, adj_masks, max_b, max_edges = build_graph(s)
        if max_b < best_max_book:
            best_max_book = max_b
            best_state = (edges, adj_masks, max_edges)
            
    if best_state is None:
        edges, adj_masks, max_b, max_edges = build_graph(0)
        best_state = (edges, adj_masks, max_edges)
        best_max_book = max_b

    edges, adj_masks, current_max_edges = best_state
    current_max_book = best_max_book

    # --- Phase 2: Optimized Local Search ---
    
    def update_adj(u, v, add):
        if add:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        else:
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
            
    def get_book_size(u, v):
        if u == v:
            return 0
        return (adj_masks[u] & adj_masks[v]).bit_count()
        
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

    no_improve_count = 0
    max_no_improve = 500
    
    # Precompute all possible edges
    all_possible = [(x, y) for x in range(n) for y in range(x + 1, n)]
    
    # Random replacement pool: sample non-edges for speed
    def find_best_replacement(u, v, sample_size=200):
        """Find a good non-edge to replace (u,v) by sampling."""
        # Get current non-edges
        current_edges = set()
        for e in edges:
            current_edges.add(e)
            current_edges.add((e[1], e[0]))
        
        candidates = []
        for x, y in all_possible:
            if x == u or x == v or y == u or y == v:
                continue
            if (x, y) in current_edges or (y, x) in current_edges:
                continue
            candidates.append((x, y))
        
        if not candidates:
            return None, None, float('inf')
        
        # Sample from candidates for speed
        if len(candidates) > sample_size:
            sampled = random.sample(candidates, sample_size)
        else:
            sampled = candidates
        
        min_book = float('inf')
        best_x, best_y = -1, -1
        for x, y in sampled:
            b = (adj_masks[x] & adj_masks[y]).bit_count()
            if b < min_book:
                min_book = b
                best_x, best_y = x, y
                if b <= 1:
                    break
        
        return best_x, best_y, min_book

    while time.time() < deadline:
        if current_max_book <= 7:
            break

        if not current_max_edges:
            break
            
        # Pick a random edge from the max-book set to remove
        u, v = random.choice(current_max_edges)
        
        # Find best replacement from sampled non-edges
        best_x, best_y, min_book_new = find_best_replacement(u, v)
        
        if best_x == -1:
            break
            
        # Perform the swap
        old_max = current_max_book
        old_max_count = len(current_max_edges)
        
        edges.remove((u, v))
        update_adj(u, v, False)
        
        edges.add((best_x, best_y))
        update_adj(best_x, best_y, True)
        
        new_max, new_max_edges = get_max_book_info()
        
        accepted = False
        if new_max < old_max:
            accepted = True
            no_improve_count = 0
        elif new_max == old_max:
            if len(new_max_edges) < old_max_count:
                accepted = True
                no_improve_count = 0
            else:
                if random.random() < 0.03:
                    accepted = True
                else:
                    accepted = False
        else:
            accepted = False

        if accepted:
            current_max_book = new_max
            current_max_edges = new_max_edges
        else:
            # Rollback
            edges.add((u, v))
            update_adj(u, v, True)
            
            edges.remove((best_x, best_y))
            update_adj(best_x, best_y, False)
            
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Try a perturbation: do 5 random swaps that don't increase max book
                perturbed = False
                for _ in range(5):
                    if time.time() > deadline - 5:
                        break
                    # Pick random edge to remove
                    rem_edge = random.choice(list(edges))
                    ru, rv = rem_edge
                    
                    # Find random non-edge to add
                    best_x2, best_y2, _ = find_best_replacement(ru, rv, sample_size=50)
                    if best_x2 == -1:
                        continue
                    
                    old_max_p = current_max_book
                    
                    edges.remove((ru, rv))
                    update_adj(ru, rv, False)
                    edges.add((best_x2, best_y2))
                    update_adj(best_x2, best_y2, True)
                    
                    new_max_p, _ = get_max_book_info()
                    
                    if new_max_p <= old_max_p:
                        current_max_book = new_max_p
                        current_max_edges, _ = get_max_book_info(), None
                        current_max_edges = get_max_book_info()[1]
                        perturbed = True
                        no_improve_count = 0
                        break
                    else:
                        # Rollback
                        edges.add((ru, rv))
                        update_adj(ru, rv, True)
                        edges.remove((best_x2, best_y2))
                        update_adj(best_x2, best_y2, False)
                
                if not perturbed:
                    break

    result = [[u, v] for u, v in edges]
    return result
