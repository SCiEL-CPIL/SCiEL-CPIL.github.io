import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin from 150s limit

    n = 40
    target_edges = 401

    def build_initial_graph(seed_val):
        """
        Builds a graph with a bipartite core plus sparse intra-part edges.
        Returns (edges_set, adj_masks_list, max_book, max_edges_list)
        """
        random.seed(seed_val)
        edges = set()

        # Partition into two sets: A (9 vertices), B (31 vertices).
        # Complete bipartite A-B: 9*31 = 279 edges.
        # Need 401 - 279 = 122 edges within B.
        
        A = list(range(9))
        B = list(range(9, 40))
        
        # 1. Add all edges between A and B
        for a in A:
            for b in B:
                edges.add((a, b))
                
        # 2. Add 122 edges within B
        # Greedy approach: pick edges that minimize common neighbors within B.
        
        # Precompute all potential intra-B edges
        potential_edges = [(u, v) for u in B for v in B if u < v]
        random.shuffle(potential_edges)
        
        adj_B = {v: set() for v in B}
        added = 0
        needed = 122
        
        for u, v in potential_edges:
            if added == needed:
                break
            # Check if u and v have common neighbors in B
            if adj_B[u] & adj_B[v]:
                continue
            
            edges.add((u, v))
            adj_B[u].add(v)
            adj_B[v].add(u)
            added += 1
            
        if added < needed:
            # Relax condition: allow some common neighbors, but prefer those with fewer
            remaining = [(u, v) for u, v in potential_edges if (u, v) not in edges]
            def common_count(e):
                u, v = e
                return len(adj_B[u] & adj_B[v])
            remaining.sort(key=common_count)
            
            for u, v in remaining:
                if added == needed:
                    break
                edges.add((u, v))
                adj_B[u].add(v)
                adj_B[v].add(u)
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

    def build_random_graph(seed_val):
        """
        Builds a random graph with 401 edges.
        """
        random.seed(seed_val)
        edges = set()
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        for i in range(target_edges):
            edges.add(all_edges[i])
        
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            
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
    
    # Try structural seeds
    structural_seeds = [42, 123, 999, 555, 777, 111, 222, 333, 444, 666]
    for s in structural_seeds:
        edges, adj_masks, max_b, max_edges = build_initial_graph(s)
        if max_b < best_max_book:
            best_max_book = max_b
            best_state = (edges, adj_masks, max_edges)
            
    # Try random seeds
    random_seeds = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
    for s in random_seeds:
        edges, adj_masks, max_b, max_edges = build_random_graph(s)
        if max_b < best_max_book:
            best_max_book = max_b
            best_state = (edges, adj_masks, max_edges)
            
    edges, adj_masks, current_max_edges = best_state
    current_max_book = best_max_book

    # --- Phase 2: Local Search Refinement ---
    
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
    max_no_improve = 5000 # Allow more exploration
    
    while time.time() < deadline:
        if current_max_book <= 7:
            # Theoretical lower bound is 40/6 = 6.66. So 7 is the smallest integer possible.
            if current_max_book == 7:
                break
            elif current_max_book <= 6:
                # This shouldn't happen if the theorem holds, but just in case.
                break

        if not current_max_edges:
            break
            
        # Pick a random edge from the max-book set to remove
        u, v = random.choice(current_max_edges)
        
        # Find the best non-edge to add
        # We want to minimize the book size of the new edge.
        # Heuristic: Find non-edge (x,y) with min common neighbors.
        min_book_new = float('inf')
        best_x, best_y = -1, -1
        
        # Iterate over all possible non-edges
        for x in range(n):
            for y in range(x + 1, n):
                if (1 << y) & adj_masks[x]:
                    continue
                
                b = (adj_masks[x] & adj_masks[y]).bit_count()
                if b < min_book_new:
                    min_book_new = b
                    best_x, best_y = x, y
                    if b == 0:
                        # Can't get better than 0
                        break
            if min_book_new == 0:
                break
        
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
                # Small probability to accept to escape local minima
                if random.random() < 0.05:
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
                # Random restart?
                # Let's try a "kick": remove a random edge (not necessarily max book) and add a random non-edge
                # if it doesn't increase the max book.
                
                # Actually, let's just break. The multi-start already provided diversity.
                break

    result = [[u, v] for u, v in edges]
    return result
