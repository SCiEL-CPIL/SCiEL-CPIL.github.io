import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin from 150s limit
    
    n = 40
    target_edges = 401
    
    # 1. Structural Initialization: Replicate the high-performing structure from Inspiration 1
    # This structure has a lower inherent book size than random or K_{20,20}+1.
    A = list(range(9))
    B = list(range(9, 40))
    
    edges = set()
    
    # 1. Add all edges between A and B (9 * 31 = 279 edges)
    for a in A:
        for b in B:
            edges.add((a, b))
            
    # 2. Add 122 edges within B to reach 401 total
    # Split B into B1 (15 vertices: 9..23) and B2 (16 vertices: 24..39)
    B1 = list(range(9, 24))
    B2 = list(range(24, 40))
    
    needed = 122
    added = 0
    
    # We want to distribute edges between B1 and B2 to minimize common neighbors.
    # A balanced distribution is good.
    deg_b1 = {v: 0 for v in B1}
    deg_b2 = {v: 0 for v in B2}
    
    # Generate all possible pairs between B1 and B2
    pairs = [(u, v) for u in B1 for v in B2]
    random.shuffle(pairs)
    
    # Add edges with a degree constraint to keep the structure "sparse" within B
    # Max degree 9 was used in inspiration. 15*9=135 > 122, so it's feasible.
    for u, v in pairs:
        if added == needed:
            break
        if deg_b1[u] < 9 and deg_b2[v] < 9:
            edges.add((u, v))
            deg_b1[u] += 1
            deg_b2[v] += 1
            added += 1
            
    if added < needed:
        # Fallback: fill remaining ignoring constraints if necessary
        for u, v in pairs:
            if added == needed:
                break
            if (u, v) not in edges:
                edges.add((u, v))
                added += 1

    # 2. Efficient Data Structures for Local Search
    # Use bitmasks for adjacency to speed up common neighbor calculation
    adj_masks = [0] * n
    
    def update_adj(u, v, add):
        if add:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        else:
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
            
    for u, v in edges:
        update_adj(u, v, True)
        
    # Precompute bit counts for 16-bit chunks to speed up popcount? 
    # Python's int.bit_count() is very fast in modern versions.
    
    def get_book_size(u, v):
        if u == v:
            return 0
        # Common neighbors are bits set in both adj_masks[u] and adj_masks[v]
        return (adj_masks[u] & adj_masks[v]).bit_count()
        
    def get_max_book_info():
        max_b = -1
        max_edges = []
        # Iterate over edges to find max book
        # Optimization: We can just iterate over all pairs if we maintain a list of edges
        for u, v in edges:
            b = get_book_size(u, v)
            if b > max_b:
                max_b = b
                max_edges = [(u, v)]
            elif b == max_b:
                max_edges.append((u, v))
        return max_b, max_edges

    current_max_book, current_max_edges = get_max_book_info()
    
    # 3. Local Search Refinement
    # Goal: Reduce current_max_book
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        if current_max_book <= 6:
            break
        if current_max_book == 7 and no_improve_count > 300:
            break

        if not current_max_edges:
            break

        # Try up to 3 candidate removal edges per outer iteration
        best_swap = None  # (new_max, new_max_edges, u, v, best_x, best_y)

        num_candidates = min(3, len(current_max_edges))
        candidates = random.sample(current_max_edges, num_candidates)

        for (u, v) in candidates:
            # Find best non-edge to add, considering net impact on max book
            min_net = float('inf')
            best_x, best_y = -1, -1
            best_new_max = float('inf')
            best_new_max_edges = []

            for x in range(n):
                if (1 << v) & adj_masks[x]:
                    continue  # x adjacent to v, skip
                for y in range(x + 1, n):
                    if (1 << x) & adj_masks[y]:
                        continue  # already an edge
                    if (1 << v) & adj_masks[y]:
                        continue  # y adjacent to v, skip (would be triangle with removed edge)

                    # Compute book size of new edge (x,y) after removing (u,v)
                    # Common neighbors of x and y, excluding u and v if they are common
                    common_xy = adj_masks[x] & adj_masks[y]
                    # Remove u and v from consideration if they were common neighbors
                    # Actually, u and v are no longer connected, but they can still be common neighbors
                    # of x and y if x-u, x-v, y-u, y-v all exist.
                    # The book size of (x,y) is just the count of common neighbors.
                    b_new = common_xy.bit_count()

                    # Estimate: after removing (u,v), the max book might decrease if (u,v) was
                    # the unique max. The new max is max of:
                    # - book sizes of all edges except (u,v), adjusted for removal
                    # - b_new for the new edge (x,y)
                    #
                    # Quick estimate: new_max ≈ max(current_max_book - 1 if (u,v) was sole max,
                    #                             current_max_book, b_new)
                    # But this is imprecise. Let's just use b_new as a proxy and prefer
                    # non-edges with low book size, and also check if removing (u,v) helps.

                    # A better heuristic: prefer (x,y) where b_new is small AND
                    # (x,y) doesn't share many neighbors with other high-book edges.
                    # For now, use b_new as primary criterion.

                    if b_new < min_net:
                        min_net = b_new
                        best_x, best_y = x, y
                        if b_new == 0:
                            break
                if min_net == 0:
                    break

            if best_x == -1:
                continue

            # Perform trial swap
            edges.remove((u, v))
            update_adj(u, v, False)
            edges.add((best_x, best_y))
            update_adj(best_x, best_y, True)

            new_max, new_max_edges = get_max_book_info()

            # Rollback for evaluation
            edges.add((u, v))
            update_adj(u, v, True)
            edges.remove((best_x, best_y))
            update_adj(best_x, best_y, False)

            # Track best swap
            if best_swap is None:
                best_swap = (new_max, new_max_edges, u, v, best_x, best_y)
            else:
                if new_max < best_swap[0]:
                    best_swap = (new_max, new_max_edges, u, v, best_x, best_y)
                elif new_max == best_swap[0] and len(new_max_edges) < len(best_swap[1]):
                    best_swap = (new_max, new_max_edges, u, v, best_x, best_y)

        if best_swap is None:
            break

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
            elif random.random() < 0.03:
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
                if time.time() > deadline - 20:
                    break

    result = [[u, v] for u, v in edges]
    return result
