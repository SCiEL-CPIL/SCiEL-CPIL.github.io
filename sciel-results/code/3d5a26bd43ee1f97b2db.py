import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin from 150s limit
    
    n = 40
    target_edges = 401
    
    best_result = None
    best_score = float('inf')
    
    # Try multiple initializations and keep the best
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        random.seed(restart * 12345 + 67890)
        
        A = list(range(9))
        B = list(range(9, 40))
        
        edges = set()
        
        # 1. Add all edges between A and B (9 * 31 = 279 edges)
        for a in A:
            for b in B:
                edges.add((a, b))
                
        # 2. Add 122 edges within B to reach 401 total
        B1 = list(range(9, 24))
        B2 = list(range(24, 40))
        
        needed = 122
        added = 0
        
        # Greedy construction: add edges one by one, minimizing book size increase
        # Build adjacency as we go
        adj_masks_init = [0] * n
        for a in A:
            for b in B:
                adj_masks_init[a] |= (1 << b)
                adj_masks_init[b] |= (1 << a)
        
        # Generate candidate pairs between B1 and B2
        pairs = [(u, v) for u in B1 for v in B2]
        random.shuffle(pairs)
        
        # Greedy: for each candidate edge, compute the book size it would create
        # and pick the one with minimum book size
        for _ in range(needed):
            best_pair = None
            best_book = float('inf')
            
            for u, v in pairs:
                if (1 << v) & adj_masks_init[u]:
                    continue  # already an edge
                
                # Book size of (u,v) = common neighbors of u and v
                book = (adj_masks_init[u] & adj_masks_init[v]).bit_count()
                
                if book < best_book:
                    best_book = book
                    best_pair = (u, v)
                    if book == 0:
                        break
            
            if best_pair is None:
                # Fallback: pick any non-edge
                for u, v in pairs:
                    if not ((1 << v) & adj_masks_init[u]):
                        best_pair = (u, v)
                        break
            
            if best_pair is None:
                break
                
            u, v = best_pair
            edges.add((u, v))
            adj_masks_init[u] |= (1 << v)
            adj_masks_init[v] |= (1 << u)
            added += 1
            
            if time.time() > deadline - 25:
                break
        
        if added < needed:
            # Fallback fill
            for u, v in pairs:
                if added == needed:
                    break
                if (u, v) not in edges:
                    edges.add((u, v))
                    adj_masks_init[u] |= (1 << v)
                    adj_masks_init[v] |= (1 << u)
                    added += 1
        
        if len(edges) != target_edges:
            continue
        
        # 3. Local Search with efficient book tracking
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        
        # Cache book sizes for all edges
        book_cache = {}
        for u, v in edges:
            key = (min(u,v), max(u,v))
            book_cache[key] = (adj_masks[u] & adj_masks[v]).bit_count()
        
        def get_book(u, v):
            key = (min(u,v), max(u,v))
            if key in book_cache:
                return book_cache[key]
            return (adj_masks[u] & adj_masks[v]).bit_count()
        
        def get_max_book_info():
            max_b = -1
            max_edges = []
            for u, v in edges:
                key = (min(u,v), max(u,v))
                b = book_cache.get(key, (adj_masks[u] & adj_masks[v]).bit_count())
                if b > max_b:
                    max_b = b
                    max_edges = [(u, v)]
                elif b == max_b:
                    max_edges.append((u, v))
            return max_b, max_edges
        
        def update_book_cache_after_swap(removed_u, removed_v, added_x, added_y):
            """Update book cache after removing (removed_u, removed_v) and adding (added_x, added_y)."""
            # Remove old edge's book
            key_old = (min(removed_u, removed_v), max(removed_u, removed_v))
            book_cache.pop(key_old, None)
            
            # Add new edge's book
            key_new = (min(added_x, added_y), max(added_x, added_y))
            book_cache[key_new] = (adj_masks[added_x] & adj_masks[added_y]).bit_count()
            
            # Update book sizes of edges incident to removed_u, removed_v, added_x, added_y
            # because their common neighbors changed
            affected = set()
            for u in [removed_u, removed_v, added_x, added_y]:
                for v in range(n):
                    if u != v and (1 << v) & adj_masks[u]:
                        key = (min(u, v), max(u, v))
                        if key in book_cache:
                            affected.add(key)
            
            for key in affected:
                u, v = key
                book_cache[key] = (adj_masks[u] & adj_masks[v]).bit_count()
        
        current_max_book, current_max_edges = get_max_book_info()
        
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < deadline - 5:
            if current_max_book <= 6:
                break
            if current_max_book == 7 and no_improve_count > 200:
                break

            if not current_max_edges:
                break

            best_swap = None

            # Try more candidates when stuck
            num_candidates = min(5, len(current_max_edges))
            candidates = random.sample(current_max_edges, num_candidates)

            for (u, v) in candidates:
                best_x, best_y = -1, -1
                best_eval = float('inf')

                # For each candidate removal edge (u,v), try adding non-edges (x,y)
                # where x is not adjacent to v and y is not adjacent to v (to avoid
                # creating triangles that would increase book sizes)
                for x in range(n):
                    if x == u or x == v:
                        continue
                    if (1 << v) & adj_masks[x]:
                        continue
                    for y in range(x + 1, n):
                        if y == u or y == v:
                            continue
                        if (1 << x) & adj_masks[y]:
                            continue
                        if (1 << v) & adj_masks[y]:
                            continue

                        # Estimate new max book
                        # After removing (u,v): books of edges sharing u or v decrease
                        # After adding (x,y): book of (x,y) is computed, and books of
                        # edges sharing x or y may increase
                        
                        b_new_xy = (adj_masks[x] & adj_masks[y]).bit_count()
                        
                        # Heuristic: prefer low b_new_xy
                        if b_new_xy >= current_max_book:
                            continue
                        
                        # Additional check: does (x,y) create high-book edges with neighbors?
                        # For a neighbor z of x, book(x,z) increases by 1 if y is common neighbor
                        # This is complex; use a simpler heuristic for now
                        
                        eval_score = b_new_xy
                        if eval_score < best_eval:
                            best_eval = eval_score
                            best_x, best_y = x, y
                            if b_new_xy == 0:
                                break
                    if best_eval == 0:
                        break

                if best_x == -1:
                    continue

                # Perform trial swap
                edges.remove((u, v))
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
                edges.add((best_x, best_y))
                adj_masks[best_x] |= (1 << best_y)
                adj_masks[best_y] |= (1 << best_x)
                
                update_book_cache_after_swap(u, v, best_x, best_y)
                
                new_max, new_max_edges = get_max_book_info()
                
                # Rollback
                edges.add((u, v))
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.remove((best_x, best_y))
                adj_masks[best_x] &= ~(1 << best_y)
                adj_masks[best_y] &= ~(1 << best_x)
                
                update_book_cache_after_swap(best_x, best_y, u, v)
                
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
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
                edges.add((best_x, best_y))
                adj_masks[best_x] |= (1 << best_y)
                adj_masks[best_y] |= (1 << best_x)
                
                update_book_cache_after_swap(u, v, best_x, best_y)
                
                current_max_book = new_max
                current_max_edges = new_max_edges
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
        
        # Track best result
        if current_max_book < best_score:
            best_score = current_max_book
            best_result = [[u, v] for u, v in edges]
        elif current_max_book == best_score and best_result is not None:
            # Keep the one with fewer max-book edges (tie-break)
            pass
    
    if best_result is not None:
        return best_result
    
    # Fallback: return a valid graph
    result = [[u, v] for u, v in edges]
    return result
