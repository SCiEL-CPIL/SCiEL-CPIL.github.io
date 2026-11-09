import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 40
    target_edges = 401
    
    # 1. Structural Initialization: Use the "Bipartite + Sparse Internal" structure from Inspiration 1.
    # This structure naturally has a lower book size than a random graph or a pure K_{20,20}+1.
    # Split vertices into A (size 9) and B (size 31).
    # Add all edges between A and B.
    # Add edges within B to reach the target count.
    
    A = list(range(9))
    B = list(range(9, 40))
    
    edges = set()
    
    # Add all edges between A and B
    for a in A:
        for b in B:
            edges.add((a, b))
            
    # Calculate remaining edges needed
    current_count = len(edges)
    needed = target_edges - current_count
    
    # Split B into B1 and B2 to distribute internal edges and minimize common neighbors.
    # B1: first 15, B2: last 16
    B1 = list(range(9, 24))
    B2 = list(range(24, 40))
    
    # Generate candidate pairs within B
    # We prioritize pairs between B1 and B2 as they are less likely to create high-degree triangles
    # compared to pairs within B1 or within B2 if we balance degrees.
    # Actually, let's just generate all pairs in B and shuffle.
    pairs_in_B = []
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            pairs_in_B.append((B[i], B[j]))
            
    random.shuffle(pairs_in_B)
    
    # Add edges with a soft degree constraint to keep the graph "balanced" within B.
    # Max degree ~9 seems reasonable given the size.
    deg = {v: 0 for v in B}
    added = 0
    max_deg_constraint = 9
    
    for u, v in pairs_in_B:
        if added == needed:
            break
        # Check if adding this edge keeps degrees balanced
        if deg[u] < max_deg_constraint and deg[v] < max_deg_constraint:
            edges.add((u, v))
            deg[u] += 1
            deg[v] += 1
            added += 1
            
    if added < needed:
        # Fallback: fill remaining ignoring constraints
        for u, v in pairs_in_B:
            if added == needed:
                break
            if (u, v) not in edges:
                edges.add((u, v))
                added += 1

    # 2. Efficient Data Structures for Local Search
    # Use bitmasks for adjacency to speed up common neighbor calculation (book size).
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

    current_max_book, current_max_edges = get_max_book_info()
    best_max_book = current_max_book
    best_edges = set(edges)
    
    # 3. Local Search Refinement
    # Goal: Reduce current_max_book
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        # If we hit the theoretical lower bound (7), we can stop or continue slightly to ensure stability.
        # 40/6 = 6.66, so 7 is the integer ceiling.
        if current_max_book == 7:
            if no_improve_count > 200:
                break
        elif current_max_book <= 6:
            break

        # Pick a random edge from the max-book set to remove
        if not current_max_edges:
            break
            
        u, v = random.choice(current_max_edges)
        
        # Find the best non-edge to add
        # Heuristic: Find non-edge (x,y) with min common neighbors (min book size for the new edge).
        # Also, we should prefer non-edges that don't create new high-book edges.
        # Since we remove (u,v), the book size of (u,v) drops.
        # We want to add an edge that has a small book size.
        
        min_book_new = float('inf')
        best_x, best_y = -1, -1
        
        # Iterate over all possible non-edges
        # Optimization: We can break early if we find 0, but 0 is rare.
        # We can also sample a subset if the full scan is too slow, but 780 pairs is fast.
        
        for x in range(n):
            for y in range(x + 1, n):
                # Check if (x,y) is an edge
                if (1 << y) & adj_masks[x]:
                    continue
                
                # Calculate book size for potential new edge
                b = (adj_masks[x] & adj_masks[y]).bit_count()
                if b < min_book_new:
                    min_book_new = b
                    best_x, best_y = x, y
                    if b == 0:
                        break
            if min_book_new == 0:
                break
        
        if best_x == -1:
            # No non-edges? Impossible with 401 edges out of 780.
            break
            
        # Perform the swap: remove (u,v), add (best_x, best_y)
        
        # Save state for rollback
        old_max = current_max_book
        old_max_count = len(current_max_edges)
        
        # Remove (u,v)
        edges.remove((u, v))
        update_adj(u, v, False)
        
        # Add (best_x, best_y)
        edges.add((best_x, best_y))
        update_adj(best_x, best_y, True)
        
        # Recalculate max book
        new_max, new_max_edges = get_max_book_info()
        
        # Acceptance Criteria
        accepted = False
        if new_max < old_max:
            accepted = True
            no_improve_count = 0
        elif new_max == old_max:
            # Prefer moves that reduce the number of max-book edges
            if len(new_max_edges) < old_max_count:
                accepted = True
                no_improve_count = 0
            else:
                # Small probability to accept to escape local minima
                if random.random() < 0.02:
                    accepted = True
                else:
                    accepted = False
        else:
            # If new_max > old_max, reject
            accepted = False

        if accepted:
            current_max_book = new_max
            current_max_edges = new_max_edges
            if current_max_book < best_max_book:
                best_max_book = current_max_book
                best_edges = set(edges)
        else:
            # Rollback
            edges.add((u, v))
            update_adj(u, v, True)
            
            edges.remove((best_x, best_y))
            update_adj(best_x, best_y, False)
            
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturb the graph to escape local minima
                # Do a few random swaps
                for _ in range(10):
                    if time.time() > deadline - 5:
                        break
                    # Random swap
                    old_e = random.choice(list(edges))
                    # Find random non-edge
                    attempts = 0
                    while True:
                        x, y = random.sample(range(n), 2)
                        new_e = (min(x,y), max(x,y))
                        if new_e not in edges:
                            break
                        attempts += 1
                        if attempts > 10:
                            break
                    
                    if new_e not in edges:
                        edges.remove(old_e)
                        edges.add(new_e)
                        # Update adj
                        ou, ov = old_e
                        update_adj(ou, ov, False)
                        update_adj(x, y, True)
                        
                        # Recalc
                        new_max, new_max_edges = get_max_book_info()
                        current_max_book = new_max
                        current_max_edges = new_max_edges
                        if current_max_book < best_max_book:
                            best_max_book = current_max_book
                            best_edges = set(edges)
                
                no_improve_count = 0
                if time.time() > deadline - 20:
                    break

    # Return best found
    return [[u, v] for u, v in best_edges]
