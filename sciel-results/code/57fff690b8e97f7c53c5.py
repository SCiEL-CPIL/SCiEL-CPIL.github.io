import time
import random
import itertools

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for safety
    
    n = 16
    # Precompute all K4s
    vertices = list(range(n))
    k4s = list(itertools.combinations(vertices, 4))
    num_k4s = len(k4s)
    
    # Map each K4 to its 6 edges
    # Edge is a tuple (i, j) with i < j
    # We need a fast way to get edge indices.
    # Let's create a dictionary mapping (i, j) to index in edge list
    edge_list = []
    edge_index = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = idx
            edge_list.append((i, j))
            idx += 1
    
    # For each K4, store the list of edge indices
    k4_edges = []
    for k4 in k4s:
        edges = []
        for a, b in itertools.combinations(k4, 2):
            edges.append(edge_index[(a, b)])
        k4_edges.append(edges)
        
    # Precompute for each edge, the list of K4 indices it belongs to
    edge_to_k4s = [[] for _ in range(len(edge_list))]
    for k4_idx, edges in enumerate(k4_edges):
        for e_idx in edges:
            edge_to_k4s[e_idx].append(k4_idx)
            
    # State: list of colours for each edge
    # We will try to find a valid colouring with C colours
    # Start with C=13
    best_colours = None
    best_cost = float('inf')
    
    # Try C from 13 to 15
    for C in [13, 14, 15]:
        if time.time() > deadline:
            break
            
        # Random initialization
        colours = [random.randint(0, C - 1) for _ in range(len(edge_list))]
        
        # Calculate initial violations
        # violations[k] = 1 if K4 k has < 5 colours, else 0
        violations = [0] * num_k4s
        current_cost = 0
        
        def update_violation(k4_idx, colours):
            edges = k4_edges[k4_idx]
            c_set = set(colours[e] for e in edges)
            is_valid = len(c_set) >= 5
            return 0 if is_valid else 1

        for k in range(num_k4s):
            v = update_violation(k, colours)
            violations[k] = v
            current_cost += v
            
        best_colours_for_C = colours[:]
        best_cost_for_C = current_cost
        
        # Local search
        iterations = 0
        max_iterations = 50000 # Cap per C to prevent infinite loop if stuck
        
        while current_cost > 0 and iterations < max_iterations:
            if time.time() > deadline:
                break
                
            # Pick a random violating K4
            violating_k4s = [k for k in range(num_k4s) if violations[k] == 1]
            if not violating_k4s:
                break
            k4_idx = random.choice(violating_k4s)
            
            # Try changing each edge in this K4 to each colour
            best_delta = float('inf')
            best_move = None
            
            for e_idx in k4_edges[k4_idx]:
                old_c = colours[e_idx]
                for new_c in range(C):
                    if new_c == old_c:
                        continue
                    
                    # Calculate delta cost
                    # Only K4s containing e_idx are affected
                    delta = 0
                    for k in edge_to_k4s[e_idx]:
                        old_v = violations[k]
                        # Simulate colour change
                        # We need to check if the K4 becomes valid/invalid
                        # Get edges of K4 k
                        edges_k = k4_edges[k]
                        c_set = set()
                        for e in edges_k:
                            if e == e_idx:
                                c_set.add(new_c)
                            else:
                                c_set.add(colours[e])
                        new_v = 0 if len(c_set) >= 5 else 1
                        delta += new_v - old_v
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (e_idx, new_c)
            
            # Apply best move
            if best_move:
                e_idx, new_c = best_move
                old_c = colours[e_idx]
                colours[e_idx] = new_c
                current_cost += best_delta
                # Update violations for affected K4s
                for k in edge_to_k4s[e_idx]:
                    edges_k = k4_edges[k]
                    c_set = set(colours[e] for e in edges_k)
                    violations[k] = 0 if len(c_set) >= 5 else 1
                    
                if current_cost < best_cost_for_C:
                    best_cost_for_C = current_cost
                    best_colours_for_C = colours[:]
                    
                # If cost is 0, we found a solution!
                if current_cost == 0:
                    best_colours = best_colours_for_C
                    return best_colours # Return immediately? Or check if we can do better with fewer colours?
                    # Since we are iterating C from 13 upwards, if we find a solution with 13, it's the best possible.
                    # So we can return.
                    
            else:
                # No improving move found, try random move to escape local minimum
                e_idx = random.choice(k4_edges[k4_idx])
                new_c = random.randint(0, C - 1)
                old_c = colours[e_idx]
                if new_c == old_c:
                    continue
                colours[e_idx] = new_c
                # Recalculate cost delta for affected K4s
                delta = 0
                for k in edge_to_k4s[e_idx]:
                    old_v = violations[k]
                    edges_k = k4_edges[k]
                    c_set = set(colours[e] for e in edges_k)
                    new_v = 0 if len(c_set) >= 5 else 1
                    violations[k] = new_v
                    delta += new_v - old_v
                current_cost += delta
                if current_cost < best_cost_for_C:
                    best_cost_for_C = current_cost
                    best_colours_for_C = colours[:]
            
            iterations += 1
            
        # End of local search for this C
        if best_cost_for_C == 0:
            best_colours = best_colours_for_C
            return best_colours
        else:
            # Keep the best found so far if it's better than previous
            if best_cost_for_C < best_cost:
                best_cost = best_cost_for_C
                best_colours = best_colours_for_C
                # We don't return here because we might find a 0-cost solution with C=14 later?
                # No, we want minimum colours. If 13 fails, 14 is worse.
                # But if 13 fails, we should try 14.
                # However, if we found a valid 13, we returned.
                # If we didn't, we continue to next C.
                
    # If we exit the loop without finding a 0-cost solution, return the best valid one?
    # The problem requires a VALID colouring.
    # If we didn't find one with 13, 14, or 15, we should return the seed (17) to be safe?
    # Or the best invalid one? No, invalid scores 0.
    # The seed is valid with 17.
    # We should ensure we return a valid colouring.
    # If local search fails to find a valid one with 13-15, we fall back to the seed.
    
    if best_colours is not None:
        # Check if best_colours is valid
        # It should be if best_cost was 0.
        # But if we broke out of loop due to time, best_colours might be invalid.
        # Let's verify.
        valid = True
        for k in range(num_k4s):
            edges = k4_edges[k]
            c_set = set(best_colours[e] for e in edges)
            if len(c_set) < 5:
                valid = False
                break
        if valid:
            return best_colours
    
    # Fallback to seed
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
