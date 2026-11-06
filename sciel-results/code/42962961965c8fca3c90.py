import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 55.0
    
    n = 256
    max_sum = 2 * n
    
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
        
    # Diagonal constraint: 2*a must not be a square
    # So we only consider 'a' such that 2*a is not a perfect square
    valid_numbers = []
    for a in range(1, n + 1):
        if 2 * a not in squares:
            valid_numbers.append(a)
            
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}
    
    # Build conflict graph (adjacency masks)
    # conflict_masks[i] is a bitmask where bit j is 1 if valid_numbers[i] and valid_numbers[j] conflict
    # Conflict if a+b is a perfect square
    conflict_masks = [0] * num_valid
    
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b > a and b in val_to_idx:
                j = val_to_idx[b]
                conflict_masks[i] |= (1 << j)
                conflict_masks[j] |= (1 << i)
                
    # We want to find the Maximum Independent Set in the conflict graph.
    # This is equivalent to finding the Maximum Clique in the complement graph.
    # However, Tomita's algorithm is usually described for Max Clique.
    # Let's adapt Tomita's Max Clique to run on the "non-conflict" graph (complement of conflict graph).
    # Wait, standard MIS = Max Clique in Complement.
    # Let G_conflict be the graph where edges are conflicts.
    # We want a set of vertices with NO edges in G_conflict.
    # This is an Independent Set in G_conflict.
    # This is a Clique in G_complement (where edges are non-conflicts).
    
    # Let's build the adjacency masks for the COMPLEMENT graph (non-conflict graph).
    # non_conflict_masks[i] has bit j set if i and j do NOT conflict (and i != j).
    
    # Optimization: 
    # It is often faster to solve Max Clique on a dense graph than Max Independent Set on a sparse graph?
    # Actually, Tomita is efficient on dense graphs for Max Clique.
    # The conflict graph is sparse (each node has few conflicts).
    # The complement graph is dense.
    # So solving Max Clique on the complement graph is the right move for Tomita.
    
    non_conflict_masks = [0] * num_valid
    full_mask = (1 << num_valid) - 1
    
    for i in range(num_valid):
        # All bits set
        mask = full_mask
        # Remove self
        mask &= ~(1 << i)
        # Remove conflicts
        mask &= ~conflict_masks[i]
        non_conflict_masks[i] = mask
        
    # Tomita's algorithm for Maximum Clique
    # Uses coloring to provide upper bounds.
    
    def expand(C, P):
        # C: list of vertices in the current clique
        # P: bitmask of vertices available to be added to C
        # Returns the size of the maximum clique found starting with C and P
        
        # Base case
        if P == 0:
            return len(C)
        
        max_size = len(C)
        
        # Order vertices in P by degree in P (greedy coloring order)
        # Tomita's algorithm uses a specific ordering based on coloring.
        # We will implement a simplified version that uses greedy coloring to bound the search.
        
        # To speed up, we can use a recursive approach with bitsets.
        
        # 1. Color the subgraph induced by P.
        # We assign colors to vertices in P such that adjacent vertices have different colors.
        # The number of colors used is an upper bound on the clique size in P.
        
        # Greedy coloring algorithm:
        # We process vertices in P in some order.
        # A common heuristic is to order by degree descending, or just iterate.
        # Tomita uses a specific "color sort".
        
        # Let's extract vertices from P.
        # Since num_valid is ~250, iterating is fine.
        
        # We will use a list of vertices sorted by the color assignment.
        # Actually, the standard Tomita algorithm:
        # 1. Compute a coloring of P.
        # 2. Sort vertices in P by color.
        # 3. Iterate through vertices in reverse color order.
        
        # Let's implement a simple greedy coloring on the subgraph induced by P.
        # Note: The graph here is the NON-CONFLICT graph (dense).
        
        # To make it fast, we'll use a heuristic ordering.
        # Let's just iterate through bits of P.
        
        # We need to map bit indices to actual vertex indices.
        # Since we are using bitmasks, we can iterate.
        
        # Greedy Coloring:
        # colors[v] = color assigned to v
        # We assign colors 1, 2, ... such that neighbors have different colors.
        
        # For efficiency, we will use a recursive backtracking for the clique,
        # but prune using the coloring bound.
        
        # Let's get the list of vertices in P.
        verts = []
        p = P
        while p:
            lsb = p & -p
            idx = lsb.bit_length() - 1
            verts.append(idx)
            p ^= lsb
            
        # Sort by degree in P? Or just use the order?
        # Tomita's algorithm sorts by color.
        
        # Let's perform greedy coloring on the induced subgraph of P.
        # We want to maximize the color bound, so we should color carefully.
        # Standard greedy:
        # For each vertex in some order, assign the smallest available color.
        
        # To get a good bound, we often order vertices by degree descending.
        # Let's compute degrees in P.
        degrees = [0] * num_valid
        for v in verts:
            deg = (non_conflict_masks[v] & P).bit_count()
            degrees[v] = deg
            
        # Sort verts by degree descending
        verts.sort(key=lambda x: degrees[x], reverse=True)
        
        colors = {}
        # color_to_verts[k] = list of vertices with color k
        color_to_verts = {}
        
        for v in verts:
            # Find smallest color not used by neighbors of v in P
            # Neighbors of v in P are (non_conflict_masks[v] & P)
            # We need to check colors of these neighbors.
            
            # To speed this up, we can keep track of which colors are used by neighbors.
            # Since the graph is dense, most colors might be used.
            
            # Let's just check existing colors.
            # The max color is at most len(verts).
            
            used_colors = set()
            neighbors_mask = non_conflict_masks[v] & P
            # Iterate neighbors? That's O(N).
            # For N=250, O(N^2) coloring is 62,500 ops. Acceptable.
            
            n_mask = neighbors_mask
            while n_mask:
                lsb = n_mask & -n_mask
                n_idx = lsb.bit_length() - 1
                if n_idx in colors:
                    used_colors.add(colors[n_idx])
                n_mask ^= lsb
                
            # Find smallest positive integer not in used_colors
            c = 1
            while c in used_colors:
                c += 1
            
            colors[v] = c
            if c not in color_to_verts:
                color_to_verts[c] = []
            color_to_verts[c].append(v)
            
        # Now we have a coloring.
        # The max color is an upper bound on the clique size in P.
        # We process colors in descending order.
        
        # We maintain a list of vertices to consider, ordered by color.
        # Tomita's algorithm:
        # For color k from max_color down to 1:
        #   For each vertex v with color k:
        #     If len(C) + k <= max_size: break (prune)
        #     Expand(C + [v], P & neighbors(v))
        #     Remove v from P
        
        max_color = max(color_to_verts.keys()) if color_to_verts else 0
        
        # We need to process vertices in a specific way to maintain the bound.
        # The standard approach:
        # Iterate through colors from max to 1.
        # For each vertex v in that color class:
        #   Bound: len(C) + current_color <= max_size?
        #   Actually, the bound is len(C) + max_color_remaining.
        
        # Let's restructure to follow Tomita more closely.
        # We have a list of vertices sorted by color.
        # We iterate from the highest color down.
        
        # To allow pruning, we need to know the max color available at each step.
        
        # Let's create a list of (color, vertex) pairs, sorted by color descending.
        # But we must remove vertices as we go?
        # Tomita's algorithm typically uses a "candidate set" that is reduced.
        
        # A simpler recursive structure often used in Python implementations:
        
        def rec(clique, candidates):
            nonlocal max_size
            if not candidates:
                return
            if len(clique) + candidates.bit_count() <= max_size:
                return
            
            # Color candidates
            # We use the coloring computed above? 
            # The coloring must be dynamic as candidates change.
            # Recomputing coloring at each node is expensive but N is small.
            
            # Let's re-color the induced subgraph of candidates.
            c_verts = []
            c_p = candidates
            while c_p:
                lsb = c_p & -c_p
                idx = lsb.bit_length() - 1
                c_verts.append(idx)
                c_p ^= lsb
                
            # Sort by degree in candidates
            c_degrees = [0] * num_valid
            for v in c_verts:
                c_degrees[v] = (non_conflict_masks[v] & candidates).bit_count()
            c_verts.sort(key=lambda x: c_degrees[x], reverse=True)
            
            c_colors = {}
            c_color_to_verts = {}
            for v in c_verts:
                used = set()
                nb_mask = non_conflict_masks[v] & candidates
                while nb_mask:
                    lsb = nb_mask & -nb_mask
                    n_idx = lsb.bit_length() - 1
                    if n_idx in c_colors:
                        used.add(c_colors[n_idx])
                    nb_mask ^= lsb
                col = 1
                while col in used:
                    col += 1
                c_colors[v] = col
                if col not in c_color_to_verts:
                    c_color_to_verts[col] = []
                c_color_to_verts[col].append(v)
                
            if not c_color_to_verts:
                return
                
            max_col = max(c_color_to_verts.keys())
            
            # Process colors from max_col down to 1
            # We need to be careful: when we pick a vertex v, we remove it and its non-neighbors?
            # No, we remove v and its NON-NEIGHBORS from the candidate set for the next recursion?
            # Wait.
            # If we add v to the clique, the next vertices must be NEIGHBORS of v.
            # So the new candidate set is (candidates & non_conflict_masks[v]) minus v.
            
            # Tomita's pruning:
            # If len(clique) + col <= max_size, we can skip all vertices with color <= col.
            
            # We iterate vertices in descending color order.
            # We maintain a list of vertices to process.
            
            # To implement the "remove from candidates" logic correctly with the bound:
            # We iterate through the color classes from high to low.
            
            # Let's create a list of vertices sorted by color descending.
            # But we must respect the dynamic candidate set reduction.
            
            # Standard Tomita:
            # P is the candidate set.
            # We color P.
            # We iterate v in P in order of decreasing color.
            # If len(C) + color(v) <= best, break.
            # Else, recurse with C+v and P & N(v).
            # Then remove v from P.
            
            # This requires P to be mutable or passed by value.
            
            # Let's implement this loop.
            
            # We need to iterate v in P sorted by color.
            # But P changes as we remove v.
            
            # Efficient way:
            # Keep a list of vertices in P sorted by color.
            # As we remove v, we just skip it in the next iterations?
            # No, the colors of remaining vertices might change?
            # No, Tomita's algorithm uses the coloring of the INITIAL P for the loop,
            # but the bound is based on the color of the vertex being considered.
            # However, the candidate set for the RECURSION is restricted.
            
            # Let's stick to the recursive structure where we re-color at each step.
            # It is safer.
            
            # Iterate through colors from max_col down to 1
            # For each color class, iterate through vertices.
            # We need to check if the vertex is still in the current candidate set?
            # Yes, because we remove vertices as we go.
            
            # Let's maintain a `current_candidates` bitmask.
            
            current_candidates = candidates
            # We need to process vertices in order of decreasing color.
            # Let's create a list of (color, vertex) and sort.
            
            ordered_verts = []
            for col in range(max_col, 0, -1):
                if col in c_color_to_verts:
                    for v in c_color_to_verts[col]:
                        ordered_verts.append((col, v))
                        
            for col, v in ordered_verts:
                if time.time() - start_time > deadline - 2:
                    return
                
                # Check if v is still in current_candidates
                if not (current_candidates & (1 << v)):
                    continue
                    
                # Pruning
                if len(clique) + col <= max_size:
                    # Since we are processing in descending color, 
                    # all subsequent vertices have color <= col.
                    # So len(clique) + subsequent_color <= len(clique) + col <= max_size.
                    # We can break out of the loop entirely.
                    break
                
                # Expand
                new_clique = clique + [v]
                # New candidates must be neighbors of v in the current candidate set
                new_candidates = current_candidates & non_conflict_masks[v]
                # Remove v itself (though it's not a neighbor of itself in simple graph)
                new_candidates &= ~(1 << v)
                
                rec(new_clique, new_candidates)
                
                # Remove v from current_candidates for the next iteration of this level
                current_candidates &= ~(1 << v)
                
                # Update max_size if we found a bigger clique
                # This is done inside rec, but we need to update the outer max_size
                # Wait, max_size is nonlocal. rec updates it.
                
                # Check time
                if time.time() - start_time > deadline - 2:
                    return

        # Initialize max_size with a greedy heuristic to get a good lower bound quickly
        # Greedy Max Clique
        greedy_clique = []
        greedy_p = P
        while greedy_p:
            # Pick vertex with highest degree in greedy_p
            best_v = -1
            best_deg = -1
            gp = greedy_p
            while gp:
                lsb = gp & -gp
                idx = lsb.bit_length() - 1
                deg = (non_conflict_masks[idx] & greedy_p).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = idx
                gp ^= lsb
            if best_v == -1:
                break
            greedy_clique.append(best_v)
            greedy_p &= non_conflict_masks[best_v]
            greedy_p &= ~(1 << best_v)
            
        max_size = len(greedy_clique)
        
        # Start recursion
        # We pass an empty clique and P
        # But we already did greedy. We can start recursion with empty clique and P.
        # The recursion will find better solutions.
        
        rec([], P)
        
        return max_size

    # Run Tomita
    # To improve performance, we can start with a good initial solution.
    # We already have a greedy one inside expand.
    
    # We call expand with empty clique and full mask
    # But expand returns the max size found.
    
    # We need to store the actual set, not just the size, to return it.
    # Let's modify the approach to store the best set.
    
    best_set = set()
    
    def expand_with_set(C, P):
        nonlocal best_set, max_size
        if P == 0:
            if len(C) > len(best_set):
                best_set = set(C)
                max_size = len(C)
            return
        
        # ... same coloring logic ...
        # This is getting complex to duplicate.
        # Let's just run the size-finding version, then reconstruct?
        # No, reconstruction is hard.
        
        # Let's modify the recursive function to track the set.
        
    # Alternative: Just run the size finder, then use the size to guide a greedy fill?
    # No, we need the exact set.
    
    # Let's refactor to track the best set.
    
    global_best_set = set()
    global_best_size = 0
    
    def find_max_clique(C, P):
        nonlocal global_best_set, global_best_size
        if time.time() - start_time > deadline - 1:
            return
            
        if P == 0:
            if len(C) > global_best_size:
                global_best_size = len(C)
                global_best_set = set(C)
            return
            
        # Color P
        verts = []
        p = P
        while p:
            lsb = p & -p
            idx = lsb.bit_length() - 1
            verts.append(idx)
            p ^= lsb
            
        if not verts:
            return
            
        # Sort by degree
        degrees = [0] * num_valid
        for v in verts:
            degrees[v] = (non_conflict_masks[v] & P).bit_count()
        verts.sort(key=lambda x: degrees[x], reverse=True)
        
        colors = {}
        color_to_verts = {}
        for v in verts:
            used = set()
            nb_mask = non_conflict_masks[v] & P
            while nb_mask:
                lsb = nb_mask & -nb_mask
                n_idx = lsb.bit_length() - 1
                if n_idx in colors:
                    used.add(colors[n_idx])
                nb_mask ^= lsb
            col = 1
            while col in used:
                col += 1
            colors[v] = col
            if col not in color_to_verts:
                color_to_verts[col] = []
            color_to_verts[col].append(v)
            
        max_col = max(color_to_verts.keys())
        
        # Process in descending color order
        # We need to iterate through vertices such that we can prune.
        
        # Create list of (col, v)
        ordered = []
        for col in range(max_col, 0, -1):
            if col in color_to_verts:
                for v in color_to_verts[col]:
                    ordered.append((col, v))
                    
        current_P = P
        
        for col, v in ordered:
            if time.time() - start_time > deadline - 1:
                return
                
            if not (current_P & (1 << v)):
                continue
                
            # Prune
            if len(C) + col <= global_best_size:
                break
                
            # Recurse
            new_C = C + [v]
            new_P = current_P & non_conflict_masks[v]
            new_P &= ~(1 << v)
            
            find_max_clique(new_C, new_P)
            
            current_P &= ~(1 << v)
            
    # Initialize with a greedy solution to set a high bar
    greedy_C = []
    greedy_P = (1 << num_valid) - 1
    while greedy_P:
        best_v = -1
        best_deg = -1
        gp = greedy_P
        while gp:
            lsb = gp & -gp
            idx = lsb.bit_length() - 1
            deg = (non_conflict_masks[idx] & greedy_P).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = idx
            gp ^= lsb
        if best_v == -1:
            break
        greedy_C.append(best_v)
        greedy_P &= non_conflict_masks[best_v]
        greedy_P &= ~(1 << best_v)
        
    global_best_set = set(greedy_C)
    global_best_size = len(greedy_C)
    
    # Run the exact solver
    find_max_clique([], (1 << num_valid) - 1)
    
    # Convert result back to original numbers
    result = [valid_numbers[i] for i in global_best_set]
    
    if not result:
        return [1]
        
    return result

# EVOLVE-BLOCK-END
