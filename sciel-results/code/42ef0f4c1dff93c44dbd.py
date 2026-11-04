import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    time_limit = 100  # leave buffer for overhead
    
    # Generate support-size-2 candidates (baseline: 220)
    candidates2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates2.append(v)
    candidates2 = np.array(candidates2, dtype=np.int64)
    baseline = len(candidates2)  # 220
    
    # Generate support-size-3 candidates
    candidates3 = []
    for support in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates3.append(v)
    candidates3 = np.array(candidates3, dtype=np.int64)
    n3 = len(candidates3)  # 1320
    
    # Build conflict graph for support-3
    # All have norm^2=3, so conflict if dot product > 1
    dots3 = candidates3 @ candidates3.T
    conflict3 = (dots3 > 1)
    np.fill_diagonal(conflict3, False)
    
    # We want Maximum Independent Set in conflict3.
    # This is Maximum Clique in the complement graph.
    # Complement graph adjacency: adj[i][j] is True if NOT conflict3[i][j]
    # Note: In conflict3, diagonal is False. In complement, diagonal should be False for clique search (no self-loops).
    # But usually MaxClique algorithms work on simple graphs.
    
    # Build adjacency list for complement graph
    # adj[i] is a set of neighbors j such that conflict3[i,j] is False and i != j
    adj = [set() for _ in range(n3)]
    for i in range(n3):
        for j in range(i + 1, n3):
            if not conflict3[i, j]:
                adj[i].add(j)
                adj[j].add(i)
    
    # Tomita's Maximum Clique Algorithm with Coloring
    # P: set of candidates (initially all nodes)
    # R: current clique
    # We use a list for P to allow ordering and coloring
    
    best_size = 0
    best_clique = []
    
    # Order vertices by degree in complement graph (descending) to help coloring
    # Higher degree in complement = more compatible neighbors
    degrees = [len(adj[i]) for i in range(n3)]
    order = sorted(range(n3), key=lambda x: -degrees[x])
    
    # Remap to ordered indices
    remap = {old: new for new, old in enumerate(order)}
    adj_remapped = [set() for _ in range(n3)]
    for i in range(n3):
        for j in adj[i]:
            adj_remapped[remap[i]].add(remap[j])
            
    # Greedy Coloring for Upper Bound
    # Returns (colors, color_classes) where colors[i] is the color of vertex i
    # and color_classes is a list of lists of vertices for each color
    # We process vertices in the given order (usually sorted by degree or just the current P order)
    def greedy_coloring(P):
        # P is a list of vertices
        colors = {}
        color_classes = []
        # Simple greedy: assign smallest available color
        # To be effective, we should sort P by degree in the induced subgraph? 
        # Tomita uses a specific ordering. 
        # For simplicity and speed, we use the global order or degree in P.
        # Let's use the pre-computed order if P is the full set, or sort by degree in P otherwise.
        # Sorting at each step is expensive. 
        # Tomita's original paper uses a static ordering or re-sorts.
        # We will sort P by degree in the current induced subgraph for better bounds.
        
        # Calculate degrees in P
        deg_in_P = {v: len([u for u in P if u in adj_remapped[v]]) for v in P}
        sorted_P = sorted(P, key=lambda x: -deg_in_P[x])
        
        for v in sorted_P:
            used_colors = set()
            for u in adj_remapped[v]:
                if u in colors:
                    used_colors.add(colors[u])
            # Find smallest unused color
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            if c >= len(color_classes):
                color_classes.append([])
            color_classes[c].append(v)
            
        return colors, color_classes, sorted_P

    def max_clique_search(R, P):
        nonlocal best_size, best_clique
        if time.time() - start_time > time_limit:
            return
        
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        
        # Upper bound
        colors, color_classes, sorted_P = greedy_coloring(P)
        max_color = max(colors.values()) if P else 0
        
        if len(R) + max_color <= best_size:
            return
        
        # Process vertices in reverse order of colors (highest color first)
        # This is a standard optimization: vertices with higher colors are more "constrained" or "central"
        # Actually, Tomita processes in reverse color order.
        # Let's iterate through sorted_P in reverse order of their assigned color.
        # Or simply, we can branch on vertices in the order of sorted_P.
        
        # To implement Tomita correctly:
        # 1. Color P.
        # 2. Iterate through vertices in P in reverse order of their color.
        # 3. For each vertex v:
        #    - If len(R) + colors[v] <= best_size: break (since subsequent vertices have lower/equal color)
        #    - Expand with v.
        
        # We need to map sorted_P to the iteration order.
        # Let's create a list of vertices sorted by color descending, then by index.
        vertices_to_process = sorted(sorted_P, key=lambda x: -colors[x])
        
        for v in vertices_to_process:
            if time.time() - start_time > time_limit:
                return
            
            if len(R) + colors[v] <= best_size:
                break
            
            # Branch: include v
            R.append(v)
            # New P is intersection of current P and neighbors of v
            P_new = [u for u in P if u in adj_remapped[v]]
            max_clique_search(R, P_new)
            R.pop()
            
            # Branch: exclude v (remove v from P)
            # In the recursive call, we just don't include v.
            # But to maintain the bound, we must remove v from P for the "exclude" branch?
            # In Tomita's algorithm, the loop structure handles this.
            # The loop iterates through vertices. After processing v (include), 
            # the next iteration effectively excludes v by moving to the next vertex.
            # However, P must be updated to remove v for the remaining iterations?
            # Yes, P should be the set of vertices not yet processed.
            # So we remove v from P.
            P.remove(v)
            
    # Initial call
    # P is all vertices
    # We need to pass a list we can modify
    initial_P = list(range(n3))
    max_clique_search([], initial_P)
    
    if best_size > baseline:
        # Map back to original indices
        original_indices = [order[i] for i in best_clique]
        result = candidates3[original_indices]
        return result
    else:
        return candidates2
