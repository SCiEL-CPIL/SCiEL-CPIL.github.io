import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    time_limit = 110  # leave buffer for overhead
    
    # Generate support-size-3 candidates
    candidates = []
    for support in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Compute dot products
    dots = candidates @ candidates.T
    
    # Build compatibility graph (complement of conflict graph)
    # Compatible if dot <= 1
    # Conflict if dot > 1 (i.e., >= 2)
    # We want Max Clique in Compatible Graph
    # Adjacency matrix for Compatible Graph
    adj = np.zeros((n, n), dtype=bool)
    # Dot <= 1
    adj = (dots <= 1)
    np.fill_diagonal(adj, False)
    
    # Tomita's Maximum Clique Algorithm
    # We use a recursive branch and bound with greedy coloring.
    
    best_clique = []
    best_size = 0
    
    # Convert to list of lists for faster access in recursion
    adj_list = [np.where(adj[i])[0] for i in range(n)]
    
    def color_sort(P):
        """
        Greedy coloring of the subgraph induced by P.
        Returns list of (color, vertex) sorted by color descending.
        Also returns the max color.
        """
        if not P:
            return [], 0
        
        # Greedy coloring
        colors = {}
        color_classes = []
        remaining = set(P)
        
        while remaining:
            color_class = set()
            # Pick a vertex with max degree in remaining? Or just first?
            # Tomita uses a specific ordering. Let's use degree in P.
            # For simplicity and speed, we can use a heuristic.
            # A common heuristic is to order by degree descending.
            # But for coloring, we want to color sequentially.
            
            # Let's use the "DSATUR" or simple greedy.
            # Simple greedy:
            # 1. Order vertices in P by degree descending.
            # 2. Assign smallest available color.
            
            # To speed up, we can just use the order of P.
            # However, Tomita's algorithm relies on the coloring bound.
            
            # Let's implement a simple greedy coloring that returns the sequence.
            # This is a simplified version.
            
            # Actually, for Python, implementing full Tomita with dynamic re-coloring is complex.
            # We can use a static coloring based on the order of P.
            
            # Let's try a simple approach:
            # 1. Order P by degree in the induced subgraph (descending).
            # 2. Color them greedily.
            
            degrees_in_P = {v: len([u for u in P if u in adj_list[v]]) for v in P}
            ordered_P = sorted(P, key=lambda x: -degrees_in_P[x])
            
            current_color = 0
            for v in ordered_P:
                # Find smallest color not used by neighbors in already colored set
                # This is O(N^2).
                # Since N=1320, this might be slow if done at every node.
                
                # Optimization: We only need the max color for the bound.
                # And we need to partition P into color classes for the loop.
                
                # Let's use a standard greedy coloring routine.
                pass
            
            # This implementation is getting heavy.
            # Let's use a simpler bound: len(P).
            # But that's weak.
            
            # Alternative: Use `networkx`? No, standard lib only.
            # Let's implement a basic greedy coloring.
            
            break # Placeholder
            
        # This function is incomplete. I need a robust Tomita implementation.
        
    # Given the complexity of writing a full Tomita from scratch in Python within the edit constraints,
    # and the risk of timeout, I should consider if there's a simpler improvement.
    # The seed program uses a greedy MIS.
    # If I just use a better greedy (e.g., Local Search with simulated annealing or just better swaps),
    # I might get close to the optimal.
    # However, the directive SPECIFICALLY asks for Tomita.
    
    # Let's try to implement a simplified Tomita.
    
    def max_clique(R, P):
        nonlocal best_size, best_clique
        
        if time.time() - start_time > time_limit:
            return
        
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
            
        # Coloring Bound
        # We need to color P.
        # If max_color <= len(R), we can prune.
        
        # Simple Coloring:
        # We will use a greedy coloring that produces color classes.
        # We sort vertices by degree in P descending.
        
        P_list = list(P)
        # Degree in P
        degs = {v: sum(1 for u in P_list if u in adj_list[v]) for v in P_list}
        P_list.sort(key=lambda x: -degs[x])
        
        colors = {}
        color_classes = []
        
        for v in P_list:
            # Find smallest color
            used_colors = set()
            for u in P_list:
                if u in adj_list[v] and u in colors:
                    used_colors.add(colors[u])
            
            c = 0
            while c in used_colors:
                c += 1
            
            colors[v] = c
            if c >= len(color_classes):
                color_classes.append([])
            color_classes[c].append(v)
            
        max_color = len(color_classes)
        
        if len(R) + max_color <= best_size:
            return
            
        # Branch and Bound
        # Iterate through color classes in reverse order (highest color first)
        for c in range(max_color - 1, -1, -1):
            if len(R) + c <= best_size:
                break
                
            # Process vertices in color class c
            for v in color_classes[c]:
                if v not in P: # Might have been removed? No, P is static for this call
                    continue
                    
                R.append(v)
                P_new = P.intersection(adj_list[v])
                max_clique(R, P_new)
                R.pop()
                
                P.discard(v)
                
                if len(R) + len(P) <= best_size:
                    break
                    
    # Initialize
    P_initial = set(range(n))
    # To help the search, we can start with a good lower bound.
    # Use greedy clique
    R = []
    P_greedy = set(range(n))
    while P_greedy:
        # Pick vertex with max degree in P_greedy
        v = max(P_greedy, key=lambda x: len(P_greedy.intersection(adj_list[x])))
        R.append(v)
        P_greedy = P_greedy.intersection(adj_list[v])
        
    if len(R) > best_size:
        best_size = len(R)
        best_clique = list(R)
        
    # Run Tomita
    max_clique([], set(range(n)))
    
    if best_size > 220:
        result = candidates[best_clique]
        return result
    else:
        # Fallback to support-2 if it fails or is smaller (shouldn't happen)
        candidates2 = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    candidates2.append(v)
        return np.array(candidates2, dtype=np.int64)
