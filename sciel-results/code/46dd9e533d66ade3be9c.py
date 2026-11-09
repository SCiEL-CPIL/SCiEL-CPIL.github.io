import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 16
    
    # --- 1. Base Graph: Grötzsch Graph (11 vertices, 20 edges) ---
    # This is a triangle-free graph with chromatic number 4.
    # It serves as a strong seed because it already satisfies the core constraint.
    groetzsch_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],  # C5
        [5, 0], [5, 2],
        [6, 1], [6, 3],
        [7, 2], [7, 4],
        [8, 3], [8, 0],
        [9, 4], [9, 1],
        [10, 5], [10, 6], [10, 7], [10, 8], [10, 9]
    ]
    
    # Initialize adjacency as bitmasks for speed
    adj = [0] * n
    for u, v in groetzsch_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    current_edges = len(groetzsch_edges)
    best_edges = current_edges
    best_adj = adj[:]

    # Precompute all possible edges not in the base graph
    all_possible_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] & (1 << v)):
                all_possible_edges.append((u, v))

    # --- 2. Optimized 3-Colorability Check ---
    # Uses bitmasks and recursive backtracking.
    # Optimized with vertex ordering (highest degree first) and forward checking (optional, but simple ordering helps).
    
    def is_not_3_colorable(adj_masks):
        # Precompute degree-based order for better pruning
        # We want to color high-degree vertices first.
        order = list(range(n))
        order.sort(key=lambda x: bin(adj_masks[x]).count('1'), reverse=True)
        
        colors = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            # Determine forbidden colors based on neighbors
            forbidden = 0
            mask = adj_masks[v]
            
            # Iterate through neighbors of v
            m = mask
            while m:
                lsb = m & -m
                nb = lsb.bit_length() - 1
                if colors[nb] != -1:
                    forbidden |= (1 << colors[nb])
                m ^= lsb
            
            # Try assigning a color not in forbidden
            for c in range(3):
                if not (forbidden & (1 << c)):
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return not backtrack(0)

    # --- 3. Local Search: Greedy Additions with Backtracking ---
    # Strategy: 
    # 1. Try to add edges greedily. If it preserves triangle-freeness AND non-3-colorability, keep it.
    # 2. If stuck (no simple addition works), try a "move": remove one edge, then greedily add back.
    # 3. Use randomization to escape local optima.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Shuffle candidates to ensure variety in which edges are tried
        candidates = all_possible_edges[:]
        random.shuffle(candidates)
        
        # --- Phase 1: Greedy Addition ---
        for u, v in candidates:
            if time.time() > deadline:
                break
                
            # Check if edge (u,v) is already present
            if (adj[u] & (1 << v)):
                continue
                
            # Check triangle-freeness:
            # Adding (u,v) creates a triangle if there is a common neighbor w.
            # i.e., adj[u] & adj[v] != 0
            if (adj[u] & adj[v]) != 0:
                continue
            
            # Tentatively add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Check 3-colorability
            if is_not_3_colorable(adj):
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
                improved = True
            else:
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)

        # --- Phase 2: Edge Swap (if no improvement in Phase 1) ---
        if not improved:
            # Try removing a random edge from the current best (or current state if it's better)
            # We operate on a copy of the best_adj to avoid destroying the best known state
            # unless we find something better.
            
            # Pick a random edge present in best_adj to remove
            # First, build a list of edges in best_adj
            current_best_edges_list = []
            for u in range(n):
                m = best_adj[u]
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    if u < v:
                        current_best_edges_list.append((u, v))
                    m ^= lsb
            
            if current_best_edges_list:
                rem_u, rem_v = random.choice(current_best_edges_list)
                
                # Start from best_adj
                trial_adj = best_adj[:]
                trial_edges = best_edges
                
                # Remove (rem_u, rem_v)
                trial_adj[rem_u] ^= (1 << rem_v)
                trial_adj[rem_v] ^= (1 << rem_u)
                trial_edges -= 1
                
                # Now try to add edges greedily to the trial graph
                # Use a fresh shuffle
                trial_candidates = all_possible_edges[:]
                random.shuffle(trial_candidates)
                
                for u, v in trial_candidates:
                    if time.time() > deadline:
                        break
                        
                    if (trial_adj[u] & (1 << v)):
                        continue
                        
                    if (trial_adj[u] & trial_adj[v]) != 0:
                        continue
                        
                    trial_adj[u] |= (1 << v)
                    trial_adj[v] |= (1 << u)
                    
                    if is_not_3_colorable(trial_adj):
                        trial_edges += 1
                    else:
                        trial_adj[u] ^= (1 << v)
                        trial_adj[v] ^= (1 << u)
                
                # If we added more than we removed (net gain > 0) or even just maintained but found a different configuration?
                # Actually, we only accept if we strictly improve edge count.
                if trial_edges > best_edges:
                    best_edges = trial_edges
                    best_adj = trial_adj
                    improved = True
                    # Sync current working state to the new best for subsequent greedy additions
                    adj = best_adj[:]
                    current_edges = best_edges

    # --- 4. Final Validation ---
    # Ensure the best_adj is valid
    if not is_not_3_colorable(best_adj):
        # Fallback to base
        final_adj = [0] * n
        for u, v in groetzsch_edges:
            final_adj[u] |= (1 << v)
            final_adj[v] |= (1 << u)
        return groetzsch_edges
        
    # Check triangle-free on best_adj
    valid = True
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] & (1 << v)):
                if (best_adj[u] & best_adj[v]) != 0:
                    valid = False
                    break
        if not valid:
            break
            
    if not valid:
        return groetzsch_edges

    # --- 5. Convert back to list of edges ---
    result_edges = []
    for u in range(n):
        mask = best_adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if u < v:
                result_edges.append([u, v])
            mask ^= lsb
            
    return result_edges
