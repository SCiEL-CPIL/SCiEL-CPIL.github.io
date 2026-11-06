import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # --- Step 1: Construct Polarity Graph of PG(2, 7) ---
    q = 7
    # Points of PG(2, 7)
    # (1, y, z) for y,z in 0..6
    # (0, 1, z) for z in 0..6
    # (0, 0, 1)
    points = []
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n_total = len(points) # 57
    
    # Adjacency matrix
    # P=(a,b,c), Q=(d,e,f)
    # a*f + b*e + c*d == 0 (mod q)
    adj = [[False] * n_total for _ in range(n_total)]
    for i in range(n_total):
        a, b, c = points[i]
        for j in range(i + 1, n_total):
            d, e, f = points[j]
            if (a * f + b * e + c * d) % q == 0:
                adj[i][j] = True
                adj[j][i] = True
                
    # --- Step 2: Select Initial Subset ---
    target_n = 43
    all_indices = list(range(n_total))
    random.shuffle(all_indices)
    current_selected = set(all_indices[:target_n])
    
    # Helper: Check if there is an independent set of size 11 in the induced subgraph of current_selected
    # Returns the set if found, else None
    def find_independent_set_size_k(selected_set, k):
        # Simple backtracking
        # Order vertices by degree in the induced subgraph (ascending) to prune faster?
        # Or just use the set.
        candidates = list(selected_set)
        # Sort by degree in the full graph? Or induced?
        # Induced degree is better for pruning.
        def get_induced_degree(v):
            deg = 0
            for u in selected_set:
                if u != v and adj[v][u]:
                    deg += 1
            return deg
            
        # Heuristic: Try to build an independent set
        # If we just want to know IF one exists, we can use a simple recursive search.
        # Given the small size (43), this should be fast if we prune well.
        
        # Sort candidates by degree (low degree first? No, high degree first for coloring? 
        # For independent set, picking low degree vertices is often better for finding large ones?
        # Actually, for finding ANY independent set of size k, picking vertices with many neighbors (high degree) 
        # removes more candidates, which helps prune the "exclude" branch? 
        # Let's just use a simple DFS.
        
        candidates.sort(key=lambda v: get_induced_degree(v))
        
        def dfs(idx, current_set, remaining_candidates):
            if len(current_set) == k:
                return current_set
            if idx >= len(remaining_candidates):
                return None
            if len(current_set) + (len(remaining_candidates) - idx) < k:
                return None
                
            v = remaining_candidates[idx]
            
            # Option 1: Include v
            new_current = current_set + [v]
            # New candidates must not be neighbors of v
            new_remaining = [u for u in remaining_candidates[idx+1:] if not adj[v][u]]
            res = dfs(idx + 1, new_current, new_remaining)
            if res:
                return res
            
            # Option 2: Exclude v
            res = dfs(idx + 1, current_set, remaining_candidates[idx+1:])
            if res:
                return res
            
            return None
            
        return dfs(0, [], candidates)

    # --- Step 3: Repair Loop ---
    # We want to ensure no independent set of size 11 exists.
    # If we find one, we remove a vertex from it.
    
    # Optimization: Only run the check if we have time.
    while time.time() < deadline - 5.0:
        # Check for independent set of size 11
        # To speed up, we can limit the search depth or use a timeout on the search itself?
        # The DFS above is exponential in worst case, but for C4-free graphs, it's usually fast.
        # However, to be safe against a pathological case, we might need a faster heuristic check.
        # But 43 vertices is small.
        
        # Let's try to find one.
        # If the graph is "good", this returns quickly.
        # If it's "bad", it might take long.
        # We'll assume it's fast enough for 43 vertices.
        
        iso_set = find_independent_set_size_k(current_selected, 11)
        
        if iso_set is None:
            # No independent set of size 11 found.
            # We can try to verify with a slightly larger k? No, we need exactly < 11.
            # If we can't find 11, it's likely <= 10.
            break
        else:
            # Found an independent set of size 11.
            # Remove one vertex from this set.
            # Which one? Random.
            v_to_remove = random.choice(iso_set)
            current_selected.remove(v_to_remove)
            
            # Add a new random vertex from the pool not in current_selected
            available = [v for v in all_indices if v not in current_selected]
            if available:
                v_to_add = random.choice(available)
                current_selected.add(v_to_add)
            else:
                # No vertices left to add? We reduced n?
                # We must maintain n=43? The task says "n" is the size.
                # If we remove one and can't add, n becomes 42.
                # The task says "Maximize n".
                # If we can't maintain 43, we return 42.
                # But we should try to find a replacement.
                pass

    # --- Step 4: Construct Output ---
    final_vertices = sorted(list(current_selected))
    n_final = len(final_vertices)
    
    # Map old indices to new indices 0..n_final-1
    old_to_new = {old: new for new, old in enumerate(final_vertices)}
    
    edges = []
    for i in range(n_final):
        old_i = final_vertices[i]
        for j in range(i + 1, n_final):
            old_j = final_vertices[j]
            if adj[old_i][old_j]:
                edges.append([i, j])
                
    return {'n': n_final, 'edges': edges}
