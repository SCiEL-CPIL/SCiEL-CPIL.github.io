import random
import time

def construct():
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))
    
    # 1. Map edges to indices and store endpoints
    edge_to_idx = {}
    idx_to_edge = []
    
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = idx
                idx_to_edge.append((u, v))
                idx += 1
                
    # 2. Build Conflict Graph (Adjacency List)
    # Two edges conflict if they share a C4 face.
    # For each edge, identify its dimension and endpoints.
    # An edge (u, v) where v = u ^ (1<<k) has dimension k.
    # It is in faces (k, j) for all j != k.
    # In face (k, j), the other edges are:
    # e_par: (u ^ (1<<j), v ^ (1<<j))
    # e_adj1: (u, u ^ (1<<j))
    # e_adj2: (v, v ^ (1<<j))
    
    adj = [set() for _ in range(n_edges)]
    
    for idx in range(n_edges):
        u, v = idx_to_edge[idx]
        # Determine dimension k
        diff = u ^ v
        k = diff.bit_length() - 1
        
        for j in range(d):
            if j == k:
                continue
            
            # Calculate other 3 edges in the face
            u_j = u ^ (1 << j)
            v_j = v ^ (1 << j)
            
            e_par = (min(u_j, v_j), max(u_j, v_j))
            e_adj1 = (min(u, u_j), max(u, u_j))
            e_adj2 = (min(v, v_j), max(v, v_j))
            
            # Get indices
            # Note: These edges are guaranteed to exist
            adj[idx].add(edge_to_idx[e_par])
            adj[idx].add(edge_to_idx[e_adj1])
            adj[idx].add(edge_to_idx[e_adj2])
            
    # 3. Heuristic: Greedy MIS + Local Search with Restarts
    best_size = 0
    best_set = set()
    
    start_time = time.time()
    num_restarts = 0
    
    while time.time() - start_time < 110: # Leave 10s buffer
        num_restarts += 1
        random.seed(num_restarts) # Different randomization for each restart
        
        # --- Greedy Phase ---
        available = set(range(n_edges))
        selected = set()
        
        # To improve greedy, we can pick nodes with min degree in available subgraph
        # But calculating degrees dynamically is expensive.
        # Simple heuristic: Pick random node, or node with min original degree (all same).
        # Let's try picking randomly to break symmetry, or by index.
        # Random picking often works well with restarts.
        
        while available:
            # Pick a node. We can pick the one with fewest neighbors in 'available'
            # to maximize future choices? Or just random.
            # Let's try random for simplicity and diversity.
            u = random.choice(list(available))
            selected.add(u)
            
            # Remove neighbors
            neighbors = [v for v in adj[u] if v in available]
            for v in neighbors:
                available.remove(v)
            available.remove(u)
            
        # --- Local Search Phase ---
        improved = True
        while improved:
            improved = False
            
            # 1. Try to add any available edge
            # We iterate through all edges not in selected
            for v in range(n_edges):
                if v in selected:
                    continue
                # Check if v has any neighbor in selected
                conflict = False
                for u in adj[v]:
                    if u in selected:
                        conflict = True
                        break
                if not conflict:
                    selected.add(v)
                    improved = True
                    break # Restart loop to catch new opportunities
            
            if improved:
                continue
                
            # 2. Try 1-2 Swap (Remove 1, Add 2)
            # Iterate through selected edges to remove
            for u in list(selected):
                # If we remove u, which edges become available?
                # An edge v (not in selected, v != u) becomes available if
                # all its neighbors in (selected \ {u}) are empty.
                # i.e., N(v) ∩ (selected \ {u}) = ∅
                
                # Optimization: Pre-compute candidates?
                # Just check all v not in selected.
                candidates = []
                for v in range(n_edges):
                    if v in selected or v == u:
                        continue
                    
                    # Check conflict with selected \ {u}
                    conflict = False
                    for w in adj[v]:
                        if w in selected and w != u:
                            conflict = True
                            break
                    if not conflict:
                        candidates.append(v)
                        
                if len(candidates) >= 2:
                    # Find two non-adjacent candidates
                    found = False
                    # Randomize order of candidates to help escape local minima?
                    # Or just first pair.
                    for i in range(len(candidates)):
                        for j in range(i + 1, len(candidates)):
                            v1 = candidates[i]
                            v2 = candidates[j]
                            if v2 not in adj[v1]:
                                # Valid swap
                                selected.remove(u)
                                selected.add(v1)
                                selected.add(v2)
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break # Restart outer loop
                
        # Update best
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
    # Construct result from best_set
    result = []
    for e_idx in best_set:
        u, v = idx_to_edge[e_idx]
        result.append([u, v])
        
    return result
