import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    d = 7
    n = 1 << d  # 128 vertices
    total_edges = d * (1 << (d - 1))  # 448
    
    # Precompute all edges as (u, v) with u < v
    edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
    
    # Precompute for each edge, the 6 other edges that share a 2-face with it
    # An edge (u, v) where v = u ^ (1<<i) shares a 2-face with edges in direction j (j != i)
    # The 2-face has vertices: u, v, u^(1<<j), v^(1<<j)
    # The 4 edges of this face are:
    #   (u, v) - direction i
    #   (u^(1<<j), v^(1<<j)) - direction i (parallel)
    #   (u, u^(1<<j)) - direction j
    #   (v, v^(1<<j)) - direction j
    # So for edge (u,v) in direction i, the other 3 edges in each 2-face (for j != i) are:
    #   (min(u^(1<<j), v^(1<<j)), max(u^(1<<j), v^(1<<j)))
    #   (min(u, u^(1<<j)), max(u, u^(1<<j)))
    #   (min(v, v^(1<<j)), max(v, v^(1<<j)))
    
    # Build edge index: edge (u,v) -> index
    edge_to_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = idx
    
    # For each edge, find all edges that form a 4-cycle with it
    # Two edges form a 4-cycle if they are in the same 2-face and are not opposite (parallel)
    # Actually, a 4-cycle in a 2-face consists of 4 edges. We need to track when all 4 are present.
    # Better approach: for each 2-face, track which of its 4 edges are present.
    
    # Enumerate all 2-faces. A 2-face is defined by choosing 2 dimensions (i, j) and a vertex u
    # such that u has 0 in bits i and j (canonical representative).
    # The 4 vertices are: u, u^(1<<i), u^(1<<j), u^(1<<i)^(1<<j)
    # The 4 edges are:
    #   (u, u^(1<<i)) - direction i
    #   (u^(1<<j), u^(1<<i)^(1<<j)) - direction i
    #   (u, u^(1<<j)) - direction j
    #   (u^(1<<i), u^(1<<i)^(1<<j)) - direction j
    
    faces = []  # list of 4 edge indices for each 2-face
    for i in range(d):
        for j in range(i + 1, d):
            # Fix bits i and j to 0, vary all other bits
            for mask in range(1 << d):
                if (mask & (1 << i)) or (mask & (1 << j)):
                    continue
                u = mask
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(u, v1), max(u, v1))
                e2 = (min(v2, v3), max(v2, v3))
                e3 = (min(u, v2), max(u, v2))
                e4 = (min(v1, v3), max(v1, v3))
                
                face_edges = [edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]]
                faces.append(face_edges)
    
    # face_count[f] = number of edges in face f that are currently selected
    face_count = [0] * len(faces)
    
    # For each edge, list of faces it belongs to
    edge_faces = [[] for _ in range(len(edges))]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_faces[e_idx].append(f_idx)
    
    # Greedy construction with random restarts
    def greedy_init(seed):
        rng = random.Random(seed)
        selected = [False] * len(edges)
        fc = [0] * len(faces)
        count = 0
        
        order = list(range(len(edges)))
        rng.shuffle(order)
        
        for e_idx in order:
            # Check if adding this edge would complete any 4-cycle
            can_add = True
            for f_idx in edge_faces[e_idx]:
                if fc[f_idx] == 3:
                    can_add = False
                    break
            if can_add:
                selected[e_idx] = True
                count += 1
                for f_idx in edge_faces[e_idx]:
                    fc[f_idx] += 1
        
        return selected, fc, count
    
    # Local search: remove an edge, then try to add back previously blocked edges
    def local_search(selected, fc, count, seed, time_limit):
        rng = random.Random(seed)
        best_count = count
        best_selected = selected[:]
        best_fc = fc[:]
        
        # Build list of edges NOT selected
        unselected = [i for i in range(len(edges)) if not selected[i]]
        
        # For each unselected edge, find which faces are blocking it (fc[f] == 3)
        def get_blockers(e_idx):
            blockers = []
            for f_idx in edge_faces[e_idx]:
                if best_fc[f_idx] == 3:
                    blockers.append(f_idx)
            return blockers
        
        # Try 1-move: remove one edge, then greedily add back
        for _ in range(2000):
            if time.time() > time_limit:
                break
            
            # Pick a random selected edge to remove
            selected_edges = [i for i in range(len(edges)) if selected[i]]
            if not selected_edges:
                break
            rem_idx = rng.choice(selected_edges)
            
            # Remove it
            selected[rem_idx] = False
            count -= 1
            for f_idx in edge_faces[rem_idx]:
                fc[f_idx] -= 1
            
            # Now try to add back all unselected edges that can now be added
            improved = True
            while improved:
                improved = False
                for e_idx in unselected:
                    if selected[e_idx]:
                        continue
                    can_add = True
                    for f_idx in edge_faces[e_idx]:
                        if fc[f_idx] == 3:
                            can_add = False
                            break
                    if can_add:
                        selected[e_idx] = True
                        count += 1
                        for f_idx in edge_faces[e_idx]:
                            fc[f_idx] += 1
                        improved = True
            
            if count > best_count:
                best_count = count
                best_selected = selected[:]
                best_fc = fc[:]
            
            # Restore original state
            selected[rem_idx] = True
            count += 1
            for f_idx in edge_faces[rem_idx]:
                fc[f_idx] += 1
        
        return best_selected, best_fc, best_count
    
    # Try multiple random initializations
    best_overall = 0
    best_selected_overall = None
    best_fc_overall = None
    
    num_restarts = 50
    for seed in range(num_restarts):
        if time.time() > deadline:
            break
        
        selected, fc, count = greedy_init(seed)
        
        # Local search
        time_limit = min(deadline - 1.0, time.time() + 2.0)
        selected, fc, count = local_search(selected, fc, count, seed, time_limit)
        
        if count > best_overall:
            best_overall = count
            best_selected_overall = selected
            best_fc_overall = fc
    
    # Final polish: try to add any edge that can be added
    if best_selected_overall is not None:
        improved = True
        while improved:
            improved = False
            for e_idx in range(len(edges)):
                if best_selected_overall[e_idx]:
                    continue
                can_add = True
                for f_idx in edge_faces[e_idx]:
                    if best_fc_overall[f_idx] == 3:
                        can_add = False
                        break
                if can_add:
                    best_selected_overall[e_idx] = True
                    best_overall += 1
                    for f_idx in edge_faces[e_idx]:
                        best_fc_overall[f_idx] += 1
                    improved = True
    
    # Convert to output format
    result = []
    for e_idx, (u, v) in enumerate(edges):
        if best_selected_overall[e_idx]:
            result.append([u, v])
    
    return result
