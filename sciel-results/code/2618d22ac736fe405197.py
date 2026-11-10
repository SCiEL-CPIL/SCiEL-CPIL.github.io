import random
import time

def construct():
    d = 7
    n_vertices = 2 ** d
    all_edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
                
    # Precompute all faces (C4s)
    faces = []
    for x in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = x
                v2 = x ^ (1 << i)
                v3 = x ^ (1 << j)
                v4 = x ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v2, v4), max(v2, v4))
                e3 = (min(v4, v3), max(v4, v3))
                e4 = (min(v3, v1), max(v3, v1))
                
                faces.append((e1, e2, e3, e4))

    best_size = 0
    best_set = set()
    
    start_time = time.time()
    time_limit = 110
    
    # We will run multiple restarts
    while time.time() - start_time < time_limit:
        current_set = set(all_edges)
        removed_edges = []
        
        # Phase 1: Remove edges to eliminate all C4s
        # We loop until no C4 is found
        while True:
            if time.time() - start_time > time_limit:
                break
            found_c4 = False
            # To make it faster, we can shuffle faces each time? 
            # No, just iterate. If we find one, break.
            # The order of faces matters for which edge gets removed.
            # Shuffling faces per iteration might help diversify.
            # But shuffling 2688 items is O(N).
            # Let's just iterate.
            
            for face in faces:
                # Check if all 4 edges are in current_set
                if face[0] in current_set and face[1] in current_set and face[2] in current_set and face[3] in current_set:
                    found_c4 = True
                    # Remove a random edge from this face
                    idx = random.randint(0, 3)
                    edge_to_remove = face[idx]
                    current_set.remove(edge_to_remove)
                    removed_edges.append(edge_to_remove)
                    break
            
            if not found_c4:
                break
        
        # Phase 2: Try to re-add removed edges
        # Shuffle to avoid bias
        random.shuffle(removed_edges)
        for edge in removed_edges:
            if edge in current_set:
                continue
            u, v = edge
            diff = u ^ v
            k = diff.bit_length() - 1
            
            can_add = True
            for j in range(d):
                if j == k:
                    continue
                u_j = u ^ (1 << j)
                v_j = v ^ (1 << j)
                
                e1 = (min(u, u_j), max(u, u_j))
                e2 = (min(v, v_j), max(v, v_j))
                e3 = (min(u_j, v_j), max(u_j, v_j))
                
                if e1 in current_set and e2 in current_set and e3 in current_set:
                    can_add = False
                    break
            
            if can_add:
                current_set.add(edge)
        
        # Update best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)

    return [[u, v] for u, v in best_set]
