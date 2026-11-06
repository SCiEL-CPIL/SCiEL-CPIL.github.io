import time

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))
    
    # Map edges to indices 0..447
    edges = []
    edge_to_id = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_id[(u, v)] = len(edges)
                edges.append((u, v))
                
    # Precompute faces
    edge_faces = [[] for _ in range(n_edges)]
    faces = [] # List of lists of 4 edge IDs
    
    for i in range(d):
        for j in range(i + 1, d):
            others = [k for k in range(d) if k != i and k != j]
            for mask in range(2 ** len(others)):
                u = 0
                for idx, bit in enumerate(others):
                    if mask & (1 << idx):
                        u |= (1 << bit)
                
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = edge_to_id[(min(v1, v2), max(v1, v2))]
                e2 = edge_to_id[(min(v3, v4), max(v3, v4))]
                e3 = edge_to_id[(min(v1, v3), max(v1, v3))]
                e4 = edge_to_id[(min(v2, v4), max(v2, v4))]
                
                face_id = len(faces)
                faces.append([e1, e2, e3, e4])
                
                for e in [e1, e2, e3, e4]:
                    edge_faces[e].append(face_id)
                    
    n_faces = len(faces)
    
    # Precompute for fast propagation:
    # For each face and each edge in it, the list of other 3 edges.
    face_other_edges = [{} for _ in range(n_faces)]
    for f_idx, f_edges in enumerate(faces):
        for e in f_edges:
            face_other_edges[f_idx][e] = [x for x in f_edges if x != e]
    
    start_time = time.time()
    timeout = 110
    
    def time_exceeded():
        return time.time() - start_time > timeout

    best_count = 0
    best_selected_edges = []
    selected_edges = []
    
    # B&B
    def branch_and_bound(available_mask, selected_count, face_counts):
        nonlocal best_count, best_selected_edges, selected_edges, time_exceeded
        
        if time_exceeded():
            return

        ub = selected_count + available_mask.bit_count()
        if ub <= best_count:
            return

        if available_mask == 0:
            if selected_count > best_count:
                best_count = selected_count
                best_selected_edges = selected_edges[:]
            return

        # Heuristic: Pick edge in face with count 2
        candidate_e = -1
        m = available_mask
        # To speed up, we can limit the search for heuristic if time is tight?
        # No, let's try to find one.
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            for f_idx in edge_faces[e_idx]:
                if face_counts[f_idx] == 2:
                    candidate_e = e_idx
                    m = 0
                    break
            if candidate_e != -1:
                break
            m ^= lsb
            
        if candidate_e == -1:
            lsb = available_mask & -available_mask
            candidate_e = lsb.bit_length() - 1

        # Branch 1: Select candidate_e
        selected_edges.append(candidate_e)
        new_available_mask = available_mask & ~(1 << candidate_e)
        new_face_counts = face_counts[:]
        
        for f_idx in edge_faces[candidate_e]:
            new_face_counts[f_idx] += 1
            if new_face_counts[f_idx] == 3:
                others = face_other_edges[f_idx][candidate_e]
                for other_e in others:
                    if new_available_mask & (1 << other_e):
                        new_available_mask &= ~(1 << other_e)
        
        branch_and_bound(new_available_mask, selected_count + 1, new_face_counts)
        
        selected_edges.pop()
        
        # Branch 2: Do not select candidate_e
        new_available_mask2 = available_mask & ~(1 << candidate_e)
        branch_and_bound(new_available_mask2, selected_count, face_counts)

    initial_mask = (1 << n_edges) - 1
    initial_face_counts = [0] * n_faces
    
    branch_and_bound(initial_mask, 0, initial_face_counts)
    
    result = []
    for e_id in best_selected_edges:
        result.append([edges[e_id][0], edges[e_id][1]])
        
    return result
