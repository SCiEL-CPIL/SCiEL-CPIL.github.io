import itertools

def construct():
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))
    
    # 1. Map edges to indices
    # Edge is defined by (min(u, v), max(u, v))
    # We need a mapping from edge tuple to index
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
                
    # 2. Generate all C4 faces
    # A face is defined by a base vertex x and two dimensions i, j
    # To get unique faces, we can just generate all and use a set of frozensets of edge indices
    faces = set()
    for x in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v0 = x
                v1 = x ^ (1 << i)
                v2 = x ^ (1 << j)
                v3 = x ^ (1 << i) ^ (1 << j)
                
                # Edges: (v0, v1), (v1, v3), (v3, v2), (v2, v0)
                e1 = (min(v0, v1), max(v0, v1))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v3, v2), max(v3, v2))
                e4 = (min(v2, v0), max(v2, v0))
                
                # Get indices
                try:
                    i1 = edge_to_idx[e1]
                    i2 = edge_to_idx[e2]
                    i3 = edge_to_idx[e3]
                    i4 = edge_to_idx[e4]
                    face = frozenset([i1, i2, i3, i4])
                    faces.add(face)
                except KeyError:
                    # Should not happen if logic is correct
                    pass
                    
    # Convert to list for easier iteration
    face_list = list(faces)
    n_faces = len(face_list)
    
    # 3. Greedy Hitting Set
    # active_faces: set of indices of faces that are not yet hit
    active_faces = set(range(n_faces))
    
    # face_count[e]: number of active faces that contain edge e
    face_count = [0] * n_edges
    
    # Initialize counts
    # Each edge is in d-1 = 6 faces
    # But we must be careful: are all faces unique? Yes, we used a set.
    # Let's just compute counts from face_list to be safe
    for f in face_list:
        for e in f:
            face_count[e] += 1
            
    removed = [False] * n_edges
    removed_count = 0
    
    # We need to efficiently find max face_count among non-removed edges
    # Since n_edges is small (448), linear scan is fine.
    
    while active_faces:
        max_count = -1
        best_edge = -1
        
        for e in range(n_edges):
            if not removed[e] and face_count[e] > max_count:
                max_count = face_count[e]
                best_edge = e
                
        if best_edge == -1:
            break
            
        # Remove best_edge
        removed[best_edge] = True
        removed_count += 1
        
        # Update active faces and counts
        # Find all active faces containing best_edge
        # We can iterate through all active faces? No, that's slow if we do it naively.
        # Better: Precompute for each edge, which faces it belongs to.
        
        # Let's precompute edge_to_faces
        # This is better done before the loop
        
    # Wait, I need to precompute edge_to_faces before the while loop to make updates efficient.
    
    # Let's restructure slightly.
    
    # Precompute edge_to_faces: list of lists
    edge_to_faces = [[] for _ in range(n_edges)]
    for f_idx, f in enumerate(face_list):
        for e in f:
            edge_to_faces[e].append(f_idx)
            
    # Re-initialize state
    active_faces = set(range(n_faces))
    face_count = [0] * n_edges
    for f in face_list:
        for e in f:
            face_count[e] += 1
            
    removed = [False] * n_edges
    
    while active_faces:
        max_count = -1
        best_edge = -1
        
        for e in range(n_edges):
            if not removed[e] and face_count[e] > max_count:
                max_count = face_count[e]
                best_edge = e
                
        if best_edge == -1:
            break
            
        removed[best_edge] = True
        
        # Update counts for all faces containing best_edge
        for f_idx in edge_to_faces[best_edge]:
            if f_idx in active_faces:
                active_faces.remove(f_idx)
                # Decrement count for other edges in this face
                for e in face_list[f_idx]:
                    if e != best_edge:
                        face_count[e] -= 1
                        
    # Construct result
    result = []
    for e_idx in range(n_edges):
        if not removed[e_idx]:
            u, v = idx_to_edge[e_idx]
            result.append([u, v])
            
    return result
