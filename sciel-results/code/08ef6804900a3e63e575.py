import time
import random
import itertools

def construct():
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin for 120s limit
    
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))
    
    # Generate all edges
    edges = []
    edge_to_idx = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    # Generate all 2D faces (squares)
    # A face is defined by 2 dimensions (i, j) and a vertex x such that x, x^i, x^j, x^i^j are vertices.
    # To avoid duplicates, we can iterate over all pairs of dimensions (i, j) and all vertices x,
    # but each face is counted 4 times (once for each corner). 
    # We can normalize the face by sorting its 4 vertices or 4 edges.
    # Better: Iterate over i < j. For each x, the face is {x, x^i, x^j, x^i^j}.
    # This generates each face 4 times (for each of the 4 vertices). 
    # We can just collect unique sets of 4 edges.
    
    faces = []
    face_set = set()
    edge_to_faces = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n_vertices):
                # Vertices of the face
                v1 = x
                v2 = x ^ (1 << i)
                v3 = x ^ (1 << j)
                v4 = x ^ (1 << i) ^ (1 << j)
                
                # Edges of the face
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v2, v4), max(v2, v4))
                e3 = (min(v4, v3), max(v4, v3))
                e4 = (min(v3, v1), max(v3, v1))
                
                face_edges = tuple(sorted([edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]]))
                
                if face_edges not in face_set:
                    face_set.add(face_edges)
                    face_idx = len(faces)
                    faces.append(face_edges)
                    for e_idx in face_edges:
                        edge_to_faces[e_idx].append(face_idx)

    n_faces = len(faces)
    
    def run_once(seed):
        random.seed(seed)
        
        # active_edge[i] = True if edge i is in the set
        active_edge = [True] * n_edges
        # face_count[f] = number of active edges in face f
        face_count = [4] * n_faces
        
        # Deletion Phase
        # We need to efficiently find faces with count 4.
        # A simple list of faces with count 4 is okay, but updates are O(1) per face.
        # However, removing an edge affects 4 faces (each edge is in 4 faces? 
        # In Q7, an edge (u,v) is in (d-1) * 2^(d-2) faces? 
        # No, an edge is in (d-1) * 2^(d-2) faces? 
        # Let's check: An edge is fixed in dimension i. The other d-1 dimensions can form faces.
        # For each other dimension j, and each combination of the remaining d-2 dimensions...
        # Actually, each edge is in (d-1) * 2^(d-2) faces?
        # For Q3, edge is in 2 faces. (3-1)*2^(1) = 4? No.
        # In Q3, total faces = 12. Total edges = 12. Each edge in 2 faces. 12*2/12 = 2.
        # In Q7, total faces = 672. Total edges = 448. 672*4/448 = 6.
        # So each edge is in exactly 6 faces.
        
        # We can maintain a set of "full" faces (count == 4)
        full_faces = set(range(n_faces))
        
        while full_faces:
            if time.time() - start_time > time_limit:
                break
                
            # Pick a random full face
            f_idx = random.choice(list(full_faces))
            full_faces.remove(f_idx)
            
            # If it's no longer full (due to previous removals in this loop iteration? No, we remove one edge at a time)
            # Wait, if we remove an edge, it affects 6 faces. Some of those might become non-full.
            # We should check if face_count[f_idx] is still 4.
            if face_count[f_idx] != 4:
                continue
                
            # Get the 4 edges in this face
            f_edges = faces[f_idx]
            
            # Choose an edge to remove
            # Heuristic: Remove the edge that is in the most "dense" area
            # Score = sum of face_count for all faces containing this edge
            best_edge = None
            best_score = -1
            for e_idx in f_edges:
                score = 0
                for f2 in edge_to_faces[e_idx]:
                    score += face_count[f2]
                if score > best_score:
                    best_score = score
                    best_edge = e_idx
            
            # Remove best_edge
            active_edge[best_edge] = False
            for f2 in edge_to_faces[best_edge]:
                face_count[f2] -= 1
                if face_count[f2] == 3:
                    if f2 in full_faces:
                        full_faces.remove(f2)
                # If it was 4, it's now 3, so remove from full_faces (already handled if it was in)
                # If it was <4, it stays out.
            
            # Note: We added f_idx to full_faces initially. If we remove an edge from f_idx, 
            # its count becomes 3. We should ensure it's not in full_faces.
            # We removed f_idx from full_faces at the start of the loop.
            # But what if f_idx was added back? No, we only add to full_faces if count becomes 4? 
            # No, we start with all 4. We only decrement. So count never increases.
            # So we never add to full_faces during deletion.
            
        # Local Search Phase: Add edges back
        # active_edge is now the set from deletion.
        # We want to add edges that don't create C4.
        # An edge e is addable if for all faces f containing e, face_count[f] < 3.
        
        # Initialize addable set
        addable = set()
        for e_idx in range(n_edges):
            if not active_edge[e_idx]:
                can_add = True
                for f_idx in edge_to_faces[e_idx]:
                    if face_count[f_idx] >= 3:
                        can_add = False
                        break
                if can_add:
                    addable.add(e_idx)
        
        while addable:
            if time.time() - start_time > time_limit:
                break
                
            # Pick a random addable edge
            e_idx = random.choice(list(addable))
            addable.remove(e_idx)
            
            # Add it
            active_edge[e_idx] = True
            
            # Update face counts and addable set
            for f_idx in edge_to_faces[e_idx]:
                old_count = face_count[f_idx]
                face_count[f_idx] += 1
                new_count = face_count[f_idx]
                
                # If old_count was 2, new_count is 3.
                # The missing edge in this face is now blocked.
                if old_count == 2:
                    # Find the missing edge in this face
                    for e2 in faces[f_idx]:
                        if not active_edge[e2]:
                            # e2 is the missing one
                            if e2 in addable:
                                addable.remove(e2)
                            break
                # If old_count was 3, we would have created a C4, but e_idx was addable, 
                # so old_count must have been < 3 for all faces.
                # So old_count can only be 0, 1, or 2.
                # If old_count was 2, it becomes 3.
                # If old_count was 0 or 1, it becomes 1 or 2. No edge becomes blocked.
            
        # Count final edges
        count = sum(active_edge)
        return count, active_edge

    best_count = 0
    best_edges = None
    
    # Run multiple times
    for i in range(1000): # Limit iterations, but check time
        if time.time() - start_time > time_limit:
            break
        
        count, active_edges = run_once(i)
        
        if count > best_count:
            best_count = count
            best_edges = active_edges
    
    # Construct output
    if best_edges is None:
        # Fallback to seed behavior if something went wrong
        d = 7
        chosen = set()
        out = []
        for x in range(2 ** d):
            for i in range(d):
                y = x ^ (1 << i)
                if x < y:
                    ok = True
                    for j in range(d):
                        if j != i:
                            c, e = x ^ (1 << j), y ^ (1 << j)
                            if all(p in chosen for p in ((min(x, c), max(x, c)), (min(y, e), max(y, e)), (min(c, e), max(c, e)))):
                                ok = False
                                break
                    if ok:
                        chosen.add((x, y))
                        out.append([x, y])
        return out

    out = []
    for i, is_active in enumerate(best_edges):
        if is_active:
            u, v = edges[i]
            out.append([u, v])
            
    return out
