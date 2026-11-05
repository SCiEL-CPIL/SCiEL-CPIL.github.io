import random
import time
import sys

def construct():
    d = 7
    n_vertices = 2 ** d
    total_edges = n_vertices * d // 2  # 128 * 7 / 2 = 448
    
    # Pre-compute edge IDs and face mappings
    # Edge ID: (u, i) where u < u ^ (1<<i)
    # We will index edges 0 to 447
    edges = []
    edge_id_map = {}
    
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                eid = len(edges)
                edges.append((u, v, i))
                edge_id_map[(u, v)] = eid
                edge_id_map[(v, u)] = eid
            else:
                # u > v, so the edge is (v, u) with dim i
                # v ^ (1<<i) == u
                # The edge ID is already created when we processed v?
                # No, we only add if u < v.
                # So if u > v, the edge (v, u) was added when we iterated v.
                # We just need to find its ID.
                pass
    
    # Actually, let's just build the list properly.
    edges = []
    edge_id_map = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                eid = len(edges)
                edges.append((u, v, i))
                edge_id_map[(u, v)] = eid
                edge_id_map[(v, u)] = eid
    
    # Build face mappings
    # Face ID: (min_v, min_i, max_i)
    # min_v is the minimum vertex in the face
    # min_i, max_i are the dimensions
    
    face_id_map = {}
    face_edges_map = {} # face_id -> list of 4 edge IDs
    edge_to_faces = [[] for _ in range(total_edges)]
    
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                
                verts = [u, v1, v2, v3]
                min_v = min(verts)
                fid = (min_v, i, j)
                
                if fid not in face_id_map:
                    face_id_map[fid] = len(face_id_map)
                    face_edges_map[face_id_map[fid]] = []
                
                fid_idx = face_id_map[fid]
                
                # Determine the 4 edges of this face
                # Edge 1: (u, v1) dim i
                e1 = edge_id_map[(min(u, v1), max(u, v1))]
                # Edge 2: (u, v2) dim j
                e2 = edge_id_map[(min(u, v2), max(u, v2))]
                # Edge 3: (v1, v3) dim j
                e3 = edge_id_map[(min(v1, v3), max(v1, v3))]
                # Edge 4: (v2, v3) dim i
                e4 = edge_id_map[(min(v2, v3), max(v2, v3))]
                
                edge_list = [e1, e2, e3, e4]
                
                # Store in face_edges_map (avoid duplicates if we encounter same face from different u)
                # Since we use min_v, and the set of edges is unique to the face,
                # we can just check if we've seen this face before.
                # If fid_idx is new, we add.
                # If it exists, we should have the same edges.
                if not face_edges_map[fid_idx]:
                    face_edges_map[fid_idx] = edge_list
                else:
                    # Verify consistency (optional)
                    pass
                
                # Update edge_to_faces
                for e in edge_list:
                    edge_to_faces[e].append(fid_idx)
    
    num_faces = len(face_id_map)
    
    # Local Search
    def greedy_init():
        selected = [False] * total_edges
        face_counts = [0] * num_faces
        # Random order
        edge_order = list(range(total_edges))
        random.shuffle(edge_order)
        
        for e in edge_order:
            can_add = True
            for f in edge_to_faces[e]:
                if face_counts[f] == 3:
                    can_add = False
                    break
            if can_add:
                selected[e] = True
                for f in edge_to_faces[e]:
                    face_counts[f] += 1
        return selected, face_counts
    
    def try_add(selected, face_counts, e):
        for f in edge_to_faces[e]:
            if face_counts[f] == 3:
                return False
        # Add
        selected[e] = True
        for f in edge_to_faces[e]:
            face_counts[f] += 1
        return True
    
    def try_swap(selected, face_counts, e_out, e_in):
        # e_out is selected, e_in is not
        # Check validity
        # Faces affected are those containing e_in or e_out
        # If a face contains e_in but not e_out, its count increases by 1.
        # It must not be 3 before (so it becomes 4).
        # If a face contains both, count is unchanged.
        # If a face contains e_out but not e_in, count decreases.
        
        # To check efficiently:
        # Collect all faces for e_in
        faces_in = edge_to_faces[e_in]
        faces_out = set(edge_to_faces[e_out])
        
        for f in faces_in:
            if f not in faces_out:
                if face_counts[f] == 3:
                    return False
        
        # Perform swap
        selected[e_out] = False
        selected[e_in] = True
        
        # Update counts
        for f in edge_to_faces[e_out]:
            face_counts[f] -= 1
        for f in edge_to_faces[e_in]:
            face_counts[f] += 1
            
        return True
    
    best_score = 0
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Multiple restarts
    while time.time() < deadline:
        selected, face_counts = greedy_init()
        current_score = sum(selected)
        
        if current_score > best_score:
            best_score = current_score
            best_selected = selected[:]
        
        # Local search
        no_improvement_count = 0
        max_iters = 5000
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to add a random missing edge
            # Find missing edges
            # Optimization: Keep a list of missing edges? 
            # Or just pick random e and check if not selected.
            
            added = False
            for _ in range(10): # Try a few random adds
                e = random.randint(0, total_edges - 1)
                if not selected[e]:
                    if try_add(selected, face_counts, e):
                        added = True
                        current_score += 1
                        if current_score > best_score:
                            best_score = current_score
                            best_selected = selected[:]
                        break
            
            if added:
                no_improvement_count = 0
                continue
            
            # Try a swap
            # Pick random e_out from selected, e_in from missing
            # To pick random selected, we can maintain a list, but it's expensive to update.
            # Just pick random e until we find selected/missing.
            
            swapped = False
            for _ in range(20):
                e_out = random.randint(0, total_edges - 1)
                if not selected[e_out]:
                    continue
                
                e_in = random.randint(0, total_edges - 1)
                if selected[e_in]:
                    continue
                
                if try_swap(selected, face_counts, e_out, e_in):
                    swapped = True
                    # Score remains same
                    break
            
            if not swapped:
                no_improvement_count += 1
                if no_improvement_count > 100:
                    break
            else:
                no_improvement_count = 0
        
    return [[edges[e][0], edges[e][1]] for e in range(total_edges) if best_selected[e]]
