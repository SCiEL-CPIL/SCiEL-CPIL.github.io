import random
import time
import math

def construct():
    d = 7
    n_vertices = 2 ** d
    total_edges = d * (2 ** (d - 1))

    # Generate all edges
    edges = []
    edge_index = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
                edge_index[(x, y)] = len(edges) - 1
                edge_index[(y, x)] = len(edges) - 1

    # Precompute faces (C4s)
    # A face is defined by a vertex and two distinct bits i, j
    # Edges in face: (v, v^i), (v^i, v^i^j), (v^j, v^i^j), (v, v^j)
    
    edge_to_faces = [[] for _ in range(total_edges)]
    face_edges = [] # List of 4 edge indices for each face
    
    seen_faces = set()
    
    for v in range(n_vertices):
        for i in range(d):
            for j in range(i+1, d):
                verts = [v, v ^ (1<<i), v ^ (1<<j), v ^ (1<<i ^ (1<<j))]
                verts.sort()
                key = tuple(verts)
                
                if key in seen_faces:
                    continue
                seen_faces.add(key)
                
                e1 = edge_index[(min(v, v^(1<<i)), max(v, v^(1<<i)))]
                e2 = edge_index[(min(v^(1<<i), v^(1<<i^(1<<j))), max(v^(1<<i), v^(1<<i^(1<<j))))]
                e3 = edge_index[(min(v^(1<<j), v^(1<<i^(1<<j))), max(v^(1<<j), v^(1<<i^(1<<j))))]
                e4 = edge_index[(min(v, v^(1<<j)), max(v, v^(1<<j)))]
                
                face_idx = len(face_edges)
                face_edges.append([e1, e2, e3, e4])
                
                for e in [e1, e2, e3, e4]:
                    edge_to_faces[e].append(face_idx)

    num_faces = len(face_edges)
    
    start_time = time.time()
    time_limit = 115
    rng = random.Random(42)
    
    best_edges = None
    best_count = 0
    
    # ILS Loop
    for iteration in range(100):
        if time.time() - start_time > time_limit:
            break
            
        # 1. Initialization: Greedy Hitting Set to remove edges
        # Start with all edges selected
        selected = [True] * total_edges
        # Count how many active faces each edge is in
        # Initially, all faces are active.
        face_active = [True] * num_faces
        edge_face_count = [0] * total_edges
        for e in range(total_edges):
            edge_face_count[e] = len(edge_to_faces[e])
            
        # Greedy removal
        # We maintain a list of active faces.
        # At each step, find the edge that is in the most active faces.
        # Remove it. Update counts for faces it was in.
        
        # Optimization: Use a max-heap or just scan?
        # Scanning 448 edges * 672 faces is fast enough.
        # But we need to know which faces are active.
        
        active_faces_set = set(range(num_faces))
        
        while active_faces_set:
            # Find edge with max active face count
            max_count = -1
            best_edge = -1
            for e in range(total_edges):
                if not selected[e]:
                    continue
                # Count active faces for this edge
                count = 0
                for f_idx in edge_to_faces[e]:
                    if f_idx in active_faces_set:
                        count += 1
                if count > max_count:
                    max_count = count
                    best_edge = e
                    
            if best_edge == -1:
                break
                
            # Remove best_edge
            selected[best_edge] = False
            
            # Update active_faces_set
            for f_idx in edge_to_faces[best_edge]:
                if f_idx in active_faces_set:
                    # Check if face is still active (all 4 edges selected?)
                    # Since we just removed best_edge, this face is no longer active.
                    active_faces_set.remove(f_idx)
                    
            # Note: We don't strictly need to update edge_face_count for the next step 
            # if we recalculate, but it's expensive. 
            # However, removing an edge only deactivates faces containing it.
            # Other edges' active face counts decrease by 1 for each deactivated face they share.
            # This is complex to maintain dynamically. 
            # Given small size, recalculating max_count by scanning is O(E * F) per removal.
            # 448 * 672 ~ 300k ops. 448 removals * 300k = 130M ops. Might be too slow in Python.
            
            # Let's optimize: 
            # Maintain edge_active_face_count[e].
            # When we remove edge e, for each face f containing e:
            #   if f is active:
            #     deactivate f
            #     for each other edge e2 in f:
            #       edge_active_face_count[e2] -= 1
            
        # Recalculate edge_active_face_count for the next iteration if we restart?
        # No, this is one initialization.
        
        # Let's re-implement the greedy removal with dynamic counts for speed.
        selected = [True] * total_edges
        edge_active_face_count = [0] * total_edges
        for e in range(total_edges):
            edge_active_face_count[e] = len(edge_to_faces[e])
            
        active_faces = [True] * num_faces
        
        # Use a list of active faces to iterate?
        # Or just loop until no active faces.
        
        # To find max edge, we can scan.
        # To update, we need to know which faces became inactive.
        
        active_face_indices = list(range(num_faces))
        
        while active_face_indices:
            # Find edge with max count
            max_count = -1
            best_edge = -1
            for e in range(total_edges):
                if selected[e] and edge_active_face_count[e] > max_count:
                    max_count = edge_active_face_count[e]
                    best_edge = e
                    
            if best_edge == -1 or max_count == 0:
                break
                
            selected[best_edge] = False
            
            # Update counts for faces containing best_edge
            for f_idx in edge_to_faces[best_edge]:
                if active_faces[f_idx]:
                    # This face becomes inactive
                    active_faces[f_idx] = False
                    # Remove from active_face_indices? 
                    # It's easier to just check active_faces[f_idx] in the scan.
                    # But active_face_indices is used to know if we're done.
                    # Let's just use a counter for active faces.
                    
                    # Update other edges in this face
                    for e2 in face_edges[f_idx]:
                        if e2 != best_edge:
                            edge_active_face_count[e2] -= 1
                            
            # Remove inactive faces from active_face_indices? 
            # It's O(F) to filter. 
            # Let's just count active faces.
            
        # Count active faces to check termination
        # Actually, the loop condition `while active_face_indices` is wrong if we don't remove from it.
        # Let's use a counter.
        
        # RESTART CLEAN IMPLEMENTATION OF GREEDY HITTING SET
        
        selected = [True] * total_edges
        edge_active_face_count = [0] * total_edges
        for e in range(total_edges):
            edge_active_face_count[e] = len(edge_to_faces[e])
            
        active_faces_count = num_faces
        active_faces = [True] * num_faces
        
        while active_faces_count > 0:
            max_count = -1
            best_edge = -1
            for e in range(total_edges):
                if selected[e] and edge_active_face_count[e] > max_count:
                    max_count = edge_active_face_count[e]
                    best_edge = e
                    
            if best_edge == -1 or max_count == 0:
                break
                
            selected[best_edge] = False
            
            for f_idx in edge_to_faces[best_edge]:
                if active_faces[f_idx]:
                    active_faces[f_idx] = False
                    active_faces_count -= 1
                    
                    for e2 in face_edges[f_idx]:
                        if e2 != best_edge:
                            edge_active_face_count[e2] -= 1
                            
        # Now we have a feasible solution.
        # Local Search: Try to add edges back?
        # If we removed edges, we can't just add them back if they create C4s.
        # But maybe we removed "too many".
        # Try to swap: Remove 1, Add 2?
        # Or just try to add any edge that doesn't create a C4.
        # Since we started with all edges and removed until feasible, 
        # any edge we removed created a C4 at the time of removal.
        # However, subsequent removals might have broken that C4.
        # So we can try to re-add edges.
        
        # Greedy Addition Phase
        # Iterate through all edges not selected.
        # If adding it doesn't create a C4, add it.
        
        # To check if adding edge e creates a C4:
        # It creates a C4 if there is a face f containing e where the other 3 edges are selected.
        
        # We can maintain a count of "selected other edges" for each face?
        # Or just check on the fly.
        
        # Let's do a simple pass.
        for e in range(total_edges):
            if not selected[e]:
                # Check if we can add e
                can_add = True
                for f_idx in edge_to_faces[e]:
                    # Check if other 3 are selected
                    count_sel = 0
                    for e2 in face_edges[f_idx]:
                        if e2 != e and selected[e2]:
                            count_sel += 1
                    if count_sel == 3:
                        can_add = False
                        break
                if can_add:
                    selected[e] = True
                    
        # Local Search: Swaps
        # Try to remove one edge and add two?
        # This is complex.
        # Let's try simple swaps: Remove e1, Add e2.
        # If e2 was blocked by e1, removing e1 might unblock e2.
        
        # Iterate through unselected edges e2.
        # For each e2, find a face f where e2 is blocked (3 other edges selected).
        # Pick one of those 3 edges, say e1.
        # Remove e1.
        # Check if e2 can be added.
        # If yes, and if removing e1 doesn't break anything (it won't, we're removing), 
        # and if the net change is 0 (remove 1, add 1), we are not improving size.
        # We want to increase size.
        # So we need to remove 1 and add 2? Or remove 0 and add 1 (impossible if feasible).
        
        # Actually, if we just do Greedy Addition, we might be stuck.
        # Let's try a few random perturbations and re-run Greedy Addition.
        
        current_count = sum(selected)
        if current_count > best_count:
            best_count = current_count
            best_edges = selected[:]
            
        # Perturb: Remove a few random edges
        if iteration > 0:
            for _ in range(5):
                idx = rng.randint(0, total_edges - 1)
                if selected[idx]:
                    selected[idx] = False
                    
            # Re-run Greedy Addition
            for e in range(total_edges):
                if not selected[e]:
                    can_add = True
                    for f_idx in edge_to_faces[e]:
                        count_sel = 0
                        for e2 in face_edges[f_idx]:
                            if e2 != e and selected[e2]:
                                count_sel += 1
                        if count_sel == 3:
                            can_add = False
                            break
                    if can_add:
                        selected[e] = True
                        
            current_count = sum(selected)
            if current_count > best_count:
                best_count = current_count
                best_edges = selected[:]

    # Convert best_edges to list of edges
    if best_edges is None:
        # Fallback
        best_edges = [True] * total_edges
        # Run greedy removal one more time to ensure validity
        # ... (same logic as above)
        
    result = []
    for i, is_sel in enumerate(best_edges):
        if is_sel:
            result.append(list(edges[i]))
            
    return result
