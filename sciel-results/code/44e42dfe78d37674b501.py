import time
import random
import numpy as np

def construct():
    """
    Finds a large C4-free edge set in Q7.
    Returns a list of edges [u, v].
    """
    d = 7
    n_vertices = 2 ** d
    total_edges = d * (2 ** (d - 1)) # 448
    
    # 1. Precompute edges
    edges = []
    edge_index = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_index[(u, v)] = idx
    
    # 2. Precompute faces
    # A face is defined by a base vertex 'base' and two distinct directions i, j.
    # To avoid duplicates, we can define the base such that bits i and j are 0 in 'base'.
    # And we fix i < j.
    faces = []
    face_edges = [] # list of 4 edge indices for each face
    
    # Iterate over all pairs of directions i, j
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all base vertices where bit i and bit j are 0
            # We can iterate 0 to 2^d - 1 and check, or construct them.
            # There are 2^(d-2) such bases.
            mask_i = 1 << i
            mask_j = 1 << j
            for base in range(n_vertices):
                if (base & mask_i) == 0 and (base & mask_j) == 0:
                    # The 4 vertices of the face are:
                    # v0 = base
                    # v1 = base ^ mask_i
                    # v2 = base ^ mask_j
                    # v3 = base ^ mask_i ^ mask_j
                    
                    # The 4 edges are:
                    # e1: v0 - v1 (dir i)
                    # e2: v2 - v3 (dir i)
                    # e3: v0 - v2 (dir j)
                    # e4: v1 - v3 (dir j)
                    
                    e1 = (min(v0 := base, v1 := base ^ mask_i), max(v0, v1))
                    e2 = (min(v2 := base ^ mask_j, v3 := base ^ mask_i ^ mask_j), max(v2, v3))
                    e3 = (min(v0, v2), max(v0, v2))
                    e4 = (min(v1, v3), max(v1, v3))
                    
                    idxs = [
                        edge_index[e1],
                        edge_index[e2],
                        edge_index[e3],
                        edge_index[e4]
                    ]
                    faces.append(idxs)
                    face_edges.append(idxs)
    
    num_faces = len(faces) # 672
    
    # 3. Precompute edge-to-face mapping
    edge_faces = [[] for _ in range(total_edges)]
    for f_idx, f_edges in enumerate(face_edges):
        for e_idx in f_edges:
            edge_faces[e_idx].append(f_idx)
            
    # 4. Search Function
    def local_search(initial_selection, time_limit, rng):
        # selection: boolean array of length total_edges
        selection = initial_selection.copy()
        # face_count: number of selected edges in each face
        face_count = np.zeros(num_faces, dtype=np.int8)
        
        # Initialize face_count
        for e_idx in range(total_edges):
            if selection[e_idx]:
                for f_idx in edge_faces[e_idx]:
                    face_count[f_idx] += 1
        
        current_size = np.sum(selection)
        current_violations = np.sum(face_count == 4)
        lambda_penalty = 50.0
        current_score = current_size - lambda_penalty * current_violations
        
        best_selection = selection.copy()
        best_size = current_size
        best_violations = current_violations
        
        T = 1.0
        cooling = 0.995
        steps_per_iter = 1000
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            for _ in range(steps_per_iter):
                # Pick a random edge to flip
                e_idx = rng.randint(0, total_edges - 1)
                
                # Calculate delta
                if selection[e_idx]:
                    # Removing
                    delta_size = -1
                    # Check faces containing e_idx
                    delta_viol = 0
                    for f_idx in edge_faces[e_idx]:
                        if face_count[f_idx] == 4:
                            # Removing one edge from a full face reduces violations by 1
                            delta_viol -= 1
                        # If face_count was 3, it becomes 2, no violation change
                        # If face_count was >4 (impossible), etc.
                    # Note: Removing an edge from a face with count 3 does not change violation status.
                    # Removing from count 4 changes 4->3, so violation count decreases by 1.
                    # Removing from count <4 does nothing to violation count.
                    
                    new_score = (current_size - 1) - lambda_penalty * (current_violations + delta_viol)
                    delta_score = new_score - current_score
                else:
                    # Adding
                    delta_size = 1
                    delta_viol = 0
                    for f_idx in edge_faces[e_idx]:
                        if face_count[f_idx] == 3:
                            # Adding one edge to a face with 3 selected makes it 4 (violation)
                            delta_viol += 1
                        # If face_count was 4, it becomes 5? No, max is 4.
                        # Wait, if face_count is 4, it's already a violation.
                        # If we add an edge to a face that already has 4?
                        # That edge must be the 5th? No, faces have exactly 4 edges.
                        # So if face_count[f_idx] == 4, the edge e_idx is NOT in that face?
                        # No, e_idx IS in that face.
                        # If face_count[f_idx] == 4, all 4 edges are selected.
                        # So selection[e_idx] must be True.
                        # So we are in the "Removing" branch if selection[e_idx] is True.
                        # If selection[e_idx] is False, face_count[f_idx] can be at most 3.
                        # So we only check for 3 -> 4.
                    
                    new_score = (current_size + 1) - lambda_penalty * (current_violations + delta_viol)
                    delta_score = new_score - current_score
                
                # Acceptance
                accept = False
                if delta_score >= 0:
                    accept = True
                else:
                    if T > 0:
                        prob = np.exp(delta_score / T)
                        if rng.random() < prob:
                            accept = True
                
                if accept:
                    # Apply move
                    selection[e_idx] = not selection[e_idx]
                    current_size += delta_size
                    current_violations += delta_viol
                    current_score = new_score
                    
                    # Update face_count
                    for f_idx in edge_faces[e_idx]:
                        if selection[e_idx]:
                            face_count[f_idx] += 1
                        else:
                            face_count[f_idx] -= 1
                    
                    # Track best valid solution
                    if current_violations == 0 and current_size > best_size:
                        best_size = current_size
                        best_violations = 0
                        best_selection = selection.copy()
                    elif current_violations < best_violations and current_size >= best_size:
                        # Prefer fewer violations, then more size
                        best_size = current_size
                        best_violations = current_violations
                        best_selection = selection.copy()
                    
                    # Check time
                    if time.time() - start_time > time_limit:
                        return best_selection, best_size
                
            T *= cooling
            if T < 0.001:
                break
                
        return best_selection, best_size

    def repair(selection, face_count):
        # Remove edges to resolve violations
        # face_count is passed to avoid recalculation, but we need to update it.
        # Let's just recalculate for safety or pass it.
        # We'll assume selection is a numpy array.
        # Recalculate face_count for safety
        fc = np.zeros(num_faces, dtype=np.int8)
        for e_idx in range(total_edges):
            if selection[e_idx]:
                for f_idx in edge_faces[e_idx]:
                    fc[f_idx] += 1
        
        # While there are violations
        changed = True
        while changed:
            changed = False
            for f_idx in range(num_faces):
                if fc[f_idx] == 4:
                    # Pick a random edge from this face to remove
                    # Prefer edges that are in the most other violating faces?
                    # Simple: random
                    e_idx = face_edges[f_idx][random.randint(0, 3)]
                    if selection[e_idx]:
                        selection[e_idx] = False
                        for ff_idx in edge_faces[e_idx]:
                            fc[ff_idx] -= 1
                        changed = True
                        # Break to restart loop? Or continue?
                        # Continue is fine, might fix multiple.
        return selection, fc

    def greedy_fill(selection, fc):
        # Try to add any edge that doesn't create a violation
        improved = True
        while improved:
            improved = False
            for e_idx in range(total_edges):
                if not selection[e_idx]:
                    # Check if adding e_idx creates any violation
                    ok = True
                    for f_idx in edge_faces[e_idx]:
                        if fc[f_idx] == 3:
                            ok = False
                            break
                    if ok:
                        selection[e_idx] = True
                        for f_idx in edge_faces[e_idx]:
                            fc[f_idx] += 1
                        improved = True
        return selection, fc

    # Main Loop
    start_time = time.time()
    time_limit = 110.0 # Leave 10s for overhead
    
    best_final_selection = None
    best_final_size = 0
    
    rng = random.Random(42)
    
    num_restarts = 20
    restart_time = time_limit / num_restarts
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize randomly
        init_density = 0.5
        init_sel = np.array([rng.random() < init_density for _ in range(total_edges)], dtype=bool)
        
        # Run local search
        sel, size = local_search(init_sel, restart_time, rng)
        
        # Repair
        sel, fc = repair(sel, None)
        
        # Greedy Fill
        sel, fc = greedy_fill(sel, fc)
        
        final_size = np.sum(sel)
        
        if final_size > best_final_size:
            best_final_size = final_size
            best_final_selection = sel.copy()
            
    # Convert to list of edges
    if best_final_selection is None:
        # Fallback to simple greedy if something failed
        # ... implement simple greedy as fallback
        chosen = set()
        out = []
        for u in range(n_vertices):
            for i in range(d):
                v = u ^ (1 << i)
                if u < v:
                    ok = True
                    for j in range(d):
                        if j != i:
                            c, e = u ^ (1 << j), v ^ (1 << j)
                            if all(p in chosen for p in ((min(u, c), max(u, c)), (min(v, e), max(v, e)), (min(c, e), max(c, e)))):
                                ok = False
                                break
                    if ok:
                        chosen.add((u, v))
                        out.append([u, v])
        return out

    out = []
    for e_idx, is_sel in enumerate(best_final_selection):
        if is_sel:
            out.append(list(edges[e_idx]))
            
    return out
