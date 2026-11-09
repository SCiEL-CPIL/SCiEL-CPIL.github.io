import time
import random
import numpy as np

def construct():
    """
    Construct a C4-free edge set of Q7 maximizing the number of edges.
    Uses a structured approach based on partitioning the hypercube into
    2D faces and selecting edges to avoid completing any face.
    
    Strategy:
    1. Represent Q7 edges as (u, v, dim) where v = u ^ (1 << dim) and u < v.
    2. Use a greedy + local search approach with a deadline guard.
    3. Start with a good initial solution using a deterministic heuristic,
       then improve via simulated annealing or hill climbing.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    d = 7
    n = 1 << d  # 128 vertices
    
    # Precompute all edges: (u, v, dim) with u < v
    all_edges = []
    for u in range(n):
        for dim in range(d):
            v = u ^ (1 << dim)
            if u < v:
                all_edges.append((u, v, dim))
    
    total_edges = len(all_edges)  # 448
    
    # Precompute for each edge, the other edges that would form a C4 with it.
    # For edge (u, v, dim), a C4 is formed with edge (u', v', dim2) where
    # dim2 != dim and u' = u ^ (1<<dim2), v' = v ^ (1<<dim2), and the
    # remaining two edges (u, u') and (v, v') are also present.
    # 
    # More precisely: edge e1=(u,v,dim1) and e2=(a,b,dim2) form part of a C4
    # if dim1 != dim2 and {a,b} = {u^(1<<dim2), v^(1<<dim2)} (as a set)
    # and the cross edges exist.
    
    # Alternative approach: for each 2D face (identified by two dimensions dim1, dim2
    # and a base vertex with those bits cleared), the four edges are determined.
    
    # Let's use a different representation. For each pair of dimensions (i, j) with i < j,
    # and for each vertex x with bits i and j cleared, we have a 2D face with vertices:
    #   x, x^(1<<i), x^(1<<j), x^(1<<i)^(1<<j)
    # The four edges of this face are:
    #   (x, x^(1<<i)) in dim i
    #   (x, x^(1<<j)) in dim j
    #   (x^(1<<i), x^(1<<i)^(1<<j)) in dim j
    #   (x^(1<<j), x^(1<<i)^(1<<j)) in dim i
    
    # For each face, we need to ensure not all 4 edges are selected.
    
    # Let's precompute for each edge, the list of faces it belongs to.
    # And for each face, the list of edges (indices into all_edges).
    
    # Map from edge (u,v) to index
    edge_to_idx = {}
    for idx, (u, v, dim) in enumerate(all_edges):
        edge_to_idx[(min(u,v), max(u,v))] = idx
    
    # For each face, store the 4 edge indices
    faces = []  # list of (face_id, [e1, e2, e3, e4])
    face_edges = {}  # edge_idx -> list of face indices
    
    for i in range(d):
        for j in range(i+1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            # Iterate over all vertices with bits i and j cleared
            for x in range(n):
                if x & mask_i or x & mask_j:
                    continue
                v0 = x
                v1 = x ^ mask_i
                v2 = x ^ mask_j
                v3 = x ^ mask_i ^ mask_j
                
                e01 = (min(v0,v1), max(v0,v1))
                e02 = (min(v0,v2), max(v0,v2))
                e13 = (min(v1,v3), max(v1,v3))
                e23 = (min(v2,v3), max(v2,v3))
                
                idx01 = edge_to_idx[e01]
                idx02 = edge_to_idx[e02]
                idx13 = edge_to_idx[e13]
                idx23 = edge_to_idx[e23]
                
                face_idx = len(faces)
                faces.append([idx01, idx02, idx13, idx23])
                
                for ei in [idx01, idx02, idx13, idx23]:
                    if ei not in face_edges:
                        face_edges[ei] = []
                    face_edges[ei].append(face_idx)
    
    num_faces = len(faces)
    
    # State: boolean array of length total_edges, True = edge selected
    # Objective: maximize sum of selected edges
    # Constraint: for each face, not all 4 edges are selected
    
    # Initial solution: greedy
    def greedy_initial():
        selected = [False] * total_edges
        # Process edges in random order for diversity
        order = list(range(total_edges))
        random.shuffle(order)
        
        for idx in order:
            u, v, dim = all_edges[idx]
            # Check if adding this edge creates a C4
            # For each face containing this edge, check if the other 3 edges are selected
            can_add = True
            for fidx in face_edges.get(idx, []):
                face = faces[fidx]
                # Check if the other 3 edges in this face are selected
                others = [e for e in face if e != idx]
                if all(selected[e] for e in others):
                    can_add = False
                    break
            if can_add:
                selected[idx] = True
        
        return selected
    
    # Try multiple random greedy starts and pick the best
    best_selected = None
    best_count = 0
    
    start_time = time.time()
    num_greedy = 0
    while time.time() - start_time < 10:
        sel = greedy_initial()
        count = sum(sel)
        if count > best_count:
            best_count = count
            best_selected = sel
        num_greedy += 1
        if time.time() > deadline:
            break
    
    if best_selected is None:
        best_selected = greedy_initial()
        best_count = sum(best_selected)
    
    # Local search: try flipping edges to improve
    # For each unselected edge, try adding it if it doesn't violate constraints
    # For each selected edge, try removing it if it allows adding more edges elsewhere
    
    def count_faces_violated(selected):
        """Count number of faces with all 4 edges selected."""
        violations = 0
        for face in faces:
            if all(selected[e] for e in face):
                violations += 1
        return violations
    
    # Verify initial solution has no violations
    assert count_faces_violated(best_selected) == 0, "Initial solution has C4 violations"
    
    # Hill climbing with random restarts
    current = best_selected[:]
    current_count = best_count
    
    while time.time() < deadline - 5:
        # Try to add edges
        improved = False
        # Random order for trying to add
        unselected = [i for i in range(total_edges) if not current[i]]
        random.shuffle(unselected)
        
        for idx in unselected:
            if time.time() > deadline - 5:
                break
            # Check if we can add this edge
            can_add = True
            for fidx in face_edges.get(idx, []):
                face = faces[fidx]
                others = [e for e in face if e != idx]
                if all(current[e] for e in others):
                    can_add = False
                    break
            if can_add:
                current[idx] = True
                current_count += 1
                improved = True
        
        if current_count > best_count:
            best_count = current_count
            best_selected = current[:]
        
        # Try to remove some edges and re-add different ones
        # This is more complex; for now, just try adding
        
        if not improved:
            # Try a more aggressive move: remove a random selected edge, then try to add
            selected_idx = [i for i in range(total_edges) if current[i]]
            if selected_idx:
                remove_idx = random.choice(selected_idx)
                current[remove_idx] = False
                current_count -= 1
                
                # Now try to add edges
                unselected2 = [i for i in range(total_edges) if not current[i]]
                random.shuffle(unselected2)
                for idx in unselected2:
                    if time.time() > deadline - 5:
                        break
                    can_add = True
                    for fidx in face_edges.get(idx, []):
                        face = faces[fidx]
                        others = [e for e in face if e != idx]
                        if all(current[e] for e in others):
                            can_add = False
                            break
                    if can_add:
                        current[idx] = True
                        current_count += 1
                
                if current_count > best_count:
                    best_count = current_count
                    best_selected = current[:]
                else:
                    # Revert
                    current = best_selected[:]
                    current_count = best_count
    
    # Build output
    result = []
    for idx, sel in enumerate(best_selected):
        if sel:
            u, v, dim = all_edges[idx]
            result.append([u, v])
    
    return result
