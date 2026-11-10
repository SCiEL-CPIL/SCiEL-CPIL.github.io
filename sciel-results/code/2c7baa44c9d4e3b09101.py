import random
import time
import numpy as np

def construct():
    deadline = time.time() + 115.0
    d = 7
    n_vertices = 1 << d
    n_edges_total = d * (1 << (d - 1))  # 448
    
    # Generate all edges
    edges = []
    edge_index = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_index[(u, v)] = len(edges)
                edges.append((u, v))
    
    n_edges = len(edges)
    
    # Precompute face structure
    # Each face is a 2D subcube: 4 vertices, 4 edges
    faces = []
    face_of_edge = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            mask_ij = (1 << i) | (1 << j)
            for mask in range(1 << d):
                if mask & mask_ij != 0:
                    continue
                v0 = mask
                v1 = mask | (1 << i)
                v2 = mask | (1 << j)
                v3 = mask | (1 << i) | (1 << j)
                
                e1 = (min(v0, v1), max(v0, v1))
                e2 = (min(v0, v2), max(v0, v2))
                e3 = (min(v1, v3), max(v1, v3))
                e4 = (min(v2, v3), max(v2, v3))
                
                face_edges = (edge_index[e1], edge_index[e2], edge_index[e3], edge_index[e4])
                face_idx = len(faces)
                faces.append(face_edges)
                for e_idx in face_edges:
                    face_of_edge[e_idx].append(face_idx)
    
    n_faces = len(faces)
    
    # Convert face_of_edge to numpy for speed
    face_of_edge_np = np.array([face_of_edge[i] for i in range(n_edges)], dtype=np.int32)
    
    def can_add_fast(face_count, e_idx):
        """O(1) check using precomputed face indices."""
        for f_idx in face_of_edge[e_idx]:
            if face_count[f_idx] == 3:
                return False
        return True
    
    def add_edge_fast(selected, face_count, e_idx):
        selected[e_idx] = True
        for f_idx in face_of_edge[e_idx]:
            face_count[f_idx] += 1
    
    def remove_edge_fast(selected, face_count, e_idx):
        selected[e_idx] = False
        for f_idx in face_of_edge[e_idx]:
            face_count[f_idx] -= 1
    
    def try_linear_init(rng):
        """Generate an initial solution using a random linear map over F2."""
        selected = np.zeros(n_edges, dtype=bool)
        face_count = np.zeros(n_faces, dtype=np.int8)
        count = 0
        
        # For each direction i, define a random linear function L_i: F2^7 -> F2
        # Select edge (u, u^e_i) if L_i(u) == 0
        linear_vecs = []
        for i in range(d):
            vec = rng.getrandbits(d)
            # Ensure it's not trivial
            while vec == 0 or vec == (1 << i):
                vec = rng.getrandbits(d)
            linear_vecs.append(vec)
        
        # Process edges in a smart order: group by direction
        for i in range(d):
            vec = linear_vecs[i]
            for u in range(n_vertices):
                v = u ^ (1 << i)
                if u >= v:
                    continue
                
                # Compute dot product mod 2
                dot = 0
                for k in range(d):
                    if (vec >> k) & 1:
                        dot ^= (u >> k) & 1
                
                if dot == 0:
                    e_idx = edge_index[(u, v)]
                    if can_add_fast(face_count, e_idx):
                        selected[e_idx] = True
                        for f_idx in face_of_edge[e_idx]:
                            face_count[f_idx] += 1
                        count += 1
        
        return selected, face_count, count
    
    def local_search(selected, face_count, count, deadline):
        """Multi-pass local search: add edges, then try swaps."""
        improved = True
        passes = 0
        
        while improved and time.time() < deadline - 5.0:
            improved = False
            passes += 1
            
            # Pass 1: Try to add any unselected edge
            for e_idx in range(n_edges):
                if selected[e_idx]:
                    continue
                if can_add_fast(face_count, e_idx):
                    selected[e_idx] = True
                    for f_idx in face_of_edge[e_idx]:
                        face_count[f_idx] += 1
                    count += 1
                    improved = True
            
            # Pass 2: Try 1-for-1 swaps (remove one, add another)
            if time.time() > deadline - 5.0:
                break
                
            for e_remove in range(n_edges):
                if not selected[e_remove]:
                    continue
                if time.time() > deadline - 5.0:
                    break
                
                # Temporarily remove
                selected[e_remove] = False
                for f_idx in face_of_edge[e_remove]:
                    face_count[f_idx] -= 1
                count -= 1
                
                # Try to add any unselected edge
                added_idx = -1
                for e_add in range(n_edges):
                    if selected[e_add]:
                        continue
                    if can_add_fast(face_count, e_add):
                        selected[e_add] = True
                        for f_idx in face_of_edge[e_add]:
                            face_count[f_idx] += 1
                        count += 1
                        added_idx = e_add
                        break
                
                if added_idx == -1:
                    # Revert
                    selected[e_remove] = True
                    for f_idx in face_of_edge[e_remove]:
                        face_count[f_idx] += 1
                    count += 1
                else:
                    improved = True
        
        return selected, face_count, count
    
    best_selected = None
    best_count = 0
    best_face_count = None
    
    # Phase 1: Try many linear initializations
    rng = random.Random(42)
    
    while time.time() < deadline - 20.0:
        selected, face_count, count = try_linear_init(rng)
        
        if count > best_count:
            best_selected = selected.copy()
            best_count = count
            best_face_count = face_count.copy()
        
        # Quick local search on this solution
        if count > best_count - 10 and time.time() < deadline - 10.0:
            selected, face_count, count = local_search(selected, face_count, count, deadline)
            if count > best_count:
                best_selected = selected.copy()
                best_count = count
                best_face_count = face_count.copy()
    
    # Phase 2: Thorough local search on the best solution
    if best_selected is not None and time.time() < deadline - 5.0:
        best_selected, best_face_count, best_count = local_search(
            best_selected, best_face_count, best_count, deadline
        )
    
    # Convert to output format
    result = []
    for e_idx in range(n_edges):
        if best_selected[e_idx]:
            u, v = edges[e_idx]
            result.append([u, v])
    
    return result
