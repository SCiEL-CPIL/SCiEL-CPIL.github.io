import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    d = 7
    n = 1 << d
    num_edges = d * (1 << (d - 1))  # 448
    
    # 1. Precompute edges
    idx_to_edge = []
    edge_to_idx = {}
    idx = 0
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = idx
                idx_to_edge.append((u, v))
                idx += 1
                
    # 2. Precompute faces
    # A face is a set of 4 edges forming a C4.
    # We identify unique faces by their set of edge indices.
    seen_faces = set()
    face_edge_list = []
    
    for i in range(d):
        for j in range(i + 1, d):
            for u in range(n):
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v4), max(v2, v4))
                e4 = (min(v3, v4), max(v3, v4))
                
                idxs = [edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]]
                key = frozenset(idxs)
                
                if key not in seen_faces:
                    seen_faces.add(key)
                    face_edge_list.append(idxs)
                    
    num_faces = len(face_edge_list)
    
    # Precompute edge -> faces mapping
    edge_to_faces = [[] for _ in range(num_edges)]
    for f_idx, face in enumerate(face_edge_list):
        for e in face:
            edge_to_faces[e].append(f_idx)
            
    # 3. Build Conflict Graph (Line Graph of Hypercube)
    # Vertices: 0..num_edges-1
    # Edge (u, v) in conflict graph if u and v share a face.
    # Since each edge is in 6 faces, and each face has 4 edges, each edge conflicts with 6*3=18 others?
    # Let's count: In a face of 4 edges, each edge is adjacent to 2 others in the C4?
    # No, in the conflict graph (line graph), two edges are connected if they share a vertex in the original graph?
    # Wait, the constraint is "No 2-dimensional face may have all four edges selected".
    # This is NOT the standard Line Graph MIS (which forbids adjacent edges, i.e., sharing a vertex).
    # This forbids selecting a set of edges that forms a C4.
    # So the "conflict" is: a set of 4 edges is forbidden if they form a face.
    # This is a 4-uniform hypergraph constraint, not a 2-uniform graph constraint.
    # However, we can still use spectral methods on a derived graph.
    
    # Let's define a "conflict" for spectral init as: two edges conflict if they are in a common face.
    # This creates a graph G_conflict.
    # If we pick an independent set in G_conflict, we guarantee no two edges share a face, which is too strong.
    # We want to pick edges such that no 4 edges form a face.
    
    # Strategy: Spectral Initialization
    # Build adjacency matrix of G_conflict (edges share a face).
    # Compute Fiedler vector (eigenvector of 2nd smallest eigenvalue of Laplacian).
    # Threshold the Fiedler vector to get an initial partition.
    # Then, greedily add edges to maximize count while respecting the C4 constraint.
    
    # Build G_conflict adjacency (sparse or dense? 448x448 is small, use dense)
    A = np.zeros((num_edges, num_edges), dtype=np.int8)
    for face in face_edge_list:
        for i in range(4):
            for j in range(i + 1, 4):
                A[face[i], face[j]] = 1
                A[face[j], face[i]] = 1
                
    # Laplacian L = D - A
    deg = np.sum(A, axis=1)
    L = np.diag(deg) - A
    
    # Compute eigenvectors
    # We want the Fiedler vector (index 1, 0-based, after sorting by eigenvalue)
    try:
        evals, evecs = np.linalg.eigh(L)
        # Sort by eigenvalue
        idxs = np.argsort(evals)
        evecs = evecs[:, idxs]
        fiedler = evecs[:, 1]
    except:
        # Fallback to random
        fiedler = np.random.randn(num_edges)
        
    # Threshold Fiedler vector
    # Mean is usually 0.
    threshold = np.mean(fiedler)
    init_selected = (fiedler > threshold).astype(np.uint8)
    
    # 4. Greedy Extension and Local Search
    # We start with init_selected, but it might violate constraints (have C4s).
    # We need to "repair" it by removing edges from any face that has 4 selected edges.
    
    def repair_and_maximize(selected):
        selected = selected.copy()
        face_counts = np.zeros(num_faces, dtype=np.int32)
        
        # Calculate initial counts
        for e in range(num_edges):
            if selected[e] == 1:
                for f_idx in edge_to_faces[e]:
                    face_counts[f_idx] += 1
        
        # Repair phase: Remove edges from saturated faces (count == 4)
        # We want to minimize removals.
        # Simple heuristic: iterate faces, if count==4, remove one edge.
        # Which one? The one that is in the most other saturated faces? Or just random?
        # Let's try removing the edge that is "least useful" (in fewest faces with count 3 or 4?).
        
        changed = True
        while changed:
            changed = False
            for f_idx in range(num_faces):
                if face_counts[f_idx] == 4:
                    # Find an edge to remove
                    # Heuristic: remove edge with highest "degree" in the subgraph of saturated faces?
                    # Or just pick the first one?
                    # Let's pick the edge that is in the most *other* saturated faces to break as many C4s as possible?
                    # No, if we remove an edge, we only fix this face.
                    # Actually, removing an edge from a face with count 4 reduces count to 3.
                    # It might help other faces containing that edge.
                    
                    # Let's pick the edge in this face that is in the most number of *currently saturated* faces.
                    face = face_edge_list[f_idx]
                    best_e = -1
                    max_sat_count = -1
                    for e in face:
                        if selected[e] == 1:
                            sat_count = 0
                            for f2 in edge_to_faces[e]:
                                if face_counts[f2] == 4:
                                    sat_count += 1
                            if sat_count > max_sat_count:
                                max_sat_count = sat_count
                                best_e = e
                                
                    if best_e != -1:
                        # Remove best_e
                        selected[best_e] = 0
                        for f2 in edge_to_faces[best_e]:
                            face_counts[f2] -= 1
                        changed = True
                        
        # Now the set is valid (no face has 4 edges).
        # Maximize: Add edges if possible.
        improved = True
        while improved:
            improved = False
            for e in range(num_edges):
                if selected[e] == 0:
                    # Check if addable
                    can_add = True
                    for f_idx in edge_to_faces[e]:
                        if face_counts[f_idx] == 3:
                            can_add = False
                            break
                    if can_add:
                        selected[e] = 1
                        for f_idx in edge_to_faces[e]:
                            face_counts[f_idx] += 1
                        improved = True
        
        # Local Search: 1-Swap
        # Try to remove one edge and add another? Or just add if possible?
        # The "Add" loop above ensures maximality.
        # To escape local optima, we can try swaps:
        # Remove edge e1, add edge e2.
        # This is complex. Let's stick to the greedy repair + maximal extension.
        # The spectral init provides a good starting point.
        
        return selected, np.sum(selected)
    
    best_selected = None
    best_count = 0
    
    # Try the spectral init
    sel, cnt = repair_and_maximize(init_selected)
    if cnt > best_count:
        best_count = cnt
        best_selected = sel
        
    # Try a few random inits to compare
    rng = np.random.default_rng(42)
    for run in range(20):
        if time.time() > deadline - 5:
            break
            
        # Random init: select edges with prob 0.5
        init_sel = (rng.random(num_edges) < 0.5).astype(np.uint8)
        sel, cnt = repair_and_maximize(init_sel)
        if cnt > best_count:
            best_count = cnt
            best_selected = sel

    # If spectral init was bad, or if we have time, do a more aggressive local search on the best found
    # Let's implement a simple 1-move local search on best_selected
    if best_selected is not None:
        sel = best_selected.copy()
        face_counts = np.zeros(num_faces, dtype=np.int32)
        for e in range(num_edges):
            if sel[e] == 1:
                for f_idx in edge_to_faces[e]:
                    face_counts[f_idx] += 1
        
        current_count = np.sum(sel)
        
        # Try to add any missing edge that is addable (should be none if repair_and_maximize did its job, but just in case)
        for e in range(num_edges):
            if sel[e] == 0:
                can_add = True
                for f_idx in edge_to_faces[e]:
                    if face_counts[f_idx] == 3:
                        can_add = False
                        break
                if can_add:
                    sel[e] = 1
                    for f_idx in edge_to_faces[e]:
                        face_counts[f_idx] += 1
                    current_count += 1
        
        # 1-Swap Local Search:
        # For each selected edge e1, try to remove it and see if we can add a new edge e2 that was blocked by e1's face.
        # This is expensive.
        # Alternative: Random restarts with different thresholds on Fiedler vector?
        
        # Let's try perturbing the Fiedler vector
        for pert_run in range(10):
            if time.time() > deadline - 2:
                break
            noise = rng.normal(0, 0.1, num_edges)
            perturbed = fiedler + noise
            threshold = np.mean(perturbed)
            init_sel = (perturbed > threshold).astype(np.uint8)
            sel, cnt = repair_and_maximize(init_sel)
            if cnt > best_count:
                best_count = cnt
                best_selected = sel

    # Convert best_selected to list of edges
    result = []
    if best_selected is not None:
        for i in range(num_edges):
            if best_selected[i] == 1:
                result.append(list(idx_to_edge[i]))
    else:
        # Fallback: simple greedy
        sel = np.zeros(num_edges, dtype=np.uint8)
        face_counts = np.zeros(num_faces, dtype=np.int32)
        for e in range(num_edges):
            can_add = True
            for f_idx in edge_to_faces[e]:
                if face_counts[f_idx] == 3:
                    can_add = False
                    break
            if can_add:
                sel[e] = 1
                for f_idx in edge_to_faces[e]:
                    face_counts[f_idx] += 1
        for i in range(num_edges):
            if sel[i] == 1:
                result.append(list(idx_to_edge[i]))
                
    return result
