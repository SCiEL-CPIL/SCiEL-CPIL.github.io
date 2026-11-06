import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (evaluator kills at 150s, but task says 120s)
    
    d = 7
    n = 1 << d
    num_edges = d * (1 << (d - 1))  # 448
    
    # 1. Precompute Edge Indices and Face Constraints
    # Edge index e corresponds to (u, i) where u < u^(1<<i)
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = idx
                idx_to_edge.append((u, v))
                idx += 1
                
    # Faces: 2D subcubes.
    # There are C(7,2) * 2^(7-2) = 21 * 32 = 672 faces.
    # A face is a set of 4 edges.
    seen_faces = set()
    face_data = [] # List of lists of 4 edge indices
    
    for i in range(d):
        for j in range(i+1, d):
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
                    face_data.append(idxs)
                    
    num_faces = len(face_data)
    
    # Precompute for each edge, the list of faces it belongs to.
    # edge_faces[e] = list of face indices
    edge_faces = [[] for _ in range(num_edges)]
    for f_idx, face_edges in enumerate(face_data):
        for e in face_edges:
            edge_faces[e].append(f_idx)
            
    # Precompute for each face, the other 3 edges for each edge in the face?
    # Or just use face_data directly.
    
    # 2. Strategy: Deterministic Weighted Greedy + Local Search
    
    # We will try multiple random weightings of edges to bias the greedy construction.
    # Then, we perform a local search to fix any violations (though greedy ensures none) 
    # and to add more edges.
    
    # Actually, the greedy construction ensures no C4. 
    # The local search will try to add edges that were skipped.
    
    # To speed up local search, we maintain:
    # state: boolean array of edge presence
    # face_count: array of length num_faces, count of edges in state for that face.
    # An edge e can be added if for all faces f containing e, face_count[f] < 3.
    
    best_edges = None
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    # Number of restarts based on time
    # Each restart: Greedy (fast) + Local Search (fast)
    
    num_restarts = 50 # We will adjust based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Random weights for edges to break symmetry and explore different structures
        weights = rng.random(num_edges)
        
        # Greedy Construction
        state = np.zeros(num_edges, dtype=np.uint8)
        face_count = np.zeros(num_faces, dtype=np.uint8)
        
        # Sort edges by weight (descending)
        order = np.argsort(-weights)
        
        count = 0
        for e in order:
            # Check if adding e creates a C4
            # e is in faces: edge_faces[e]
            # If any face has count 3, we cannot add e.
            can_add = True
            for f in edge_faces[e]:
                if face_count[f] == 3:
                    can_add = False
                    break
            
            if can_add:
                state[e] = 1
                count += 1
                for f in edge_faces[e]:
                    face_count[f] += 1
        
        current_state = state
        current_count = count
        best_state_run = current_state.copy()
        best_count_run = current_count
        
        # Local Search: Hill Climbing
        # We try to add edges that are currently 0.
        # We prioritize edges that are "almost" valid (i.e., blocking them is few faces with count 3).
        # Or just random attempts.
        
        # To make it fast, we can precompute the "blockers" for each edge?
        # No, face_count changes.
        
        # Simple Hill Climbing:
        # Iterate through all edges not in state. If valid, add it.
        # Repeat until no improvement.
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Shuffle order of candidate edges to avoid bias
            candidates = np.where(current_state == 0)[0]
            rng.shuffle(candidates)
            
            for e in candidates:
                # Check validity
                can_add = True
                for f in edge_faces[e]:
                    if face_count[f] == 3:
                        can_add = False
                        break
                
                if can_add:
                    current_state[e] = 1
                    current_count += 1
                    for f in edge_faces[e]:
                        face_count[f] += 1
                    improved = True
                    if current_count > best_count_run:
                        best_count_run = current_count
                        best_state_run = current_state.copy()
                    
                    # Early stop if we hit theoretical max? No, max is unknown.
                    # But if we add many, maybe break inner loop to re-shuffle?
                    # No, just continue.
        
        if best_count_run > best_count:
            best_count = best_count_run
            best_edges = best_state_run.copy()
            
            # If we hit a high count, we can be more aggressive with time
            if best_count >= 304:
                # We are beating the known best. Try to push higher.
                pass
        
        # Optional: If we have time, do a more sophisticated move (remove 1, add 2)?
        # Given the density, simple addition is likely the main gain.
        # The greedy already adds as many as it can in that order.
        # The local search adds the rest.
        
        # To explore more, we can also try removing some edges and re-adding?
        # That's complex. Let's stick to addition-only hill climbing after greedy.
        
    # Convert best_edges to list of lists
    if best_edges is None:
        return []
        
    result = []
    for i in range(num_edges):
        if best_edges[i] == 1:
            result.append(list(idx_to_edge[i]))
            
    return result
