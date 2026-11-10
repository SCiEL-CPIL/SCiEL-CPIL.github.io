import random
import time
import numpy as np

def construct():
    deadline = time.time() + 110  # safety margin before 120s kill
    d = 7
    n = 1 << d  # 128 vertices
    best_count = 0
    best_chosen = None
    
    # All edges
    all_edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v, i))
    
    num_edges = len(all_edges)
    
    # Map edge tuple to index
    edge_tuple_to_idx = {}
    for idx, (u, v, i) in enumerate(all_edges):
        edge_tuple_to_idx[(u, v)] = idx
    
    # Generate all 448*6/2? No, number of faces is 2^(7-2) * C(7,2) = 32 * 21 = 672 faces.
    faces = []
    face_id = 0
    
    for i in range(d):
        for j in range(i + 1, d):
            for mask in range(1 << d):
                if (mask & (1 << i)) or (mask & (1 << j)):
                    continue
                
                v1 = mask
                v2 = mask ^ (1 << i)
                v3 = mask ^ (1 << j)
                v4 = mask ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v2, v4), max(v2, v4))
                e3 = (min(v4, v3), max(v4, v3))
                e4 = (min(v3, v1), max(v3, v1))
                
                idx1 = edge_tuple_to_idx[e1]
                idx2 = edge_tuple_to_idx[e2]
                idx3 = edge_tuple_to_idx[e3]
                idx4 = edge_tuple_to_idx[e4]
                
                faces.append((idx1, idx2, idx3, idx4))
                face_id += 1

    num_faces = len(faces)
    
    # Map edge index to list of face indices it belongs to
    edge_to_faces = [[] for _ in range(num_edges)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_to_faces[e_idx].append(f_idx)
    
    # Precompute for each edge, the set of other edges in its faces (for compatibility checks)
    # edge_neighbors[e_idx] = set of edge indices sharing a face with e_idx
    edge_neighbors = [set() for _ in range(num_edges)]
    for f_idx, face in enumerate(faces):
        for e1 in face:
            for e2 in face:
                if e1 != e2:
                    edge_neighbors[e1].add(e2)
    
    # Convert to numpy arrays for vectorized operations
    # faces_arr: shape (num_faces, 4)
    faces_arr = np.array(faces, dtype=np.int32)  # (672, 4)
    
    # edge_to_faces_arr: for each edge, list of face indices
    # We'll use a flat structure with offsets for vectorized access
    edge_to_faces_flat = []
    edge_to_faces_offsets = [0]
    for e in range(num_edges):
        edge_to_faces_flat.extend(edge_to_faces[e])
        edge_to_faces_offsets.append(len(edge_to_faces_flat))
    edge_to_faces_flat = np.array(edge_to_faces_flat, dtype=np.int32)
    edge_to_faces_offsets = np.array(edge_to_faces_offsets, dtype=np.int32)
    
    def try_greedy(order_seed=None):
        rng = random.Random(order_seed)
        chosen = np.zeros(num_edges, dtype=bool)
        count = 0
        
        indices = list(range(num_edges))
        rng.shuffle(indices)
        
        face_counts = np.zeros(num_faces, dtype=np.int32)
        
        for idx in indices:
            can_add = True
            for f_idx in edge_to_faces[idx]:
                if face_counts[f_idx] == 3:
                    can_add = False
                    break
            if can_add:
                chosen[idx] = True
                count += 1
                for f_idx in edge_to_faces[idx]:
                    face_counts[f_idx] += 1
        
        return count, chosen, face_counts

    def kick_and_readd(chosen, face_counts, count, kick_size, rng, deadline):
        """Remove kick_size random chosen edges, then greedily re-add all valid edges."""
        chosen_indices = np.where(chosen)[0]
        if len(chosen_indices) < kick_size:
            return count, chosen, face_counts
        
        # Select kick_size random edges to remove
        remove_idx = rng.sample(list(chosen_indices), kick_size)
        
        # Remove them
        for r_idx in remove_idx:
            chosen[r_idx] = False
            count -= 1
            for f_idx in edge_to_faces[r_idx]:
                face_counts[f_idx] -= 1
        
        # Find all currently unchosen edges that are valid
        # An edge is valid if for all faces containing it, face_count < 3
        unchosen = np.where(~chosen)[0]
        
        # For each unchosen edge, check if all its faces have count < 3
        valid_edges = []
        for e_idx in unchosen:
            valid = True
            for f_idx in edge_to_faces[e_idx]:
                if face_counts[f_idx] >= 3:
                    valid = False
                    break
            if valid:
                valid_edges.append(e_idx)
        
        # Greedily add valid edges
        rng.shuffle(valid_edges)
        for e_idx in valid_edges:
            can_add = True
            for f_idx in edge_to_faces[e_idx]:
                if face_counts[f_idx] == 3:
                    can_add = False
                    break
            if can_add:
                chosen[e_idx] = True
                count += 1
                for f_idx in edge_to_faces[e_idx]:
                    face_counts[f_idx] += 1
        
        return count, chosen, face_counts

    def local_search(chosen, face_counts, count, deadline):
        """Kick-based local search with varying kick sizes."""
        rng = random.Random(int(time.time() * 1000))
        improved = True
        stagnation = 0
        best_local = count
        best_local_chosen = chosen.copy()
        best_local_face_counts = face_counts.copy()
        
        while improved and time.time() < deadline:
            improved = False
            stagnation += 1
            
            # Vary kick size: start small, increase when stuck
            if stagnation < 3:
                kick_size = 2
            elif stagnation < 6:
                kick_size = 4
            elif stagnation < 10:
                kick_size = 8
            else:
                kick_size = min(16, 2 ** (stagnation // 2))
            
            new_count, new_chosen, new_face_counts = kick_and_readd(
                chosen, face_counts, count, kick_size, rng, deadline
            )
            
            if new_count > count:
                count = new_count
                chosen = new_chosen
                face_counts = new_face_counts
                improved = True
                stagnation = 0
                
                if count > best_local:
                    best_local = count
                    best_local_chosen = chosen.copy()
                    best_local_face_counts = face_counts.copy()
            else:
                # Try a few more kicks with same size
                for _ in range(3):
                    if time.time() >= deadline:
                        break
                    new_count, new_chosen, new_face_counts = kick_and_readd(
                        chosen, face_counts, count, kick_size, rng, deadline
                    )
                    if new_count > count:
                        count = new_count
                        chosen = new_chosen
                        face_counts = new_face_counts
                        improved = True
                        stagnation = 0
                        if count > best_local:
                            best_local = count
                            best_local_chosen = chosen.copy()
                            best_local_face_counts = face_counts.copy()
                        break
        
        return best_local, best_local_chosen, best_local_face_counts

    attempt = 0
    while time.time() < deadline:
        count, chosen, face_counts = try_greedy(attempt)
        
        # Local search
        count, chosen, face_counts = local_search(chosen, face_counts, count, deadline)
        
        if count > best_count:
            best_count = count
            best_chosen = chosen.copy()
        attempt += 1
    
    out = []
    for idx in range(num_edges):
        if best_chosen[idx]:
            u, v, i = all_edges[idx]
            out.append([u, v])
    return out
