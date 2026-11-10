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
    
    # Precompute face edge indices for fast lookup
    # face_edge_indices[idx] is a list of tuples of 3 indices (the other edges in the face)
    # Also, for each face, we need to know which edges are in it to update counts.
    # Let's create a list of faces. Each face is a set of 4 edge indices.
    faces = []
    face_id = 0
    face_to_edges = {} # face_id -> set of 4 edge indices
    
    # Generate all 448*6/2? No, number of faces is 2^(7-2) * C(7,2) = 32 * 21 = 672 faces.
    # Each face has 4 edges.
    # We can iterate through all pairs of bits (i, j) and all combinations of other bits.
    
    for i in range(d):
        for j in range(i + 1, d):
            # Fix other 5 bits
            for mask in range(1 << d):
                # Ensure mask has 0 bits at i and j?
                # Actually, we can just iterate through all vertices u.
                # A face is defined by fixing all bits except i and j.
                # The 4 vertices are u, u^(1<<i), u^(1<<j), u^(1<<i)^(1<<j).
                # We can generate unique faces by requiring u has 0 at i and j.
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

    def try_greedy(order_seed=None):
        rng = random.Random(order_seed)
        chosen = [False] * num_edges
        count = 0
        
        indices = list(range(num_edges))
        rng.shuffle(indices)
        
        # face_counts[f] = number of chosen edges in face f
        face_counts = [0] * num_faces
        
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

    def local_search(chosen, face_counts, count, deadline):
        # Returns improved count, chosen, face_counts
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each chosen edge
            for r_idx in range(num_edges):
                if not chosen[r_idx]:
                    continue
                
                # Tentatively remove r_idx
                chosen[r_idx] = False
                count -= 1
                for f_idx in edge_to_faces[r_idx]:
                    face_counts[f_idx] -= 1
                
                # Find all edges that are now valid
                # An edge e is valid if for all f in edge_to_faces[e], face_counts[f] < 3
                # We only need to check edges that were previously invalid?
                # Actually, we can just iterate through all unchosen edges and check validity.
                # But that's 448 * 6 checks per removal. 448 * 448 * 6 ~ 1.2M ops per pass.
                # 110s / 1.2M ~ 90 passes? Might be slow in Python.
                
                # Optimization: Only check edges that share a face with r_idx.
                # If an edge e does not share a face with r_idx, its blocking status didn't change.
                # If it was invalid before, it's still invalid.
                # If it was valid before, it was already chosen (or we could have chosen it).
                # Wait, if it was valid before, greedy would have chosen it?
                # Not necessarily, greedy is random.
                # But in local search, we assume we are at a local optimum of "no single add possible".
                # So any unchosen edge e must be invalid.
                # e becomes valid only if all its blocking faces contained r_idx.
                
                candidates = []
                for e_idx in range(num_edges):
                    if chosen[e_idx]:
                        continue
                    # Check if e_idx is valid
                    valid = True
                    for f_idx in edge_to_faces[e_idx]:
                        if face_counts[f_idx] == 3:
                            valid = False
                            break
                    if valid:
                        candidates.append(e_idx)
                
                # We need to pick at least 2 candidates that don't conflict.
                # Conflict: e1, e2 form a 4-cycle with each other?
                # No, e1, e2 conflict if adding both creates a 4-cycle.
                # This happens if they share a face?
                # If e1 and e2 are in the same face, and the other 2 edges of that face are chosen,
                # then adding e1 makes count 4? No, if other 2 are chosen, count was 2.
                # Adding e1 makes 3. Adding e2 makes 4. Violation.
                # So e1 and e2 conflict if they are in a face F where the other 2 edges are chosen.
                
                # Let's find a pair e1, e2 from candidates that are compatible.
                added = []
                for i in range(len(candidates)):
                    e1 = candidates[i]
                    if e1 in added: continue
                    for j in range(i + 1, len(candidates)):
                        e2 = candidates[j]
                        if e2 in added: continue
                        
                        # Check if e1 and e2 are compatible
                        # They are incompatible if there is a face F containing both e1 and e2
                        # such that the other 2 edges in F are chosen.
                        compatible = True
                        # Intersect faces of e1 and e2
                        # Since num faces per edge is small (6), we can just check all faces of e1
                        for f_idx in edge_to_faces[e1]:
                            if e2 in faces[f_idx]:
                                # Check other 2 edges
                                face_edges = faces[f_idx]
                                # Count chosen edges in this face (excluding e1, e2 which are not chosen yet)
                                # face_counts[f_idx] currently includes r_idx? No, we removed r_idx.
                                # face_counts[f_idx] is count of chosen edges in F.
                                # e1, e2 are NOT chosen.
                                # So if face_counts[f_idx] >= 2, it means at least 2 other edges are chosen.
                                # Since there are only 4 edges in a face, and e1, e2 are 2 of them,
                                # the other 2 are the rest.
                                # If face_counts[f_idx] == 2, both other edges are chosen.
                                # Then adding e1 and e2 makes 4. Conflict.
                                if face_counts[f_idx] >= 2:
                                    compatible = False
                                    break
                        if compatible:
                            added.append(e1)
                            added.append(e2)
                            break
                    if len(added) >= 2:
                        break
                
                if len(added) >= 2:
                    # Commit the move
                    for e_idx in added:
                        chosen[e_idx] = True
                        count += 1
                        for f_idx in edge_to_faces[e_idx]:
                            face_counts[f_idx] += 1
                    improved = True
                    break # Restart the outer loop? Or continue?
                    # Breaking allows us to re-evaluate from scratch, which is safer.
                
                # If no improvement, restore r_idx
                chosen[r_idx] = True
                count += 1
                for f_idx in edge_to_faces[r_idx]:
                    face_counts[f_idx] += 1
        
        return count, chosen, face_counts

    attempt = 0
    while time.time() < deadline:
        count, chosen, face_counts = try_greedy(attempt)
        
        # Local search
        count, chosen, face_counts = local_search(chosen, face_counts, count, deadline)
        
        if count > best_count:
            best_count = count
            best_chosen = chosen
        attempt += 1
    
    out = []
    for idx in range(num_edges):
        if best_chosen[idx]:
            u, v, i = all_edges[idx]
            out.append([u, v])
    return out
