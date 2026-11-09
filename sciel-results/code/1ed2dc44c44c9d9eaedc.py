import time
import random
import numpy as np

def construct():
    d = 7
    n_vertices = 1 << d  # 128
    n_edges = n_vertices * d // 2  # 448
    
    # Build edge list and mapping
    edge_to_idx = {}
    idx_to_edge = []
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(idx_to_edge)
                idx_to_edge.append((u, v))
                edge_to_idx[(u, v)] = idx
    
    # Precompute faces: each face is a set of 4 edge indices
    faces = []
    face_edges = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            for x in range(n_vertices):
                if x & mask_i or x & mask_j:
                    continue
                v0 = x
                v1 = x ^ mask_i
                v2 = x ^ mask_j
                v3 = x ^ mask_i ^ mask_j
                
                e01 = (min(v0, v1), max(v0, v1))
                e02 = (min(v0, v2), max(v0, v2))
                e13 = (min(v1, v3), max(v1, v3))
                e23 = (min(v2, v3), max(v2, v3))
                
                idx01 = edge_to_idx[e01]
                idx02 = edge_to_idx[e02]
                idx13 = edge_to_idx[e13]
                idx23 = edge_to_idx[e23]
                
                face_idx = len(faces)
                faces.append([idx01, idx02, idx13, idx23])
                
                for ei in [idx01, idx02, idx13, idx23]:
                    face_edges[ei].append(face_idx)
    
    num_faces = len(faces)
    faces_arr = np.array(faces, dtype=np.int32)  # shape (num_faces, 4)
    
    def face_violation_count(sel):
        """Return number of faces where all 4 edges are selected."""
        if isinstance(sel, np.ndarray):
            return int(np.sum(np.all(sel[faces_arr], axis=1)))
        else:
            cnt = 0
            for f in faces:
                if all(sel[e] for e in f):
                    cnt += 1
            return cnt
    
    def can_add(idx, sel):
        """Check if edge idx can be added without creating a C4."""
        for fidx in face_edges[idx]:
            face = faces[fidx]
            others = [e for e in face if e != idx]
            if all(sel[e] for e in others):
                return False
        return True
    
    def can_remove_allow_add(idx_remove, idx_add, sel):
        """Check if removing idx_remove and adding idx_add keeps C4-free."""
        # After removal, check if idx_add can be added
        for fidx in face_edges[idx_add]:
            face = faces[fidx]
            # Count how many of the other 3 edges are selected, excluding idx_remove
            cnt = 0
            for e in face:
                if e == idx_add:
                    continue
                if e == idx_remove:
                    continue
                if sel[e]:
                    cnt += 1
            if cnt >= 3:
                return False
        return True
    
    deadline = time.time() + 115
    
    # Phase 1: Complementary deletion - start with all edges, remove to break C4s
    def complementary_deletion():
        sel = np.ones(n_edges, dtype=bool)
        # Randomize order of faces to process
        face_order = list(range(num_faces))
        random.shuffle(face_order)
        
        removed_count = 0
        for fidx in face_order:
            if time.time() > deadline:
                break
            face = faces[fidx]
            # Check if all 4 edges are still selected
            if all(sel[e] for e in face):
                # Remove one edge from this face - pick the one with fewest remaining conflicts
                best_to_remove = None
                best_score = float('inf')
                for e in face:
                    if sel[e]:
                        # Count how many other faces this edge is in (that are still fully selected)
                        conflict_count = 0
                        for other_fidx in face_edges[e]:
                            if other_fidx == fidx:
                                continue
                            other_face = faces[other_fidx]
                            if all(sel[oe] for oe in other_face):
                                conflict_count += 1
                        if conflict_count < best_score:
                            best_score = conflict_count
                            best_to_remove = e
                if best_to_remove is not None:
                    sel[best_to_remove] = False
                    removed_count += 1
        return sel
    
    # Phase 2: Greedy fill - add back edges that can be added
    def greedy_fill(sel, order=None):
        if order is None:
            order = list(range(n_edges))
            random.shuffle(order)
        added = 0
        for idx in order:
            if time.time() > deadline:
                break
            if not sel[idx]:
                if can_add(idx, sel):
                    sel[idx] = True
                    added += 1
        return added
    
    # Phase 3: Local search with swaps
    def local_search(sel, max_time=30):
        start = time.time()
        current_count = int(np.sum(sel))
        best_count = current_count
        best_sel = sel.copy()
        
        while time.time() - start < max_time and time.time() < deadline:
            improved = False
            
            # Try adding edges
            missing = [i for i in range(n_edges) if not sel[i]]
            random.shuffle(missing)
            for idx in missing:
                if time.time() > deadline:
                    break
                if can_add(idx, sel):
                    sel[idx] = True
                    current_count += 1
                    improved = True
            
            if current_count > best_count:
                best_count = current_count
                best_sel = sel.copy()
            
            if not improved:
                # Try swaps: remove one, add one or more
                present = [i for i in range(n_edges) if sel[i]]
                if not present:
                    break
                
                # Try a few random swap attempts
                attempts = 0
                while attempts < 50 and time.time() - start < max_time and time.time() < deadline:
                    attempts += 1
                    rem = random.choice(present)
                    missing_now = [i for i in range(n_edges) if not sel[i]]
                    if not missing_now:
                        break
                    add = random.choice(missing_now)
                    
                    if can_remove_allow_add(rem, add, sel):
                        sel[rem] = False
                        sel[add] = True
                        # Now try to add more
                        missing_after = [i for i in range(n_edges) if not sel[i]]
                        random.shuffle(missing_after)
                        for idx in missing_after:
                            if time.time() > deadline:
                                break
                            if can_add(idx, sel):
                                sel[idx] = True
                                current_count += 1
                                improved = True
                        
                        if current_count > best_count:
                            best_count = current_count
                            best_sel = sel.copy()
                        break  # Restart after a successful swap
                
                # If no swap worked, try a kick: remove 2 random edges
                if not improved and len(present) >= 2:
                    rem1, rem2 = random.sample(present, 2)
                    sel[rem1] = False
                    sel[rem2] = False
                    current_count -= 2
                    
                    # Greedy refill
                    missing_after = [i for i in range(n_edges) if not sel[i]]
                    random.shuffle(missing_after)
                    for idx in missing_after:
                        if time.time() > deadline:
                            break
                        if can_add(idx, sel):
                            sel[idx] = True
                            current_count += 1
                            improved = True
                    
                    if current_count > best_count:
                        best_count = current_count
                        best_sel = sel.copy()
        
        return best_sel, best_count
    
    # Main search loop
    best_sel = None
    best_count = 0
    
    # Multiple restarts with complementary deletion
    while time.time() < deadline - 10:
        # Complementary deletion
        sel = complementary_deletion()
        count = int(np.sum(sel))
        
        # Greedy fill
        greedy_fill(sel)
        count = int(np.sum(sel))
        
        if count > best_count:
            best_count = count
            best_sel = sel.copy()
        
        # Local search
        if time.time() < deadline - 15:
            sel_ls, count_ls = local_search(sel, max_time=min(30, deadline - time.time() - 10))
            if count_ls > best_count:
                best_count = count_ls
                best_sel = sel_ls
    
    # Final verification and polish
    if best_sel is None:
        best_sel = complementary_deletion()
        greedy_fill(best_sel)
        best_count = int(np.sum(best_sel))
    
    # Final polish: try to add any remaining edges
    for idx in range(n_edges):
        if time.time() > deadline:
            break
        if not best_sel[idx]:
            if can_add(idx, best_sel):
                best_sel[idx] = True
                best_count += 1
    
    # Convert to list of edges
    result = []
    for i in range(n_edges):
        if best_sel[i]:
            result.append(list(idx_to_edge[i]))
    
    return result
