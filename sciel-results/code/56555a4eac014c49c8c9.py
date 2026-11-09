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
    
    # Precompute for faster addability checks
    # For each edge, store the face indices it belongs to
    edge_face_indices = np.array(face_edges, dtype=object)
    
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
    
    def can_add_fast(idx, sel):
        """Faster addability check using precomputed data."""
        for fidx in face_edges[idx]:
            face = faces[fidx]
            # Check if the other 3 edges are all selected
            cnt = 0
            for e in face:
                if e != idx and sel[e]:
                    cnt += 1
            if cnt == 3:
                return False
        return True
    
    def can_remove_allow_add(idx_remove, idx_add, sel):
        """Check if removing idx_remove and adding idx_add keeps C4-free."""
        for fidx in face_edges[idx_add]:
            face = faces[fidx]
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
    
    # Phase 1: Improved complementary deletion with better heuristic
    def complementary_deletion():
        sel = np.ones(n_edges, dtype=bool)
        face_order = list(range(num_faces))
        random.shuffle(face_order)
        
        removed_count = 0
        for fidx in face_order:
            if time.time() > deadline:
                break
            face = faces[fidx]
            if all(sel[e] for e in face):
                # Score each edge by how many other violated faces it's in
                best_to_remove = None
                best_score = float('inf')
                for e in face:
                    if sel[e]:
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
    
    # Phase 2: Vectorized greedy fill
    def greedy_fill(sel, order=None):
        if order is None:
            order = list(range(n_edges))
            random.shuffle(order)
        added = 0
        for idx in order:
            if time.time() > deadline:
                break
            if not sel[idx]:
                if can_add_fast(idx, sel):
                    sel[idx] = True
                    added += 1
        return added
    
    # Phase 3: Enhanced local search with 2-opt and 3-opt moves
    def local_search(sel, max_time=30):
        start = time.time()
        current_count = int(np.sum(sel))
        best_count = current_count
        best_sel = sel.copy()
        
        no_improve_rounds = 0
        max_no_improve = 5
        
        while time.time() - start < max_time and time.time() < deadline:
            improved = False
            
            # Try adding all possible edges
            missing = [i for i in range(n_edges) if not sel[i]]
            random.shuffle(missing)
            for idx in missing:
                if time.time() > deadline:
                    break
                if can_add_fast(idx, sel):
                    sel[idx] = True
                    current_count += 1
                    improved = True
            
            if current_count > best_count:
                best_count = current_count
                best_sel = sel.copy()
            
            if not improved:
                no_improve_rounds += 1
                if no_improve_rounds >= max_no_improve:
                    break
            else:
                no_improve_rounds = 0
            
            # Try 1-1 swaps
            present = [i for i in range(n_edges) if sel[i]]
            if not present:
                break
            
            swap_attempts = 0
            max_swap_attempts = 100
            while swap_attempts < max_swap_attempts and time.time() - start < max_time and time.time() < deadline:
                swap_attempts += 1
                rem = random.choice(present)
                missing_now = [i for i in range(n_edges) if not sel[i]]
                if not missing_now:
                    break
                add = random.choice(missing_now)
                
                if can_remove_allow_add(rem, add, sel):
                    sel[rem] = False
                    sel[add] = True
                    # Try to add more edges
                    missing_after = [i for i in range(n_edges) if not sel[i]]
                    random.shuffle(missing_after)
                    for idx in missing_after:
                        if time.time() > deadline:
                            break
                        if can_add_fast(idx, sel):
                            sel[idx] = True
                            current_count += 1
                            improved = True
                    
                    if current_count > best_count:
                        best_count = current_count
                        best_sel = sel.copy()
                    break
            
            # Try 2-2 swaps (remove 2, add 2)
            if not improved and len(present) >= 2:
                rem1, rem2 = random.sample(present, 2)
                missing_now = [i for i in range(n_edges) if not sel[i]]
                if len(missing_now) >= 2:
                    add1, add2 = random.sample(missing_now, 2)
                    
                    # Check if removing rem1, rem2 and adding add1, add2 is valid
                    # Temporarily remove
                    old_state = sel.copy()
                    sel[rem1] = False
                    sel[rem2] = False
                    
                    # Try to add add1 and add2
                    can_add_both = True
                    if can_add_fast(add1, sel):
                        sel[add1] = True
                        if can_add_fast(add2, sel):
                            sel[add2] = True
                            # Try to add more
                            missing_after = [i for i in range(n_edges) if not sel[i]]
                            random.shuffle(missing_after)
                            for idx in missing_after:
                                if time.time() > deadline:
                                    break
                                if can_add_fast(idx, sel):
                                    sel[idx] = True
                                    current_count += 2  # We added 2 net
                                    improved = True
                            
                            if current_count > best_count:
                                best_count = current_count
                                best_sel = sel.copy()
                        else:
                            sel[add1] = False
                            can_add_both = False
                    else:
                        can_add_both = False
                    
                    if not can_add_both:
                        sel[:] = old_state
            
            # Try kick: remove 3 random edges and refill
            if not improved and len(present) >= 3 and random.random() < 0.3:
                rem_set = random.sample(present, 3)
                for r in rem_set:
                    sel[r] = False
                current_count -= 3
                
                missing_after = [i for i in range(n_edges) if not sel[i]]
                random.shuffle(missing_after)
                for idx in missing_after:
                    if time.time() > deadline:
                        break
                    if can_add_fast(idx, sel):
                        sel[idx] = True
                        current_count += 1
                        improved = True
                
                if current_count > best_count:
                    best_count = current_count
                    best_sel = sel.copy()
        
        return best_sel, best_count
    
    # Main search loop with multiple strategies
    best_sel = None
    best_count = 0
    
    # Strategy 1: Complementary deletion + greedy fill + local search
    while time.time() < deadline - 15:
        sel = complementary_deletion()
        count = int(np.sum(sel))
        
        greedy_fill(sel)
        count = int(np.sum(sel))
        
        if count > best_count:
            best_count = count
            best_sel = sel.copy()
        
        if time.time() < deadline - 20:
            remaining_time = deadline - time.time() - 15
            sel_ls, count_ls = local_search(sel, max_time=min(40, remaining_time))
            if count_ls > best_count:
                best_count = count_ls
                best_sel = sel_ls
    
    # Strategy 2: Start from a random subset and build up
    if time.time() < deadline - 30:
        for trial in range(3):
            if time.time() > deadline - 30:
                break
            # Start with ~70% of edges
            sel = np.random.random(n_edges) < 0.7
            # Remove violated faces
            for _ in range(10):
                violated = []
                for fidx in range(num_faces):
                    face = faces[fidx]
                    if all(sel[e] for e in face):
                        violated.append(fidx)
                if not violated:
                    break
                for fidx in violated:
                    face = faces[fidx]
                    # Remove one random edge from each violated face
                    removable = [e for e in face if sel[e]]
                    if removable:
                        sel[random.choice(removable)] = False
            
            greedy_fill(sel)
            count = int(np.sum(sel))
            
            if count > best_count:
                best_count = count
                best_sel = sel.copy()
            
            if time.time() < deadline - 35:
                sel_ls, count_ls = local_search(sel, max_time=min(20, deadline - time.time() - 30))
                if count_ls > best_count:
                    best_count = count_ls
                    best_sel = sel_ls
    
    # Final polish
    if best_sel is None:
        best_sel = complementary_deletion()
        greedy_fill(best_sel)
        best_count = int(np.sum(best_sel))
    
    # Final aggressive polish
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        missing = [i for i in range(n_edges) if not best_sel[i]]
        random.shuffle(missing)
        added_any = False
        for idx in missing:
            if time.time() > deadline - 5:
                break
            if can_add_fast(idx, best_sel):
                best_sel[idx] = True
                best_count += 1
                added_any = True
        if not added_any:
            break
    
    # Convert to list of edges
    result = []
    for i in range(n_edges):
        if best_sel[i]:
            result.append(list(idx_to_edge[i]))
    
    return result
