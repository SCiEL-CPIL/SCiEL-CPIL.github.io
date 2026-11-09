"""Structural change: simulated annealing on C4-free edge sets of Q7.
Starts from a greedy construction, then uses SA with 1-flip and 2-swap moves.
The key structural difference: SA can accept worsening moves (escape local optima),
and we use a face-based conflict count for fast energy evaluation."""
import numpy as np
import time
import random
import math

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
    
    # Precompute faces
    faces = []
    face_edges = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            for x in range(n_vertices):
                if (x & mask_i) or (x & mask_j):
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
    
    # For each edge, precompute the 3 other edges in each face it belongs to
    edge_face_others = [[] for _ in range(n_edges)]
    for fidx, face in enumerate(faces):
        for pos, e in enumerate(face):
            others = [face[k] for k in range(4) if k != pos]
            edge_face_others[e].append((fidx, others))
    
    deadline = time.time() + 115
    rng = random.Random(12345)
    
    def count_edges(sel):
        return int(np.sum(sel))
    
    def can_add(idx, sel):
        for fidx, others in edge_face_others[idx]:
            if sel[others[0]] and sel[others[1]] and sel[others[2]]:
                return False
        return True
    
    def can_swap(idx_rm, idx_add, sel):
        for fidx, others in edge_face_others[idx_add]:
            cnt = 0
            for e in others:
                if e == idx_rm:
                    continue
                if sel[e]:
                    cnt += 1
            if cnt >= 3:
                return False
        return True
    
    def greedy_fill(sel, order=None, max_time=None):
        if order is None:
            order = list(range(n_edges))
            rng.shuffle(order)
        for idx in order:
            if max_time is not None and time.time() > max_time:
                break
            if not sel[idx]:
                if can_add(idx, sel):
                    sel[idx] = True
    
    def complementary_deletion(rng_local):
        """Start with all edges, remove to break all C4s."""
        sel = np.ones(n_edges, dtype=bool)
        
        face_order = list(range(num_faces))
        rng_local.shuffle(face_order)
        
        for fidx in face_order:
            if time.time() > deadline:
                break
            face = faces[fidx]
            if sel[face[0]] and sel[face[1]] and sel[face[2]] and sel[face[3]]:
                # Remove edge with most conflicts
                best_to_remove = None
                best_score = -1
                for e in face:
                    if not sel[e]:
                        continue
                    conflict_count = 0
                    for ofidx, others in edge_face_others[e]:
                        if ofidx == fidx:
                            continue
                        if all(sel[oe] for oe in others):
                            conflict_count += 1
                    if conflict_count > best_score:
                        best_score = conflict_count
                        best_to_remove = e
                if best_to_remove is not None:
                    sel[best_to_remove] = False
        
        return sel
    
    def simulated_annealing(sel, rng_local, t_start, t_end, n_iter):
        """SA with 1-flip and 2-swap moves."""
        current_sel = sel.copy()
        current_count = count_edges(current_sel)
        best_sel = current_sel.copy()
        best_count = current_count
        
        # Track selected/missing edges for fast sampling
        selected_edges = [i for i in range(n_edges) if current_sel[i]]
        missing_edges = [i for i in range(n_edges) if not current_sel[i]]
        
        for it in range(n_iter):
            if time.time() > deadline - 2:
                break
            
            # Temperature schedule: linear cooling
            frac = it / max(n_iter, 1)
            temp = t_start * (t_end / t_start) ** frac
            
            # Choose move type: 60% 1-flip, 40% 2-swap
            move_type = rng_local.random()
            
            if move_type < 0.6 and len(missing_edges) > 0:
                # 1-flip: try to add a random missing edge
                idx = rng_local.choice(missing_edges)
                if can_add(idx, current_sel):
                    # Accept (always improves)
                    current_sel[idx] = True
                    current_count += 1
                    missing_edges.remove(idx)
                    selected_edges.append(idx)
                    
                    if current_count > best_count:
                        best_count = current_count
                        best_sel = current_sel.copy()
                # If can't add, skip (no worsening move for 1-flip)
            
            elif len(selected_edges) > 0 and len(missing_edges) > 0:
                # 2-swap: remove one selected, add one missing
                idx_rm = rng_local.choice(selected_edges)
                idx_add = rng_local.choice(missing_edges)
                
                if can_swap(idx_rm, idx_add, current_sel):
                    # Check if this swap improves count (it does: -1 +1 = 0 net, but may enable future adds)
                    # Actually swap keeps count same but changes structure
                    # For SA, we accept with probability based on "energy" = -count
                    # Since count doesn't change, delta_e = 0, always accept
                    current_sel[idx_rm] = False
                    current_sel[idx_add] = True
                    # Update lists
                    selected_edges.remove(idx_rm)
                    missing_edges.remove(idx_add)
                    selected_edges.append(idx_add)
                    missing_edges.append(idx_rm)
                    
                    # After swap, try to add more edges (greedy step)
                    added_any = False
                    for _ in range(10):
                        if not missing_edges:
                            break
                        idx_try = rng_local.choice(missing_edges)
                        if can_add(idx_try, current_sel):
                            current_sel[idx_try] = True
                            current_count += 1
                            missing_edges.remove(idx_try)
                            selected_edges.append(idx_try)
                            added_any = True
                            if current_count > best_count:
                                best_count = current_count
                                best_sel = current_sel.copy()
                        else:
                            # Try a few more
                            for _ in range(5):
                                idx_try2 = rng_local.choice(missing_edges)
                                if can_add(idx_try2, current_sel):
                                    current_sel[idx_try2] = True
                                    current_count += 1
                                    missing_edges.remove(idx_try2)
                                    selected_edges.append(idx_try2)
                                    if current_count > best_count:
                                        best_count = current_count
                                        best_sel = current_sel.copy()
                                    break
        
        return best_sel, best_count
    
    def local_search_1flip(sel, rng_local, max_iters=200):
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            for _ in range(80):
                idx = rng_local.randint(0, n_edges - 1)
                if sel[idx]:
                    continue
                if can_add(idx, sel):
                    sel[idx] = True
                    improved = True
                    break
        return sel
    
    def local_search_2swap(sel, rng_local, max_iters=150):
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            for _ in range(40):
                idx_rm = rng_local.randint(0, n_edges - 1)
                if not sel[idx_rm]:
                    continue
                idx_add = rng_local.randint(0, n_edges - 1)
                if sel[idx_add]:
                    continue
                if can_swap(idx_rm, idx_add, sel):
                    sel[idx_rm] = False
                    sel[idx_add] = True
                    improved = True
                    break
        return sel
    
    best_sel = None
    best_count = 0
    
    # Multiple restarts with different strategies
    restart = 0
    while time.time() < deadline - 5:
        restart += 1
        rng_local = random.Random(1000 + restart * 7919)
        
        # Alternate between complementary deletion and greedy from scratch
        if restart % 2 == 1:
            # Complementary deletion
            sel = complementary_deletion(rng_local)
        else:
            # Start empty, greedy fill
            sel = np.zeros(n_edges, dtype=bool)
            greedy_fill(sel, max_time=deadline - 10)
        
        cnt = count_edges(sel)
        
        # Quick local search
        if time.time() < deadline - 10:
            sel = local_search_1flip(sel, rng_local, max_iters=50)
            sel = local_search_2swap(sel, rng_local, max_iters=50)
            cnt = count_edges(sel)
        
        # Simulated annealing
        if time.time() < deadline - 5:
            t_start = 1.0
            t_end = 0.01
            # Use remaining time to determine iterations
            remaining = deadline - 5 - time.time()
            # Estimate ~0.001s per iteration
            n_iter = int(remaining / 0.001)
            n_iter = max(n_iter, 1000)
            n_iter = min(n_iter, 50000)
            
            sa_sel, sa_count = simulated_annealing(sel, rng_local, t_start, t_end, n_iter)
            cnt = max(cnt, sa_count)
            if sa_count > cnt:
                sel = sa_sel
        
        if cnt > best_count:
            best_count = cnt
            best_sel = sel.copy()
        
        # Additional polish on best
        if best_count >= 300 and time.time() < deadline - 5:
            sel2 = best_sel.copy()
            sel2 = local_search_1flip(sel2, rng_local, max_iters=100)
            sel2 = local_search_2swap(sel2, rng_local, max_iters=100)
            cnt2 = count_edges(sel2)
            if cnt2 > best_count:
                best_count = cnt2
                best_sel = sel2
    
    # Final polish
    if best_sel is not None and time.time() < deadline - 2:
        rng_final = random.Random(99999)
        best_sel = local_search_1flip(best_sel, rng_final, max_iters=200)
        best_sel = local_search_2swap(best_sel, rng_final, max_iters=200)
    
    # Convert to edge list
    if best_sel is None:
        return []
    
    edges = []
    for i in range(n_edges):
        if best_sel[i]:
            u, v = idx_to_edge[i]
            edges.append([u, v])
    
    return edges
