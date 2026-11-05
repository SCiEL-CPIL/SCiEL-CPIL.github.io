"""Combined approach: greedy initialization with repair-based simulated annealing."""
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    d = 7
    n = 2 ** d
    m = d * (1 << (d - 1))  # 448
    
    # Generate all edges
    edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
                edge_to_idx[(y, x)] = idx
    
    # Precompute face membership
    # A face is defined by two dimensions (i, j) and a base vertex with those bits cleared
    face_members = {}  # face_id -> list of 4 edge indices
    edge_faces = [[] for _ in range(m)]  # edge_idx -> list of face_ids
    
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all base vertices (bits i and j are 0)
            for x in range(n):
                # Check if x has bits i and j cleared
                if (x & (1 << i)) or (x & (1 << j)):
                    continue
                
                # Four vertices of this face
                v0 = x
                v1 = x | (1 << i)
                v2 = x | (1 << j)
                v3 = x | (1 << i) | (1 << j)
                
                # Four edges
                e0 = (min(v0, v1), max(v0, v1))
                e1 = (min(v2, v3), max(v2, v3))
                e2 = (min(v0, v2), max(v0, v2))
                e3 = (min(v1, v3), max(v1, v3))
                
                idx0 = edge_to_idx.get(e0)
                idx1 = edge_to_idx.get(e1)
                idx2 = edge_to_idx.get(e2)
                idx3 = edge_to_idx.get(e3)
                
                if None in (idx0, idx1, idx2, idx3):
                    continue
                
                face_id = (i, j, x)
                face_members[face_id] = [idx0, idx1, idx2, idx3]
                for idx in [idx0, idx1, idx2, idx3]:
                    edge_faces[idx].append(face_id)
    
    num_faces = len(face_members)
    
    # Greedy initialization with random ordering
    selected = [False] * m
    count = 0
    face_counts = [0] * num_faces
    
    order = list(range(m))
    random.shuffle(order)
    
    for ei in order:
        # Check if adding ei creates a face with 4 edges
        ok = True
        for fi in edge_faces[ei]:
            if face_counts[fi] == 3:
                ok = False
                break
        if ok:
            selected[ei] = True
            count += 1
            for fi in edge_faces[ei]:
                face_counts[fi] += 1
    
    best_selected = selected[:]
    best_count = count
    
    # Simulated Annealing with repair
    T = 1.0
    cooling = 0.995
    min_T = 0.001
    
    while T > min_T:
        if time.time() > deadline:
            break
        
        # Pick random move: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add a random unselected edge
            ei = random.randint(0, m - 1)
            if selected[ei]:
                continue
            
            # Check conflicts
            conflicts = []
            for fi in edge_faces[ei]:
                if face_counts[fi] == 3:
                    conflicts.append(fi)
            
            if not conflicts:
                # No conflicts, just add
                selected[ei] = True
                count += 1
                delta = 1
                for fi in edge_faces[ei]:
                    face_counts[fi] += 1
            else:
                # Repair: add ei, then remove one edge from each conflicting face
                selected[ei] = True
                count += 1
                for fi in edge_faces[ei]:
                    face_counts[fi] += 1
                
                removed = 0
                for fi in conflicts:
                    # Find a selected edge in this face (not ei) to remove
                    face_edges_list = face_members[fi]
                    candidates = [e for e in face_edges_list if e != ei and selected[e]]
                    if candidates:
                        rem = random.choice(candidates)
                        selected[rem] = False
                        count -= 1
                        removed += 1
                        for fj in edge_faces[rem]:
                            face_counts[fj] -= 1
                
                delta = 1 - removed
                # Only accept if we didn't lose too many edges
                if delta < 0:
                    # Revert
                    selected[ei] = False
                    count -= 1
                    for fi in edge_faces[ei]:
                        face_counts[fi] -= 1
                    for fi in conflicts:
                        face_edges_list = face_members[fi]
                        candidates = [e for e in face_edges_list if e != ei and not selected[e]]
                        # This revert logic is complex; just skip this move
                    continue
        
        elif move_type < 0.7:
            # Try to remove a random selected edge
            ei = random.randint(0, m - 1)
            if not selected[ei]:
                continue
            
            selected[ei] = False
            count -= 1
            delta = -1
            for fi in edge_faces[ei]:
                face_counts[fi] -= 1
        
        else:
            # Swap: remove one selected, add one unselected
            ei_remove = random.randint(0, m - 1)
            ei_add = random.randint(0, m - 1)
            
            if not selected[ei_remove] or selected[ei_add] or ei_remove == ei_add:
                continue
            
            # Remove ei_remove
            selected[ei_remove] = False
            count -= 1
            for fi in edge_faces[ei_remove]:
                face_counts[fi] -= 1
            
            # Try to add ei_add
            conflicts = []
            for fi in edge_faces[ei_add]:
                if face_counts[fi] == 3:
                    conflicts.append(fi)
            
            if not conflicts:
                selected[ei_add] = True
                count += 1
                for fi in edge_faces[ei_add]:
                    face_counts[fi] += 1
                delta = 0  # Net change is 0
            else:
                # Revert the removal
                selected[ei_remove] = True
                count += 1
                for fi in edge_faces[ei_remove]:
                    face_counts[fi] += 1
                continue
        
        # Acceptance criterion
        if delta > 0:
            # Always accept improvement
            if count > best_count:
                best_count = count
                best_selected = selected[:]
        elif delta == 0:
            # Accept with moderate probability
            if random.random() < 0.5:
                pass
            else:
                # Revert is complex for swap; just accept for simplicity
                pass
        else:
            # delta < 0: accept with probability exp(delta / T)
            if random.random() < math.exp(delta / T):
                pass
            else:
                # Revert
                if move_type < 0.4:
                    # This shouldn't happen due to earlier check
                    pass
                elif move_type < 0.7:
                    # Revert removal
                    selected[ei] = True
                    count += 1
                    for fi in edge_faces[ei]:
                        face_counts[fi] += 1
                else:
                    # Swap was accepted or reverted earlier
                    pass
        
        T *= cooling
    
    # Return best found
    out = []
    for ei in range(m):
        if best_selected[ei]:
            u, v = edges[ei]
            out.append([u, v])
    
    return out
