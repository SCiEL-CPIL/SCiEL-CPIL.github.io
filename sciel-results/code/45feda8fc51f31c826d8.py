import time
import random
import numpy as np

def construct():
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))
    
    # Generate all edges
    edges = []
    edge_to_id = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_id[(u, v)] = len(edges)
                edges.append((u, v))
    
    # Precompute faces and edge-face incidence
    face_edges = []
    edge_faces = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            others = [k for k in range(d) if k != i and k != j]
            for mask in range(1 << len(others)):
                u = 0
                for idx, bit in enumerate(others):
                    if mask & (1 << idx):
                        u |= (1 << bit)
                
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = edge_to_id[(min(v1, v2), max(v1, v2))]
                e2 = edge_to_id[(min(v3, v4), max(v3, v4))]
                e3 = edge_to_id[(min(v1, v3), max(v1, v3))]
                e4 = edge_to_id[(min(v2, v4), max(v2, v4))]
                
                face_id = len(face_edges)
                face_edges.append([e1, e2, e3, e4])
                
                for e in [e1, e2, e3, e4]:
                    edge_faces[e].append(face_id)
    
    n_faces = len(face_edges)
    
    start_time = time.time()
    deadline = start_time + 140
    
    def greedy_construct(randomized=False, seed=None):
        if seed is not None:
            random.seed(seed)
        
        selected = [True] * n_edges
        face_count = [4] * n_faces
        violated = set(range(n_faces))
        
        while violated:
            edge_viol_count = [0] * n_edges
            for f in violated:
                for e in face_edges[f]:
                    edge_viol_count[e] += 1
            
            if max(edge_viol_count) == 0:
                break
            
            if randomized:
                max_c = max(edge_viol_count)
                candidates = [e for e in range(n_edges) if edge_viol_count[e] == max_c]
                best_e = random.choice(candidates)
            else:
                best_e = max(range(n_edges), key=lambda e: edge_viol_count[e])
            
            selected[best_e] = False
            for f in edge_faces[best_e]:
                face_count[f] -= 1
                if face_count[f] == 4:
                    violated.add(f)
                else:
                    violated.discard(f)
        
        return selected, face_count
    
    def try_add_edges(selected, face_count):
        unselected = [e for e in range(n_edges) if not selected[e]]
        random.shuffle(unselected)
        for e in unselected:
            can_add = True
            for f in edge_faces[e]:
                if face_count[f] >= 3:
                    can_add = False
                    break
            if can_add:
                selected[e] = True
                for f in edge_faces[e]:
                    face_count[f] += 1
    
    def simple_local_search(selected, face_count, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            unselected = [e for e in range(n_edges) if not selected[e]]
            random.shuffle(unselected)
            for e in unselected:
                if time.time() > time_limit:
                    break
                can_add = True
                for f in edge_faces[e]:
                    if face_count[f] >= 3:
                        can_add = False
                        break
                if can_add:
                    selected[e] = True
                    for f in edge_faces[e]:
                        face_count[f] += 1
                    improved = True
        return improved
    
    def swap_local_search(selected, face_count, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            selected_edges = [e for e in range(n_edges) if selected[e]]
            random.shuffle(selected_edges)
            
            for e_remove in selected_edges:
                if time.time() > time_limit:
                    break
                
                selected[e_remove] = False
                for f in edge_faces[e_remove]:
                    face_count[f] -= 1
                
                added_back = []
                unselected_now = [e for e in range(n_edges) if not selected[e]]
                random.shuffle(unselected_now)
                
                for e_add in unselected_now:
                    if len(added_back) >= 2:
                        break
                    can_add = True
                    for f in edge_faces[e_add]:
                        if face_count[f] >= 3:
                            can_add = False
                            break
                    if can_add:
                        selected[e_add] = True
                        for f in edge_faces[e_add]:
                            face_count[f] += 1
                        added_back.append(e_add)
                
                if len(added_back) < 2:
                    selected[e_remove] = True
                    for f in edge_faces[e_remove]:
                        face_count[f] += 1
                    for e_add in added_back:
                        selected[e_add] = False
                        for f in edge_faces[e_add]:
                            face_count[f] -= 1
                else:
                    improved = True
                    break
        
        return improved
    
    def triple_swap(selected, face_count, time_limit):
        """Remove 2 edges, try to add 3 back (net +1)."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            selected_edges = [e for e in range(n_edges) if selected[e]]
            random.shuffle(selected_edges)
            
            for i in range(min(len(selected_edges), 200)):
                if time.time() > time_limit:
                    break
                e1 = selected_edges[i]
                # Find a second edge to remove
                candidates = [e for e in selected_edges[i+1:] if e != e1]
                if not candidates:
                    continue
                random.shuffle(candidates)
                e2 = candidates[0]
                
                # Remove both
                for e in (e1, e2):
                    selected[e] = False
                    for f in edge_faces[e]:
                        face_count[f] -= 1
                
                added_back = []
                unselected_now = [e for e in range(n_edges) if not selected[e]]
                random.shuffle(unselected_now)
                
                for e_add in unselected_now:
                    if len(added_back) >= 3:
                        break
                    can_add = True
                    for f in edge_faces[e_add]:
                        if face_count[f] >= 3:
                            can_add = False
                            break
                    if can_add:
                        selected[e_add] = True
                        for f in edge_faces[e_add]:
                            face_count[f] += 1
                        added_back.append(e_add)
                
                if len(added_back) < 3:
                    # Revert
                    for e in (e1, e2):
                        selected[e] = True
                        for f in edge_faces[e]:
                            face_count[f] += 1
                    for e_add in added_back:
                        selected[e_add] = False
                        for f in edge_faces[e_add]:
                            face_count[f] -= 1
                else:
                    improved = True
                    break
        
        return improved
    
    def polish(selected, face_count, time_limit):
        """High-precision polish: try all possible single adds and 1-for-2 swaps exhaustively."""
        # First, try all single adds
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for e in range(n_edges):
                if not selected[e]:
                    can_add = all(face_count[f] < 3 for f in edge_faces[e])
                    if can_add:
                        selected[e] = True
                        for f in edge_faces[e]:
                            face_count[f] += 1
                        improved = True
        
        # Then try 1-for-2 swaps exhaustively
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            selected_edges = [e for e in range(n_edges) if selected[e]]
            for e_remove in selected_edges:
                if time.time() > time_limit:
                    break
                selected[e_remove] = False
                for f in edge_faces[e_remove]:
                    face_count[f] -= 1
                
                added_back = []
                for e_add in range(n_edges):
                    if selected[e_add]:
                        continue
                    can_add = all(face_count[f] < 3 for f in edge_faces[e_add])
                    if can_add:
                        selected[e_add] = True
                        for f in edge_faces[e_add]:
                            face_count[f] += 1
                        added_back.append(e_add)
                    if len(added_back) >= 2:
                        break
                
                if len(added_back) < 2:
                    selected[e_remove] = True
                    for f in edge_faces[e_remove]:
                        face_count[f] += 1
                    for e_add in added_back:
                        selected[e_add] = False
                        for f in edge_faces[e_add]:
                            face_count[f] -= 1
                else:
                    improved = True
                    break
        
        return improved
    
    # Phase 1: Initial greedy
    selected, face_count = greedy_construct(randomized=False)
    try_add_edges(selected, face_count)
    best_count = sum(selected)
    best_selected = selected.copy()
    best_face_count = face_count.copy()
    
    # Phase 2: Local search
    simple_local_search(selected, face_count, deadline)
    swap_local_search(selected, face_count, deadline)
    triple_swap(selected, face_count, deadline)
    
    current_count = sum(selected)
    if current_count > best_count:
        best_count = current_count
        best_selected = selected.copy()
        best_face_count = face_count.copy()
    
    # Phase 3: Randomized restarts
    for restart in range(500):
        if time.time() > deadline - 3:
            break
        
        seed = random.randint(0, 2**31 - 1)
        selected, face_count = greedy_construct(randomized=True, seed=seed)
        try_add_edges(selected, face_count)
        simple_local_search(selected, face_count, deadline - 2)
        swap_local_search(selected, face_count, deadline - 2)
        triple_swap(selected, face_count, deadline - 2)
        
        current_count = sum(selected)
        if current_count > best_count:
            best_count = current_count
            best_selected = selected.copy()
            best_face_count = face_count.copy()
    
    # Phase 4: Polish the best solution
    if time.time() < deadline - 1:
        polish(best_selected, best_face_count, deadline - 1)
        best_count = sum(best_selected)
    
    # Build result
    result = []
    for i, (u, v) in enumerate(edges):
        if best_selected[i]:
            result.append([u, v])
    
    return result
