import time
import random
import sys

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))
    
    # 1. Build Edge List and Face Data
    edges = []
    edge_to_id = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_id[(u, v)] = len(edges)
                edges.append((u, v))
                
    edge_faces = [[] for _ in range(n_edges)]
    faces = []
    
    for i in range(d):
        for j in range(i + 1, d):
            others = [k for k in range(d) if k != i and k != j]
            for mask in range(2 ** len(others)):
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
                
                face_id = len(faces)
                faces.append([e1, e2, e3, e4])
                
                for e in [e1, e2, e3, e4]:
                    edge_faces[e].append(face_id)
                    
    n_faces = len(faces)
    
    # 2. Initial State: Greedy remove until valid (work with removed set)
    removed = set()
    face_counts = [0] * n_faces
    for e in range(n_edges):
        for f in edge_faces[e]:
            face_counts[f] += 1
    
    changed = True
    while changed:
        changed = False
        for f in range(n_faces):
            if face_counts[f] == 4:
                e_to_remove = random.choice(faces[f])
                if e_to_remove not in removed:
                    removed.add(e_to_remove)
                    for ff in edge_faces[e_to_remove]:
                        face_counts[ff] -= 1
                changed = True
                break
    
    current_removed = len(removed)
    best_removed = current_removed
    best_removed_set = set(removed)
    
    start_time = time.time()
    timeout = 110
    T = 1.0
    cooling_rate = 0.99995
    all_edges_list = list(range(n_edges))
    
    # Work with removed set: goal is to minimize |removed|
    # Move types:
    # 1. Un-remove an edge (if it doesn't create a full face) -> decreases |removed|
    # 2. Swap: remove one edge, un-remove another (size unchanged, restructures)
    # 3. Remove an edge (increases |removed|, for exploration)
    
    def can_unremove(e):
        """Check if un-removing edge e (i.e., adding it back) is valid."""
        for f in edge_faces[e]:
            if face_counts[f] == 3:
                return False
        return True
    
    def do_unremove(e):
        removed.discard(e)
        for f in edge_faces[e]:
            face_counts[f] += 1
    
    def do_remove(e):
        removed.add(e)
        for f in edge_faces[e]:
            face_counts[f] -= 1
    
    def try_swap_remove(e_rm, e_unrm):
        """Try to remove e_rm and un-remove e_unrm simultaneously."""
        if e_rm == e_unrm:
            return False
        if e_rm not in removed or e_unrm in removed:
            return False
        # Temporarily remove e_rm
        for f in edge_faces[e_rm]:
            face_counts[f] -= 1
        # Check if e_unrm can be un-removed
        valid = True
        for f in edge_faces[e_unrm]:
            if face_counts[f] == 3:
                valid = False
                break
        if valid:
            removed.add(e_rm)
            removed.discard(e_unrm)
            for f in edge_faces[e_unrm]:
                face_counts[f] += 1
            return True
        else:
            for f in edge_faces[e_rm]:
                face_counts[f] += 1
            return False
    
    steps = 0
    last_improvement_time = start_time
    no_improve_count = 0
    
    while time.time() - start_time < timeout:
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to un-remove an edge (decrease |removed|)
            e = random.choice(all_edges_list)
            if e in removed:
                if can_unremove(e):
                    do_unremove(e)
                    current_removed -= 1
                    if current_removed < best_removed:
                        best_removed = current_removed
                        best_removed_set = set(removed)
                        last_improvement_time = time.time()
                        no_improve_count = 0
                        T = max(T, 0.3)
        elif move_type < 0.85:
            # Swap: remove one, un-remove another
            e_rm = random.choice(all_edges_list)
            if e_rm not in removed:
                e_unrm = random.choice(all_edges_list)
                if e_unrm in removed and e_unrm != e_rm:
                    if try_swap_remove(e_rm, e_unrm):
                        pass  # size unchanged, accepted for exploration
        else:
            # Remove an edge (increase |removed|, for exploration)
            e = random.choice(all_edges_list)
            if e not in removed:
                if random.random() < 0.03:
                    do_remove(e)
                    current_removed += 1
        
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
        
        steps += 1
        
        if time.time() - last_improvement_time > 2.0:
            no_improve_count += 1
            if no_improve_count > 40:
                removed = set(best_removed_set)
                face_counts = [0] * n_faces
                for e in range(n_edges):
                    if e not in removed:
                        for f in edge_faces[e]:
                            face_counts[f] += 1
                current_removed = best_removed
                T = 1.0
                no_improve_count = 0
                last_improvement_time = time.time()
                break
    
    # 4. Final Polish: Greedy un-remove
    removed = set(best_removed_set)
    face_counts = [0] * n_faces
    for e in range(n_edges):
        if e not in removed:
            for f in edge_faces[e]:
                face_counts[f] += 1
    current_removed = best_removed
    
    improved = True
    while improved:
        improved = False
        for e in all_edges_list:
            if e in removed:
                if can_unremove(e):
                    do_unremove(e)
                    current_removed -= 1
                    improved = True
        if current_removed < best_removed:
            best_removed = current_removed
            best_removed_set = set(removed)
    
    # Build result
    result = []
    for i in range(n_edges):
        if i not in best_removed_set:
            result.append(list(edges[i]))
    
    return result
