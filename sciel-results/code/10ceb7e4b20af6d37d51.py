import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
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
    seen_faces = set()
    face_edge_list = []
    
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
                    face_edge_list.append(idxs)
    
    num_faces = len(face_edge_list)
    
    # Precompute edge -> faces mapping
    edge_to_faces = [[] for _ in range(num_edges)]
    for f_idx, face in enumerate(face_edge_list):
        for e in face:
            edge_to_faces[e].append(f_idx)
    
    # 3. Improved greedy hitting set initialization
    def greedy_hit():
        active = np.ones(num_edges, dtype=np.uint8)
        face_counts = np.full(num_faces, 4, dtype=np.int32)
        removed = 0
        while True:
            active_faces = np.where(face_counts == 4)[0]
            if len(active_faces) == 0:
                break
            scores = np.zeros(num_edges, dtype=np.int32)
            for f_idx in active_faces:
                for e in face_edge_list[f_idx]:
                    scores[e] += 1
            max_score = np.max(scores)
            if max_score == 0:
                break
            candidates = np.where(scores == max_score)[0]
            # Among max-score edges, pick one that appears in the most active faces
            # (tie-breaking by degree in conflict graph)
            best_e = candidates[0]
            best_deg = -1
            for e in candidates:
                deg = sum(1 for ff in edge_to_faces[e] if face_counts[ff] == 4)
                if deg > best_deg:
                    best_deg = deg
                    best_e = e
            active[best_e] = 0
            removed += 1
            for f_idx in edge_to_faces[best_e]:
                face_counts[f_idx] -= 1
        return active, removed
    
    # 4. Simulated annealing with proper save/restore
    def count_violations(active):
        """Count faces with all 4 edges selected."""
        violations = 0
        for face in face_edge_list:
            if all(active[e] == 1 for e in face):
                violations += 1
        return violations
    
    def try_add_edge(active, face_counts, e):
        """Try to add edge e. Returns (success, new_face_counts)."""
        for f_idx in edge_to_faces[e]:
            if face_counts[f_idx] == 3:
                return False, face_counts
        new_counts = face_counts.copy()
        for f_idx in edge_to_faces[e]:
            new_counts[f_idx] += 1
        return True, new_counts
    
    def sa_refine(active, face_counts, rng, time_limit):
        """Simulated annealing with proper state save/restore."""
        T = 1.0
        T_min = 0.005
        cooling = 0.9998
        current_violations = count_violations(active)
        best_active = active.copy()
        best_violations = current_violations
        best_size = int(np.sum(active))
        
        removed_edges = np.where(active == 0)[0]
        
        while time.time() < time_limit:
            if len(removed_edges) == 0:
                break
            
            e = removed_edges[rng.integers(0, len(removed_edges))]
            
            success, new_counts = try_add_edge(active, face_counts, e)
            
            if success:
                # Save state before accepting
                old_active = active.copy()
                old_fc = face_counts.copy()
                old_removed = removed_edges.copy()
                
                active[e] = 1
                face_counts = new_counts
                removed_edges = np.where(active == 0)[0]
                current_violations = count_violations(active)
                current_size = int(np.sum(active))
                
                if current_violations < best_violations or (current_violations == best_violations and current_size > best_size):
                    best_violations = current_violations
                    best_size = current_size
                    best_active = active.copy()
            else:
                # Try kick: remove 1-2 edges from problematic face, re-add e
                problem_faces = []
                for f_idx in edge_to_faces[e]:
                    if face_counts[f_idx] == 3:
                        problem_faces.append(f_idx)
                
                if len(problem_faces) > 0:
                    f_idx = problem_faces[rng.integers(0, len(problem_faces))]
                    face = face_edge_list[f_idx]
                    
                    other_edges = [x for x in face if x != e]
                    num_remove = rng.integers(1, 3)
                    edges_to_remove = rng.choice(other_edges, size=min(num_remove, len(other_edges)), replace=False)
                    
                    # Save state before kick
                    old_active = active.copy()
                    old_fc = face_counts.copy()
                    old_removed = removed_edges.copy()
                    
                    # Execute kick
                    for re in edges_to_remove:
                        active[re] = 0
                        for ff in edge_to_faces[re]:
                            face_counts[ff] -= 1
                    
                    success2, new_counts2 = try_add_edge(active, face_counts, e)
                    if success2:
                        active[e] = 1
                        face_counts = new_counts2
                        removed_edges = np.where(active == 0)[0]
                        current_violations = count_violations(active)
                        current_size = int(np.sum(active))
                        
                        if current_violations < best_violations or (current_violations == best_violations and current_size > best_size):
                            best_violations = current_violations
                            best_size = current_size
                            best_active = active.copy()
                    else:
                        # Revert kick
                        active = old_active
                        face_counts = old_fc
                        removed_edges = old_removed
            
            T *= cooling
            if T < T_min:
                break
        
        return best_active, best_violations, best_size
    
    # 5. Local search: try swapping removed edge with active edge
    def local_search(active, face_counts, rng, time_limit):
        """Local search: try to swap a removed edge with an active edge to increase size."""
        best_active = active.copy()
        best_size = int(np.sum(active))
        best_violations = count_violations(active)
        
        removed_edges = np.where(active == 0)[0]
        active_edges = np.where(active == 1)[0]
        
        while time.time() < time_limit:
            if len(removed_edges) == 0:
                break
            
            e_add = removed_edges[rng.integers(0, len(removed_edges))]
            
            # Check if e_add can be added directly
            can_add = True
            for f_idx in edge_to_faces[e_add]:
                if face_counts[f_idx] == 3:
                    can_add = False
                    break
            
            if can_add:
                old_active = active.copy()
                old_fc = face_counts.copy()
                active[e_add] = 1
                for f_idx in edge_to_faces[e_add]:
                    face_counts[f_idx] += 1
                removed_edges = np.where(active == 0)[0]
                active_edges = np.where(active == 1)[0]
                current_size = int(np.sum(active))
                current_violations = count_violations(active)
                
                if current_size > best_size:
                    best_size = current_size
                    best_violations = current_violations
                    best_active = active.copy()
            else:
                # Need to remove some active edges first
                problem_faces = []
                for f_idx in edge_to_faces[e_add]:
                    if face_counts[f_idx] == 3:
                        problem_faces.append(f_idx)
                
                if not problem_faces:
                    continue
                
                f_idx = problem_faces[rng.integers(0, len(problem_faces))]
                face = face_edge_list[f_idx]
                other_active = [x for x in face if x != e_add and active[x] == 1]
                
                if not other_active:
                    continue
                
                # Try removing 1-2 active edges from this face
                num_remove = rng.integers(1, 3)
                edges_to_remove = rng.choice(other_active, size=min(num_remove, len(other_active)), replace=False)
                
                old_active = active.copy()
                old_fc = face_counts.copy()
                old_removed = removed_edges.copy()
                old_active_edges = active_edges.copy()
                
                for re in edges_to_remove:
                    active[re] = 0
                    for ff in edge_to_faces[re]:
                        face_counts[ff] -= 1
                
                # Now try to add e_add
                can_add2 = True
                for ff in edge_to_faces[e_add]:
                    if face_counts[ff] == 3:
                        can_add2 = False
                        break
                
                if can_add2:
                    active[e_add] = 1
                    for ff in edge_to_faces[e_add]:
                        face_counts[ff] += 1
                    removed_edges = np.where(active == 0)[0]
                    active_edges = np.where(active == 1)[0]
                    current_size = int(np.sum(active))
                    current_violations = count_violations(active)
                    
                    if current_size > best_size:
                        best_size = current_size
                        best_violations = current_violations
                        best_active = active.copy()
                else:
                    # Revert
                    active = old_active
                    face_counts = old_fc
                    removed_edges = old_removed
                    active_edges = old_active_edges
        
        return best_active, best_violations, best_size
    
    # 6. Final polish: iteratively add back any removable edges
    def final_polish(active, face_counts):
        changed = True
        while changed:
            changed = False
            for e in range(num_edges):
                if active[e] == 0:
                    can_add = True
                    for f_idx in edge_to_faces[e]:
                        if face_counts[f_idx] == 3:
                            can_add = False
                            break
                    if can_add:
                        active[e] = 1
                        for f_idx in edge_to_faces[e]:
                            face_counts[f_idx] += 1
                        changed = True
        return active, face_counts
    
    # Main search
    rng = np.random.default_rng(42)
    
    # Greedy init
    best_active, _ = greedy_hit()
    best_size = int(np.sum(best_active))
    best_violations = count_violations(best_active)
    
    face_counts = np.full(num_faces, 4, dtype=np.int32)
    for e in range(num_edges):
        if best_active[e] == 0:
            for f_idx in edge_to_faces[e]:
                face_counts[f_idx] -= 1
    
    # Run SA + local search with restarts
    num_restarts = 25
    
    for restart in range(num_restarts):
        if time.time() > deadline - 8:
            break
        
        if restart == 0:
            current_active = best_active.copy()
        else:
            current_active = best_active.copy()
            num_perturb = rng.integers(5, 16)
            all_edges = np.arange(num_edges)
            selected = rng.choice(all_edges, size=num_perturb, replace=False)
            for e in selected:
                if current_active[e] == 1:
                    current_active[e] = 0
        
        fc = np.full(num_faces, 4, dtype=np.int32)
        for e in range(num_edges):
            if current_active[e] == 0:
                for f_idx in edge_to_faces[e]:
                    fc[f_idx] -= 1
        
        # Phase 1: SA
        sa_limit = deadline - 10 - (num_restarts - restart) * 0.5
        if time.time() > sa_limit:
            sa_limit = time.time() + 1
        improved_active, improved_viol, improved_size = sa_refine(current_active, fc, rng, sa_limit)
        
        # Phase 2: Local search on the SA result
        fc2 = np.full(num_faces, 4, dtype=np.int32)
        for e in range(num_edges):
            if improved_active[e] == 0:
                for f_idx in edge_to_faces[e]:
                    fc2[f_idx] -= 1
        
        ls_limit = deadline - 8 - (num_restarts - restart) * 0.3
        if time.time() > ls_limit:
            ls_limit = time.time() + 1
        ls_active, ls_viol, ls_size = local_search(improved_active, fc2, rng, ls_limit)
        
        # Phase 3: Final polish
        fc3 = np.full(num_faces, 4, dtype=np.int32)
        for e in range(num_edges):
            if ls_active[e] == 0:
                for f_idx in edge_to_faces[e]:
                    fc3[f_idx] -= 1
        polished_active, _ = final_polish(ls_active, fc3)
        polished_size = int(np.sum(polished_active))
        polished_viol = count_violations(polished_active)
        
        if polished_size > best_size or (polished_size == best_size and polished_viol < best_violations):
            best_size = polished_size
            best_violations = polished_viol
            best_active = polished_active
    
    # Final polish on best
    if time.time() < deadline - 2:
        fc = np.full(num_faces, 4, dtype=np.int32)
        for e in range(num_edges):
            if best_active[e] == 0:
                for f_idx in edge_to_faces[e]:
                    fc[f_idx] -= 1
        best_active, _ = final_polish(best_active, fc)
    
    # Convert to list of edges
    result = []
    for i in range(num_edges):
        if best_active[i] == 1:
            result.append(list(idx_to_edge[i]))
            
    return result
