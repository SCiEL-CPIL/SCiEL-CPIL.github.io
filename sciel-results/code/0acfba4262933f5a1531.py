import time
import random
import numpy as np

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))  # 448
    
    # Build edge list
    edges = []
    edge_to_id = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_id[(u, v)] = len(edges)
                edges.append((u, v))
    
    # Identify faces (2D squares)
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
    
    # Precompute face membership as sets for faster intersection
    face_edge_sets = [set(f) for f in faces]
    
    # Precompute for each edge, the set of faces it belongs to
    edge_face_sets = [set(faces) for faces in edge_faces]
    
    start_time = time.time()
    deadline = 145  # Safety margin
    
    def greedy_init():
        """Greedy hitting set initialization"""
        removed = set()
        face_counts = np.zeros(n_faces, dtype=np.int32)
        
        def add_to_removed(e):
            removed.add(e)
            for f in edge_faces[e]:
                face_counts[f] += 1
                
        uncovered_faces = set(range(n_faces))
        
        while uncovered_faces:
            # Pick a random uncovered face
            f = random.choice(list(uncovered_faces))
            
            best_e = -1
            max_cover = -1
            for e in faces[f]:
                if e in removed:
                    continue
                cover = 0
                for ff in edge_faces[e]:
                    if ff in uncovered_faces:
                        cover += 1
                if cover > max_cover:
                    max_cover = cover
                    best_e = e
            
            if best_e == -1:
                best_e = faces[f][0]
                
            add_to_removed(best_e)
            
            for ff in edge_faces[best_e]:
                if face_counts[ff] == 1:
                    uncovered_faces.discard(ff)
                    
        return removed, face_counts
    
    def try_deletions(removed, face_counts):
        """Try to delete non-critical edges"""
        improved = False
        for e in list(removed):
            critical = False
            for f in edge_faces[e]:
                if face_counts[f] == 1:
                    critical = True
                    break
            if not critical:
                removed.remove(e)
                for f in edge_faces[e]:
                    face_counts[f] -= 1
                improved = True
        return improved
    
    def try_swaps(removed, face_counts):
        """Try 1-for-1 swaps to enable future deletions"""
        improved = False
        for e_out in list(removed):
            critical_faces = []
            for f in edge_faces[e_out]:
                if face_counts[f] == 1:
                    critical_faces.append(f)
            
            if not critical_faces:
                continue
                
            # Find candidates for e_in that cover all critical faces
            candidates = face_edge_sets[critical_faces[0]].copy()
            for f in critical_faces[1:]:
                candidates &= face_edge_sets[f]
            
            candidates.discard(e_out)
            
            valid_candidates = [e for e in candidates if e not in removed]
            
            if valid_candidates:
                # Pick a random candidate
                e_in = random.choice(valid_candidates)
                
                # Check if this swap enables any deletions
                # Simulate the swap
                old_counts = face_counts.copy()
                
                # Remove e_out
                for f in edge_faces[e_out]:
                    face_counts[f] -= 1
                removed.discard(e_out)
                
                # Add e_in
                for f in edge_faces[e_in]:
                    face_counts[f] += 1
                removed.add(e_in)
                
                # Check if any edge can now be deleted
                can_delete = False
                for e in list(removed):
                    if e == e_in:
                        continue
                    critical = False
                    for f in edge_faces[e]:
                        if face_counts[f] == 1:
                            critical = True
                            break
                    if not critical:
                        can_delete = True
                        break
                
                if can_delete:
                    improved = True
                    # Keep the swap
                else:
                    # Revert
                    for f in edge_faces[e_in]:
                        face_counts[f] -= 1
                    removed.discard(e_in)
                    for f in edge_faces[e_out]:
                        face_counts[f] += 1
                    removed.add(e_out)
                    
        return improved
    
    def try_multi_swaps(removed, face_counts):
        """Try 2-for-1 or 2-for-0 swaps"""
        improved = False
        
        if len(removed) < 2:
            return improved
            
        attempts = 0
        max_attempts = 100
        
        while attempts < max_attempts and time.time() - start_time < deadline:
            e1, e2 = random.sample(list(removed), 2)
            attempts += 1
            
            # Determine required faces for e_in
            required = set()
            valid = True
            
            all_faces = edge_face_sets[e1] | edge_face_sets[e2]
            
            for f in all_faces:
                cnt = face_counts[f]
                cov_e1 = 1 if e1 in face_edge_sets[f] else 0
                cov_e2 = 1 if e2 in face_edge_sets[f] else 0
                removals = cov_e1 + cov_e2
                
                if cnt - removals < 1:
                    if cnt - removals < 0:
                        valid = False
                        break
                    else:
                        required.add(f)
                        
            if not valid:
                continue
                
            if not required:
                # Can just remove both
                removed.discard(e1)
                removed.discard(e2)
                for f in edge_faces[e1]:
                    face_counts[f] -= 1
                for f in edge_faces[e2]:
                    face_counts[f] -= 1
                improved = True
                continue
            
            # Find e_in that covers all required faces
            candidates = face_edge_sets[list(required)[0]].copy()
            for f in list(required)[1:]:
                candidates &= face_edge_sets[f]
                
            candidates.discard(e1)
            candidates.discard(e2)
            
            valid_candidates = [e for e in candidates if e not in removed]
            
            if valid_candidates:
                e_in = random.choice(valid_candidates)
                removed.discard(e1)
                removed.discard(e2)
                removed.add(e_in)
                for f in edge_faces[e1]:
                    face_counts[f] -= 1
                for f in edge_faces[e2]:
                    face_counts[f] -= 1
                for f in edge_faces[e_in]:
                    face_counts[f] += 1
                improved = True
                
        return improved
    
    def local_search(removed, face_counts, time_limit):
        """Run local search until no improvement or time limit"""
        while time.time() - start_time < time_limit:
            improved = False
            
            # 1. Deletions
            if try_deletions(removed, face_counts):
                improved = True
                continue
            
            # 2. Swaps
            if try_swaps(removed, face_counts):
                improved = True
                continue
            
            # 3. Multi-Swaps
            if try_multi_swaps(removed, face_counts):
                improved = True
                continue
            
            # No improvement - try perturbation
            if time.time() - start_time < time_limit - 5:
                # Add a random edge, then run deletions
                non_removed = [e for e in range(n_edges) if e not in removed]
                if non_removed:
                    e_perturb = random.choice(non_removed)
                    removed.add(e_perturb)
                    for f in edge_faces[e_perturb]:
                        face_counts[f] += 1
                    # Run deletions
                    try_deletions(removed, face_counts)
                    
            else:
                break
                
        return removed, face_counts
    
    # Main optimization loop with multiple restarts
    best_removed = None
    best_size = float('inf')
    
    # First greedy init
    removed, face_counts = greedy_init()
    removed, face_counts = local_search(removed, face_counts, deadline - 10)
    
    if len(removed) < best_size:
        best_size = len(removed)
        best_removed = set(removed)
    
    # Try multiple restarts with different random seeds
    restarts = 0
    max_restarts = 50
    
    while time.time() - start_time < deadline - 20 and restarts < max_restarts:
        restarts += 1
        
        # New greedy init
        removed, face_counts = greedy_init()
        
        # Local search
        removed, face_counts = local_search(removed, face_counts, deadline - 10)
        
        if len(removed) < best_size:
            best_size = len(removed)
            best_removed = set(removed)
    
    # Final polish on best solution
    face_counts_final = np.zeros(n_faces, dtype=np.int32)
    for e in best_removed:
        for f in edge_faces[e]:
            face_counts_final[f] += 1
            
    # More aggressive deletion pass
    changed = True
    while changed:
        changed = False
        for e in list(best_removed):
            critical = False
            for f in edge_faces[e]:
                if face_counts_final[f] == 1:
                    critical = True
                    break
            if not critical:
                best_removed.remove(e)
                for f in edge_faces[e]:
                    face_counts_final[f] -= 1
                changed = True
    
    # Try swaps on the final solution
    for _ in range(100):
        if time.time() - start_time > deadline - 5:
            break
        if not try_swaps(best_removed, face_counts_final):
            break
        # After a swap, try deletions again
        changed = True
        while changed:
            changed = False
            for e in list(best_removed):
                critical = False
                for f in edge_faces[e]:
                    if face_counts_final[f] == 1:
                        critical = True
                        break
                if not critical:
                    best_removed.remove(e)
                    for f in edge_faces[e]:
                        face_counts_final[f] -= 1
                    changed = True
    
    result = []
    for i in range(n_edges):
        if i not in best_removed:
            result.append(list(edges[i]))
            
    return result
