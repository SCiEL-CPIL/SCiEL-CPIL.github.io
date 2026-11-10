import random
import time
import numpy as np

def construct():
    deadline = time.time() + 110
    d = 7
    n = 1 << d
    
    # Build all edges
    all_edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v, i))
    
    num_edges = len(all_edges)
    
    edge_tuple_to_idx = {}
    for idx, (u, v, i) in enumerate(all_edges):
        edge_tuple_to_idx[(u, v)] = idx
    
    # Build faces
    faces = []
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
    
    num_faces = len(faces)
    
    edge_to_faces = [[] for _ in range(num_edges)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_to_faces[e_idx].append(f_idx)
    
    # Precompute face membership as sets for fast "in" checks
    face_sets = [set(face) for face in faces]
    
    def greedy_random(seed):
        rng = random.Random(seed)
        chosen = [False] * num_edges
        count = 0
        face_counts = [0] * num_faces
        
        indices = list(range(num_edges))
        rng.shuffle(indices)
        
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
    
    def greedy_degree_based(seed):
        """Order edges by how many faces they're in that are still 'open'"""
        rng = random.Random(seed)
        chosen = [False] * num_edges
        count = 0
        face_counts = [0] * num_faces
        
        remaining = list(range(num_edges))
        rng.shuffle(remaining)
        
        while remaining and time.time() < deadline - 5:
            best_idx = -1
            best_score = float('inf')
            sample_size = min(len(remaining), 50)
            sample = rng.sample(remaining, sample_size) if len(remaining) > sample_size else remaining
            
            for idx in sample:
                score = 0
                for f_idx in edge_to_faces[idx]:
                    if face_counts[f_idx] == 2:
                        score += 1
                    elif face_counts[f_idx] == 3:
                        score += 100
                if score < best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx < 0:
                break
            
            can_add = True
            for f_idx in edge_to_faces[best_idx]:
                if face_counts[f_idx] == 3:
                    can_add = False
                    break
            
            if can_add:
                chosen[best_idx] = True
                count += 1
                for f_idx in edge_to_faces[best_idx]:
                    face_counts[f_idx] += 1
            
            remaining.remove(best_idx)
        
        return count, chosen, face_counts
    
    def local_search(chosen, face_counts, count, deadline):
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            for r_idx in range(num_edges):
                if not chosen[r_idx]:
                    continue
                
                chosen[r_idx] = False
                count -= 1
                for f_idx in edge_to_faces[r_idx]:
                    face_counts[f_idx] -= 1
                
                candidate_set = set()
                for f_idx in edge_to_faces[r_idx]:
                    for e_idx in faces[f_idx]:
                        if e_idx != r_idx and not chosen[e_idx]:
                            candidate_set.add(e_idx)
                
                candidates = []
                for e_idx in candidate_set:
                    valid = True
                    for f_idx in edge_to_faces[e_idx]:
                        if face_counts[f_idx] == 3:
                            valid = False
                            break
                    if valid:
                        candidates.append(e_idx)
                
                added = []
                for i in range(len(candidates)):
                    e1 = candidates[i]
                    for j in range(i + 1, len(candidates)):
                        e2 = candidates[j]
                        compatible = True
                        for f_idx in edge_to_faces[e1]:
                            if e2 in face_sets[f_idx]:
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
                    for e_idx in added:
                        chosen[e_idx] = True
                        count += 1
                        for f_idx in edge_to_faces[e_idx]:
                            face_counts[f_idx] += 1
                    improved = True
                    break
                
                if len(candidates) > 0:
                    best_e = candidates[0]
                    chosen[best_e] = True
                    count += 1
                    for f_idx in edge_to_faces[best_e]:
                        face_counts[f_idx] += 1
                    improved = True
                    break
                
                chosen[r_idx] = True
                count += 1
                for f_idx in edge_to_faces[r_idx]:
                    face_counts[f_idx] += 1
        
        for e_idx in range(num_edges):
            if not chosen[e_idx] and time.time() < deadline - 5:
                valid = True
                for f_idx in edge_to_faces[e_idx]:
                    if face_counts[f_idx] == 3:
                        valid = False
                        break
                if valid:
                    chosen[e_idx] = True
                    count += 1
                    for f_idx in edge_to_faces[e_idx]:
                        face_counts[f_idx] += 1
        
        return count, chosen, face_counts
    
    def simulated_annealing(seed, start_count, start_chosen, start_face_counts):
        """Simulated annealing with remove-1-add-2 moves"""
        rng = random.Random(seed)
        chosen = start_chosen[:]
        face_counts = start_face_counts[:]
        count = start_count
        
        T = 1.0
        cooling = 0.999
        
        while time.time() < deadline - 15 and T > 0.01:
            # Pick a random chosen edge to remove
            chosen_indices = [i for i in range(num_edges) if chosen[i]]
            if not chosen_indices:
                break
            
            r_idx = rng.choice(chosen_indices)
            
            # Remove it
            chosen[r_idx] = False
            count -= 1
            for f_idx in edge_to_faces[r_idx]:
                face_counts[f_idx] -= 1
            
            # Find candidates
            candidate_set = set()
            for f_idx in edge_to_faces[r_idx]:
                for e_idx in faces[f_idx]:
                    if e_idx != r_idx and not chosen[e_idx]:
                        candidate_set.add(e_idx)
            
            candidates = []
            for e_idx in candidate_set:
                valid = True
                for f_idx in edge_to_faces[e_idx]:
                    if face_counts[f_idx] == 3:
                        valid = False
                        break
                if valid:
                    candidates.append(e_idx)
            
            # Try to add two compatible edges
            added = []
            if len(candidates) >= 2:
                for _ in range(min(5, len(candidates))):
                    e1, e2 = rng.sample(candidates, 2)
                    compatible = True
                    for f_idx in edge_to_faces[e1]:
                        if e2 in face_sets[f_idx]:
                            if face_counts[f_idx] >= 2:
                                compatible = False
                                break
                    if compatible:
                        added = [e1, e2]
                        break
            
            if len(added) >= 2:
                for e_idx in added:
                    chosen[e_idx] = True
                    count += 1
                    for f_idx in edge_to_faces[e_idx]:
                        face_counts[f_idx] += 1
                # Always accept improvement
                improved = True
            else:
                # Restore
                chosen[r_idx] = True
                count += 1
                for f_idx in edge_to_faces[r_idx]:
                    face_counts[f_idx] += 1
                # Try single add with probability T
                if candidates and rng.random() < T:
                    e_idx = rng.choice(candidates)
                    chosen[r_idx] = False
                    count -= 1
                    for f_idx in edge_to_faces[r_idx]:
                        face_counts[f_idx] -= 1
                    chosen[e_idx] = True
                    count += 1
                    for f_idx in edge_to_faces[e_idx]:
                        face_counts[f_idx] += 1
            
            T *= cooling
        
        return count, chosen, face_counts
    
    def kick_and_optimize(seed):
        """Perturb a good solution and re-optimize"""
        rng = random.Random(seed)
        # Start with a greedy solution
        count, chosen, face_counts = greedy_random(seed)
        
        # Kick: randomly remove some edges
        kick_size = rng.randint(10, 30)
        chosen_indices = [i for i in range(num_edges) if chosen[i]]
        to_remove = rng.sample(chosen_indices, min(kick_size, len(chosen_indices)))
        for idx in to_remove:
            chosen[idx] = False
            count -= 1
            for f_idx in edge_to_faces[idx]:
                face_counts[f_idx] -= 1
        
        # Re-optimize with local search
        count, chosen, face_counts = local_search(chosen, face_counts, count, deadline)
        
        # Simulated annealing
        count, chosen, face_counts = simulated_annealing(seed + 100000, count, chosen, face_counts)
        
        return count, chosen, face_counts
    
    best_count = 0
    best_chosen = None
    
    # First, try several greedy approaches
    for seed in range(20):
        if time.time() > deadline - 20:
            break
        if seed % 2 == 0:
            count, chosen, face_counts = greedy_random(seed)
        else:
            count, chosen, face_counts = greedy_degree_based(seed + 1000)
        
        count, chosen, face_counts = local_search(chosen, face_counts, count, deadline)
        
        if count > best_count:
            best_count = count
            best_chosen = chosen[:]
    
    # Now try kick-and-optimize from the best solution
    for seed in range(10):
        if time.time() > deadline - 10:
            break
        count, chosen, face_counts = kick_and_optimize(seed + 50000)
        
        if count > best_count:
            best_count = count
            best_chosen = chosen[:]
    
    # Final: try simulated annealing from best
    if best_chosen is not None and time.time() < deadline - 5:
        # Reconstruct face_counts
        face_counts = [0] * num_faces
        for idx in range(num_edges):
            if best_chosen[idx]:
                for f_idx in edge_to_faces[idx]:
                    face_counts[f_idx] += 1
        
        count, chosen, face_counts = simulated_annealing(99999, best_count, best_chosen, face_counts)
        
        if count > best_count:
            best_count = count
            best_chosen = chosen[:]
    
    if best_chosen is None:
        return []
    
    out = []
    for idx in range(num_edges):
        if best_chosen[idx]:
            u, v, i = all_edges[idx]
            out.append([u, v])
    return out
