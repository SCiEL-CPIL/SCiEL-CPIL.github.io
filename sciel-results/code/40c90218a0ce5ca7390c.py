import random
import time
import numpy as np

def construct():
    d = 7
    n_vertices = 2 ** d
    total_edges = n_vertices * d // 2  # 448
    
    # Precompute all edges
    all_edges = []
    edge_to_index = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                e = (x, y)
                edge_to_index[e] = len(all_edges)
                all_edges.append(e)
    
    # Precompute faces as edge indices
    # A 2-face is defined by vertex u and two distinct dimensions i, j
    # The 4 edges of the face are:
    # (u, u^bit_i), (u^bit_i, u^bit_i^bit_j), (u^bit_i^bit_j, u^bit_j), (u^bit_j, u)
    faces_idx = []
    edge_faces_idx = [[] for _ in range(total_edges)]
    
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v = u ^ (1 << i)
                w = u ^ (1 << j)
                z = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(u, v), max(u, v))
                e2 = (min(v, z), max(v, z))
                e3 = (min(z, w), max(z, w))
                e4 = (min(w, u), max(w, u))
                
                idx1 = edge_to_index[e1]
                idx2 = edge_to_index[e2]
                idx3 = edge_to_index[e3]
                idx4 = edge_to_index[e4]
                
                face = (idx1, idx2, idx3, idx4)
                faces_idx.append(face)
                
                for idx in face:
                    edge_faces_idx[idx].append(len(faces_idx) - 1)
    
    n_faces = len(faces_idx)
    
    # Convert faces to numpy array for faster operations
    faces_arr = np.array(faces_idx, dtype=np.int32)
    
    def is_valid_addition(chosen_arr, e_idx):
        """
        Check if adding edge e_idx to chosen set is valid.
        chosen_arr is a numpy boolean array of length total_edges.
        """
        # Get all faces containing e_idx
        for f_idx in edge_faces_idx[e_idx]:
            face = faces_arr[f_idx]
            # Count how many of the other 3 edges are in chosen
            count = 0
            for idx in face:
                if idx != e_idx and chosen_arr[idx]:
                    count += 1
                    if count == 3:
                        return False
    return True  # This line is unreachable if the loop completes

    def greedy_random(seed=None):
        if seed is not None:
            random.seed(seed)
        chosen = np.zeros(total_edges, dtype=bool)
        indices = list(range(total_edges))
        random.shuffle(indices)
        for e_idx in indices:
            if is_valid_addition(chosen, e_idx):
                chosen[e_idx] = True
        return chosen

    def local_search(chosen, time_limit):
        """
        Local search with 2-for-1 swaps and perturbations.
        chosen is a numpy boolean array.
        """
        start_time = time.time()
        chosen = chosen.copy()
        
        def get_missing():
            return [i for i in range(total_edges) if not chosen[i]]
        
        def compute_unblockers():
            """
            For each missing edge, find which single edge removal would unblock it.
            Returns dict: missing_edge -> set of edges that block it (single unblockers)
            and dict: chosen_edge -> list of missing edges it blocks
            """
            missing = get_missing()
            single_unblockers = {}
            unblocks = {b: [] for b in range(total_edges) if chosen[b]}
            
            for e in missing:
                bad_faces = []
                for f_idx in edge_faces_idx[e]:
                    face = faces_arr[f_idx]
                    others_in_s = [idx for idx in face if idx != e and chosen[idx]]
                    if len(others_in_s) == 3:
                        bad_faces.append(others_in_s)
                
                if not bad_faces:
                    # Can add directly
                    chosen[e] = True
                    continue
                
                if len(bad_faces) == 1:
                    single_unblockers[e] = set(bad_faces[0])
                else:
                    common = set(bad_faces[0])
                    for bf in bad_faces[1:]:
                        common &= set(bf)
                    single_unblockers[e] = common if common else set()
                
                for b in single_unblockers[e]:
                    if b in unblocks:
                        unblocks[b].append(e)
            
            return unblocks
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            unblocks = compute_unblockers()
            
            # Try 2-for-1 swaps
            b_list = [b for b in range(total_edges) if chosen[b]]
            random.shuffle(b_list)
            
            for b in b_list:
                if time.time() - start_time > time_limit:
                    break
                if not chosen[b]:
                    continue
                candidates = unblocks.get(b, [])
                if len(candidates) >= 2:
                    # Try to find two edges that can be added after removing b
                    random.shuffle(candidates)
                    for i in range(len(candidates)):
                        for j in range(i + 1, len(candidates)):
                            e1, e2 = candidates[i], candidates[j]
                            # Temporarily remove b
                            chosen[b] = False
                            # Check if e1 can be added
                            if is_valid_addition(chosen, e1):
                                chosen[e1] = True
                                # Check if e2 can be added
                                if is_valid_addition(chosen, e2):
                                    chosen[e2] = True
                                    improved = True
                                else:
                                    chosen[e1] = False
                            chosen[b] = True
                            if improved:
                                break
                        if improved:
                            break
                if improved:
                    break
            
            if not improved:
                # Try perturbation: remove a random edge and repair
                if random.random() < 0.15 and len(chosen) > 100:
                    b_to_remove = random.choice([b for b in range(total_edges) if chosen[b]])
                    chosen[b_to_remove] = False
                    # Greedy repair
                    missing = get_missing()
                    random.shuffle(missing)
                    for e in missing:
                        if is_valid_addition(chosen, e):
                            chosen[e] = True
                            improved = True
                    if time.time() - start_time > time_limit:
                        break
        
        return chosen

    best_set = None
    best_size = 0
    
    start = time.time()
    deadline = start + 140  # 10s safety margin
    
    # Phase 1: Multiple greedy restarts
    num_restarts = 2000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        s = greedy_random()
        size = s.sum()
        if size > best_size:
            best_size = size
            best_set = s.copy()
    
    # Phase 2: Local search on the best set
    if time.time() < deadline - 10 and best_set is not None:
        improved_set = local_search(best_set, deadline - time.time() - 5)
        if improved_set.sum() > best_size:
            best_set = improved_set
            best_size = best_set.sum()
    
    # Phase 3: More local search iterations if time remains
    if time.time() < deadline - 5 and best_set is not None:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            improved_set = local_search(best_set, min(20, deadline - time.time() - 5))
            if improved_set.sum() > best_size:
                best_set = improved_set
                best_size = best_set.sum()
    
    # Convert to list of [u, v]
    result = []
    for e_idx in range(total_edges):
        if best_set is not None and best_set[e_idx]:
            u, v = all_edges[e_idx]
            result.append([u, v])
    
    return result
