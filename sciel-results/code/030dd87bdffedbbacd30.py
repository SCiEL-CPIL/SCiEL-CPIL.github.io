import random
import time

def construct():
    d = 7
    n_vertices = 2 ** d
    total_edges = n_vertices * d // 2
    
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
                
                idxs = (edge_to_index[e1], edge_to_index[e2], edge_to_index[e3], edge_to_index[e4])
                faces_idx.append(idxs)
                
                for idx in idxs:
                    edge_faces_idx[idx].append(len(faces_idx) - 1)
                    
    # Convert faces to a flat list of tuples for faster iteration? 
    # Or keep as list of tuples.
    
    def is_valid_addition(chosen_mask, e_idx):
        """
        Check if adding edge e_idx to chosen set is valid.
        chosen_mask is a bytearray or similar where chosen_mask[i] is 1 if edge i is chosen.
        """
        # Check all faces containing e
        for f_idx in edge_faces_idx[e_idx]:
            face = faces_idx[f_idx]
            # Check if other 3 edges are in chosen
            # We know e_idx is NOT in chosen yet (we are checking addition)
            count = 0
            for idx in face:
                if idx != e_idx and chosen_mask[idx]:
                    count += 1
                    if count == 3:
                        return False
        return True

    def greedy_random():
        chosen_mask = bytearray(total_edges)
        indices = list(range(total_edges))
        random.shuffle(indices)
        for e_idx in indices:
            if is_valid_addition(chosen_mask, e_idx):
                chosen_mask[e_idx] = 1
        return chosen_mask

    def local_search(chosen_mask, time_limit):
        """
        Try to improve chosen set using 2-for-1 swaps and 1-for-1 swaps.
        """
        start_time = time.time()
        chosen_count = sum(chosen_mask)
        
        # Helper to get list of chosen/missing edges
        def get_missing():
            return [i for i in range(total_edges) if not chosen_mask[i]]
        
        def get_chosen():
            return [i for i in range(total_edges) if chosen_mask[i]]
            
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            missing = get_missing()
            chosen_edges = get_chosen()
            
            if not missing:
                break
                
            # Calculate single unblockers for missing edges
            # unblockers[e] = set of b in chosen such that removing b allows e to be added
            # We only care if |unblockers[e]| >= 1.
            # For 2-for-1, we need b to unblock e1 AND e2.
            
            # Optimization: Instead of sets, we can just count how many missing edges each b unblocks.
            # But we need to know WHICH edges to add.
            
            # Map: b -> list of e that b unblocks
            unblocks = {b: [] for b in chosen_edges}
            
            for e in missing:
                bad_faces = []
                for f_idx in edge_faces_idx[e]:
                    face = faces_idx[f_idx]
                    others_in_s = []
                    for idx in face:
                        if idx != e and chosen_mask[idx]:
                            others_in_s.append(idx)
                    if len(others_in_s) == 3:
                        bad_faces.append(others_in_s)
                        
                if not bad_faces:
                    # Shouldn't happen if maximal, but handle
                    chosen_mask[e] = 1
                    chosen_count += 1
                    improved = True
                    continue
                
                # Find intersection of bad_faces
                if len(bad_faces) == 1:
                    blockers = bad_faces[0]
                else:
                    common = set(bad_faces[0])
                    for bf in bad_faces[1:]:
                        common &= set(bf)
                    if not common:
                        continue # No single unblocker
                    blockers = list(common)
                
                for b in blockers:
                    if b in unblocks:
                        unblocks[b].append(e)
                        
            # Look for b that unblocks >= 2 edges
            # Shuffle chosen edges to avoid bias
            random.shuffle(chosen_edges)
            
            for b in chosen_edges:
                if time.time() - start_time > time_limit:
                    break
                if not chosen_mask[b]:
                    continue
                candidates = unblocks.get(b, [])
                if len(candidates) >= 2:
                    # Try to add two
                    # Pick 2 random candidates
                    e1, e2 = random.sample(candidates, 2)
                    
                    # Check validity
                    # Temporarily remove b
                    chosen_mask[b] = 0
                    
                    valid = True
                    if not is_valid_addition(chosen_mask, e1):
                        valid = False
                    else:
                        chosen_mask[e1] = 1
                        if not is_valid_addition(chosen_mask, e2):
                            valid = False
                        else:
                            chosen_mask[e2] = 1
                    
                    if valid:
                        chosen_count += 1
                        improved = True
                        break # Restart loop with new state
                    else:
                        # Revert
                        if chosen_mask[e1]:
                            chosen_mask[e1] = 0
                        chosen_mask[b] = 1
            
            if not improved:
                # Try 1-for-1 swaps to escape local optima
                # Pick random b from chosen, random e from missing
                # If removing b allows adding e, do it.
                if chosen_edges and missing:
                    b = random.choice(chosen_edges)
                    e = random.choice(missing)
                    
                    chosen_mask[b] = 0
                    if is_valid_addition(chosen_mask, e):
                        chosen_mask[e] = 1
                        improved = True # Trigger recalculation, size same
                    else:
                        chosen_mask[b] = 1
                        # Perturbation: Remove b and repair?
                        if random.random() < 0.05:
                             chosen_mask[b] = 0
                             chosen_count -= 1
                             # Greedy repair
                             for e2 in missing:
                                 if e2 == b: continue
                                 if not chosen_mask[e2] and is_valid_addition(chosen_mask, e2):
                                     chosen_mask[e2] = 1
                                     chosen_count += 1
                             improved = True
                             
        return chosen_mask, chosen_count

    best_mask = None
    best_size = 0
    
    start = time.time()
    deadline = start + 115 # 5s safety margin
    
    # Phase 1: Random Greedy Restarts
    # Estimate time for one greedy: ~448 edges * checks. Should be fast.
    # Let's run as many as possible.
    while time.time() < deadline:
        mask = greedy_random()
        size = sum(mask)
        if size > best_size:
            best_size = size
            best_mask = mask.copy()
            
    # Phase 2: Local Search
    if best_mask is not None and time.time() < deadline - 2:
        improved_mask, improved_size = local_search(best_mask, deadline - time.time() - 1)
        if improved_size > best_size:
            best_size = improved_size
            best_mask = improved_mask

    # Convert to list of [u, v]
    result = []
    for e_idx in range(total_edges):
        if best_mask[e_idx]:
            u, v = all_edges[e_idx]
            result.append([u, v])
        
    return result
