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

    # Precompute faces
    faces = []
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

                face_edges = [e1, e2, e3, e4]
                faces.append(face_edges)

    # Optimization: Store faces as tuples of edge indices
    faces_idx = []
    for f in faces:
        idxs = tuple(edge_to_index[e] for e in f)
        faces_idx.append(idxs)

    # Map edge index to list of face indices
    edge_to_faces_idx = [[] for _ in range(total_edges)]
    for f_idx, f in enumerate(faces_idx):
        for e_idx in f:
            edge_to_faces_idx[e_idx].append(f_idx)

    def is_valid_addition(chosen, e_idx):
        """
        Check if adding edge e_idx to chosen set is valid.
        chosen is a set of edge indices.
        """
        # Check all faces containing e_idx
        for f_idx in edge_to_faces_idx[e_idx]:
            face = faces_idx[f_idx]
            # Check if other 3 edges are in chosen
            count = 0
            for idx in face:
                if idx == e_idx:
                    continue
                if idx in chosen:
                    count += 1
                if count == 3:
                    return False
        return True

    def greedy_random():
        chosen = set()
        indices = list(range(total_edges))
        random.shuffle(indices)
        for e_idx in indices:
            if is_valid_addition(chosen, e_idx):
                chosen.add(e_idx)
        return chosen

    def local_search(chosen, time_limit):
        """
        Improved local search with 1-for-1, 2-for-1 swaps and perturbations.
        """
        start_time = time.time()
        current_size = len(chosen)
        
        # Helper to try adding any valid missing edge
        def try_add_all():
            nonlocal current_size
            added = False
            for e_idx in range(total_edges):
                if e_idx not in chosen and is_valid_addition(chosen, e_idx):
                    chosen.add(e_idx)
                    current_size += 1
                    added = True
            return added

        # Initial repair
        try_add_all()

        while time.time() - start_time < time_limit:
            improved = False
            
            # 1. Try to add any missing edge (should be few if we call try_add_all often)
            if try_add_all():
                improved = True
                continue

            # 2. Try 1-for-1 swaps
            # We want to find b in S, e not in S such that S - {b} + {e} is valid and |S| stays same? 
            # No, 1-for-1 keeps size. It helps escape local optima for 2-for-1.
            # Actually, let's look for 2-for-1 first as it increases size.
            
            # 3. Try 2-for-1 swaps
            # For each b in S, check if removing it allows adding two edges.
            # We can precompute "unblockers" or just iterate.
            # To be fast, we'll sample a subset of edges to try if we are tight on time, 
            # but usually iterating all is fine if is_valid_addition is fast.
            
            # Optimization: Identify "critical" edges for missing edges.
            # If e is missing, let BadFaces(e) be faces where 3 edges are in S.
            # If we remove b, e can be added if b is in the intersection of "others" for all BadFaces(e).
            
            single_unblockers = {}
            missing_edges = [i for i in range(total_edges) if i not in chosen]
            
            for e in missing_edges:
                bad_faces = []
                for f_idx in edge_to_faces_idx[e]:
                    face = faces_idx[f_idx]
                    others_in_s = [idx for idx in face if idx != e and idx in chosen]
                    if len(others_in_s) == 3:
                        bad_faces.append(others_in_s)
                
                if not bad_faces:
                    # Should have been added in try_add_all
                    continue
                
                if len(bad_faces) == 1:
                    single_unblockers[e] = set(bad_faces[0])
                else:
                    common = set(bad_faces[0])
                    for bf in bad_faces[1:]:
                        common &= set(bf)
                    if common:
                        single_unblockers[e] = common
                    else:
                        single_unblockers[e] = set()

            # Map b -> list of e's it unblocks
            unblocks = {b: [] for b in chosen}
            for e, blockers in single_unblockers.items():
                for b in blockers:
                    if b in unblocks:
                        unblocks[b].append(e)

            # Try to find a b that unblocks >= 2 edges
            found_swap = False
            # Shuffle to randomize order
            b_list = list(chosen)
            random.shuffle(b_list)
            
            for b in b_list:
                if time.time() - start_time > time_limit:
                    break
                if b not in chosen:
                    continue
                candidates = unblocks.get(b, [])
                if len(candidates) >= 2:
                    # Try random pairs from candidates
                    # Limit checks to avoid slowdown
                    pairs_to_try = random.sample(candidates, min(5, len(candidates)))
                    # Need 2 distinct edges
                    if len(pairs_to_try) < 2:
                        continue
                    for i in range(len(pairs_to_try)):
                        for j in range(i + 1, len(pairs_to_try)):
                            e1, e2 = pairs_to_try[i], pairs_to_try[j]
                            # Check validity
                            new_chosen = set(chosen)
                            new_chosen.discard(b)
                            if is_valid_addition(new_chosen, e1):
                                new_chosen.add(e1)
                                if is_valid_addition(new_chosen, e2):
                                    new_chosen.add(e2)
                                    # Accept
                                    chosen = new_chosen
                                    current_size += 1
                                    improved = True
                                    found_swap = True
                                    break
                        if found_swap:
                            break
                if found_swap:
                    break
            
            if found_swap:
                # Recalculate state for next iteration
                try_add_all()
                continue

            # 4. If no 2-for-1 found, try 1-for-1 swaps or perturbation
            # 1-for-1: Remove b, add e. Size unchanged.
            # We look for b in S, e not in S such that S - {b} + {e} is valid.
            # This is equivalent to: e is unblocked by b (single unblocker) AND no other face of e is blocked by S-{b}.
            # Actually, if b is a single unblocker, it means for all bad faces of e, b is in the others.
            # So removing b fixes all bad faces.
            # We just need to ensure that adding e doesn't create a NEW bad face (which is impossible as e was not in S).
            # Wait, adding e creates faces. The condition is that no face of e has 3 other edges in S-{b}.
            # Since b was the only blocker for all bad faces, and e was not in S, 
            # the other edges in any face of e are either in S or not.
            # If a face F of e had 3 edges in S-{b}, then those 3 edges are in S.
            # Then F would have been a bad face for e (since e not in S, 3 others in S).
            # And b must be one of those 3? No, b is NOT in F if b is not in F.
            # If b is not in F, then b being a single unblocker implies b is in ALL bad faces.
            # So if F is a face of e with 3 edges in S-{b}, then F has 3 edges in S.
            # So F is a bad face.
            # So b must be in F.
            # But b is removed, so F has 3 edges in S-{b}?
            # Let's re-verify.
            # If b is in single_unblockers[e], then for every bad face F of e, b is in F \ {e}.
            # So if we remove b, F \ {e} \ {b} has 2 edges.
            # Are there other faces of e?
            # If there is a face G of e that is NOT a bad face (i.e., < 3 edges in S), 
            # then after removing b, it has <= 3 edges in S-{b}.
            # If it had 3 edges in S, and b not in G, then it has 3 edges in S-{b}.
            # Then G would be a bad face for e in S.
            # But we assumed b is in ALL bad faces.
            # So if G is a bad face, b is in G.
            # So if G has 3 edges in S, b is one of them.
            # So G \ {e} \ {b} has 2 edges.
            # Thus, removing b makes all faces of e have <= 2 edges in S-{b}.
            # So adding e is valid.
            
            # So, any single unblocker b allows adding e.
            # We can do 1-for-1: remove b, add e.
            # This doesn't increase size, but changes the configuration.
            
            if random.random() < 0.5:
                # Try 1-for-1
                # Pick a random e from missing, find its unblockers, pick a random b.
                if missing_edges:
                    e = random.choice(missing_edges)
                    blockers = single_unblockers.get(e, set())
                    if blockers:
                        b = random.choice(list(blockers))
                        # Perform swap
                        chosen.discard(b)
                        chosen.add(e)
                        improved = True
                        # try_add_all() might not be needed immediately, but good to check
                        # Actually, size is same.
            
            if not improved:
                # Perturbation: Kick
                # Remove a random edge, then greedy repair
                if chosen:
                    b_to_remove = random.choice(list(chosen))
                    chosen.discard(b_to_remove)
                    current_size -= 1
                    try_add_all()

        return chosen

    best_set = set()
    best_size = 0
    
    start = time.time()
    deadline = start + 110 # 10s safety margin
    
    # Phase 1: Random Greedy Restarts
    # Run many restarts to find a good starting point
    # Each greedy run is fast (~ms)
    while time.time() < deadline - 10: # Leave 10s for local search
        s = greedy_random()
        if len(s) > best_size:
            best_size = len(s)
            best_set = s.copy()
            
    # Phase 2: Local Search on the best set
    if time.time() < deadline - 2: # Leave 2s for final check
        improved_set = local_search(best_set, deadline - time.time() - 2)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
            
    # Convert to list of [u, v]
    result = []
    for e_idx in best_set:
        u, v = all_edges[e_idx]
        result.append([u, v])
        
    return result
