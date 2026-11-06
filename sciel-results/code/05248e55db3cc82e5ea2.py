import random
import time
import numpy as np

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
    faces_idx = []
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
                idxs = tuple(edge_to_index[e] for e in face_edges)
                faces_idx.append(idxs)

    # Map edge index to list of face indices
    edge_to_faces = [[] for _ in range(total_edges)]
    for f_idx, f in enumerate(faces_idx):
        for e_idx in f:
            edge_to_faces[e_idx].append(f_idx)

    # Precompute for each edge, the list of (face_idx, other_edge_indices) for fast validity checks
    edge_face_others = [[] for _ in range(total_edges)]
    for f_idx, f in enumerate(faces_idx):
        for e_idx in f:
            others = [idx for idx in f if idx != e_idx]
            edge_face_others[e_idx].append((f_idx, tuple(others)))

    # Precompute face edge sets for quick counting
    face_edges_set = [set(f) for f in faces_idx]

    def count_face_edges(f_idx, chosen_set):
        """Count how many edges of face f_idx are in chosen_set"""
        return sum(1 for e in faces_idx[f_idx] if e in chosen_set)

    def is_valid_addition(chosen, e_idx):
        """
        Check if adding edge e_idx to chosen set is valid.
        chosen is a set of edge indices.
        """
        for f_idx, others in edge_face_others[e_idx]:
            count = 0
            for idx in others:
                if idx in chosen:
                    count += 1
                    if count == 3:
                        return False
        return True

    def greedy_random(seed=None):
        rng = random.Random(seed)
        chosen = set()
        indices = list(range(total_edges))
        rng.shuffle(indices)
        for e_idx in indices:
            if is_valid_addition(chosen, e_idx):
                chosen.add(e_idx)
        return chosen

    def local_search(chosen, time_limit, seed=None):
        """
        Local search with 1-for-1, 2-for-1 swaps and perturbations.
        """
        rng = random.Random(seed)
        start_time = time.time()
        current_size = len(chosen)
        
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
            
            # 1. Try to add any missing edge
            if try_add_all():
                improved = True
                continue

            # 2. Try 2-for-1 swaps
            missing_edges = [i for i in range(total_edges) if i not in chosen]
            
            # For each missing edge, find blockers (single edges whose removal would allow adding it)
            single_unblockers = {}
            for e in missing_edges:
                bad_faces = []
                for f_idx, others in edge_face_others[e]:
                    count_in_s = 0
                    for idx in others:
                        if idx in chosen:
                            count_in_s += 1
                    if count_in_s == 3:
                        bad_faces.append(others)
                
                if not bad_faces:
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
            b_list = list(chosen)
            rng.shuffle(b_list)
            
            for b in b_list:
                if time.time() - start_time > time_limit:
                    break
                if b not in chosen:
                    continue
                candidates = unblocks.get(b, [])
                if len(candidates) >= 2:
                    candidates = candidates[:15]
                    for i in range(len(candidates)):
                        for j in range(i + 1, len(candidates)):
                            e1, e2 = candidates[i], candidates[j]
                            chosen.discard(b)
                            valid1 = is_valid_addition(chosen, e1)
                            if valid1:
                                chosen.add(e1)
                                valid2 = is_valid_addition(chosen, e2)
                                if valid2:
                                    chosen.add(e2)
                                    current_size += 1
                                    improved = True
                                    found_swap = True
                                    break
                                else:
                                    chosen.discard(e1)
                            chosen.add(b)
                            if found_swap:
                                break
                        if found_swap:
                            break
                if found_swap:
                    break
            
            if found_swap:
                try_add_all()
                continue

            # 3. Try 1-for-1 swaps
            if missing_edges:
                e = rng.choice(missing_edges)
                blockers = single_unblockers.get(e, set())
                if blockers:
                    b = rng.choice(list(blockers))
                    chosen.discard(b)
                    chosen.add(e)
                    improved = True

            if not improved:
                # Perturbation: remove random edge, greedy repair
                if chosen:
                    b_to_remove = rng.choice(list(chosen))
                    chosen.discard(b_to_remove)
                    current_size -= 1
                    try_add_all()

        # Final greedy fill
        try_add_all()
        return chosen

    def hill_climb_with_restarts(num_restarts, time_per_ls, seed_base=0):
        """
        Run multiple random restarts with local search on each.
        """
        best_set = set()
        best_size = 0
        start = time.time()
        deadline = start + 110

        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            s = greedy_random(seed=seed_base + r)
            if len(s) > best_size:
                best_size = len(s)
                best_set = s.copy()
            
            # Run local search on this solution
            remaining = deadline - time.time() - 3
            if remaining > 2:
                ls_time = min(time_per_ls, remaining)
                improved_set = local_search(best_set.copy(), ls_time, seed=seed_base + r + 1000)
                if len(improved_set) > best_size:
                    best_size = len(improved_set)
                    best_set = improved_set

        # Final local search on best
        remaining = deadline - time.time() - 2
        if remaining > 1:
            improved_set = local_search(best_set.copy(), remaining, seed=seed_base + 5000)
            if len(improved_set) > best_size:
                best_set = improved_set
                best_size = len(best_set)

        # Final greedy fill
        for e_idx in range(total_edges):
            if e_idx not in best_set and is_valid_addition(best_set, e_idx):
                best_set.add(e_idx)

        return best_set

    # Use multiple strategies
    start = time.time()
    deadline = start + 110

    # Strategy 1: Many quick random restarts with short local search
    best_set = hill_climb_with_restarts(num_restarts=30, time_per_ls=1.5, seed_base=42)
    best_size = len(best_set)

    # Strategy 2: Fewer restarts with longer local search on best
    if time.time() < deadline - 20:
        remaining = deadline - time.time() - 10
        improved = local_search(best_set.copy(), remaining, seed=9999)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)

    # Strategy 3: Another round of restarts with different seed
    if time.time() < deadline - 10:
        best2 = hill_climb_with_restarts(num_restarts=15, time_per_ls=2.0, seed_base=12345)
        if len(best2) > best_size:
            best_set = best2
            best_size = len(best_set)

    # Final greedy fill to ensure maximality
    for e_idx in range(total_edges):
        if e_idx not in best_set and is_valid_addition(best_set, e_idx):
            best_set.add(e_idx)

    # Convert to list of [u, v]
    result = []
    for e_idx in best_set:
        u, v = all_edges[e_idx]
        result.append([u, v])

    return result
