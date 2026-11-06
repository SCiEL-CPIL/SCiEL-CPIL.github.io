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

    # Precompute faces as tuples of edge indices
    # A face is defined by (u, i, j) where i != j
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

                e1_idx = edge_to_index[e1]
                e2_idx = edge_to_index[e2]
                e3_idx = edge_to_index[e3]
                e4_idx = edge_to_index[e4]

                face = (e1_idx, e2_idx, e3_idx, e4_idx)
                faces_idx.append(face)
                face_idx = len(faces_idx) - 1

                for e_idx in face:
                    edge_faces_idx[e_idx].append(face_idx)

    # Number of faces
    num_faces = len(faces_idx)

    # Convert faces to numpy array for fast operations
    faces_arr = np.array(faces_idx, dtype=np.int32)

    def is_valid_addition(chosen_mask, e_idx):
        """Check if adding edge e_idx to the set defined by chosen_mask is valid."""
        for f_idx in edge_faces_idx[e_idx]:
            face = faces_arr[f_idx]
            # Count how many of the other 3 edges are in chosen
            count = 0
            for idx in face:
                if idx != e_idx and chosen_mask[idx]:
                    count += 1
                    if count == 3:
                        return False
        return True

    def greedy_random(chosen_mask, order):
        """Add edges in the given order if valid."""
        for e_idx in order:
            if not chosen_mask[e_idx]:
                if is_valid_addition(chosen_mask, e_idx):
                    chosen_mask[e_idx] = True
        return chosen_mask

    def greedy_repair(chosen_mask):
        """Try to add any missing edge that can be added."""
        improved = True
        while improved:
            improved = False
            for e_idx in range(total_edges):
                if not chosen_mask[e_idx]:
                    if is_valid_addition(chosen_mask, e_idx):
                        chosen_mask[e_idx] = True
                        improved = True
        return chosen_mask

    def local_search(chosen_mask, time_limit):
        """
        Local search with 2-for-1 swaps.
        """
        start_time = time.time()
        chosen_mask = chosen_mask.copy()

        def compute_unblocks():
            """
            For each missing edge e, compute the set of edges in S that block it.
            An edge b in S is a blocker for e if removing b allows e to be added
            (i.e., for every face F containing e, if b is the only edge in F \ {e} ∩ S,
            then removing b fixes F).
            
            More precisely: e is blocked by face F if |F \ {e} ∩ S| == 3.
            A single edge b unblocks e if for every bad face F of e, b ∈ F \ {e}.
            """
            missing = [i for i in range(total_edges) if not chosen_mask[i]]
            single_unblockers = {}
            for e in missing:
                bad_faces = []
                for f_idx in edge_faces_idx[e]:
                    face = faces_arr[f_idx]
                    others_in_s = [idx for idx in face if idx != e and chosen_mask[idx]]
                    if len(others_in_s) == 3:
                        bad_faces.append(others_in_s)
                
                if not bad_faces:
                    # Should not happen in a maximal set
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
            
            # Build unblocks: for each b in S, which missing edges does it unblock?
            unblocks = {b: [] for b in range(total_edges) if chosen_mask[b]}
            for e, blockers in single_unblockers.items():
                for b in blockers:
                    if b in unblocks:
                        unblocks[b].append(e)
            
            return missing, single_unblockers, unblocks

        def try_2_for_1(unblocks):
            """Try to find a 2-for-1 swap: remove b, add e1, e2."""
            b_list = [b for b in range(total_edges) if chosen_mask[b]]
            random.shuffle(b_list)
            
            for b in b_list:
                candidates = unblocks.get(b, [])
                if len(candidates) >= 2:
                    # Try all pairs? Or just a random sample?
                    # For efficiency, try a few random pairs
                    num_trials = min(10, len(candidates) * (len(candidates) - 1) // 2)
                    for _ in range(num_trials):
                        e1, e2 = random.sample(candidates, 2)
                        # Check validity
                        new_mask = chosen_mask.copy()
                        new_mask[b] = False
                        if is_valid_addition(new_mask, e1):
                            new_mask[e1] = True
                            if is_valid_addition(new_mask, e2):
                                new_mask[e2] = True
                                return new_mask, b, e1, e2
            return None, None, None, None

        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            missing, single_unblockers, unblocks = compute_unblocks()
            
            # First, add any edge that can be added directly (should be none in maximal set)
            for e in missing:
                if not chosen_mask[e] and is_valid_addition(chosen_mask, e):
                    chosen_mask[e] = True
                    improved = True
                    # Restart computation since state changed
                    break
            
            if improved:
                continue
            
            # Try 2-for-1 swaps
            new_mask, b, e1, e2 = try_2_for_1(unblocks)
            if new_mask is not None:
                chosen_mask = new_mask
                improved = True
                continue
            
            # If no 2-for-1 found, try perturbation: remove a random edge and repair
            if random.random() < 0.15:
                chosen_edges = [i for i in range(total_edges) if chosen_mask[i]]
                if chosen_edges:
                    b_to_remove = random.choice(chosen_edges)
                    chosen_mask[b_to_remove] = False
                    greedy_repair(chosen_mask)
                    improved = True
                    continue
            
            # If stuck, try a different perturbation: remove 2 random edges and repair
            if random.random() < 0.05:
                chosen_edges = [i for i in range(total_edges) if chosen_mask[i]]
                if len(chosen_edges) >= 2:
                    b1, b2 = random.sample(chosen_edges, 2)
                    chosen_mask[b1] = False
                    chosen_mask[b2] = False
                    greedy_repair(chosen_mask)
                    improved = True
                    continue
            
            # Otherwise, break to avoid infinite loop
            break
        
        return chosen_mask

    # Main search loop
    start = time.time()
    deadline = start + 140  # 10s safety margin before the 150s kill

    best_mask = np.zeros(total_edges, dtype=bool)
    best_size = 0

    # Phase 1: Multiple random greedy restarts
    num_restarts = 500
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        mask = np.zeros(total_edges, dtype=bool)
        order = list(range(total_edges))
        random.shuffle(order)
        greedy_random(mask, order)
        greedy_repair(mask)
        
        size = int(np.sum(mask))
        if size > best_size:
            best_size = size
            best_mask = mask.copy()

    # Phase 2: Local search on the best set
    if time.time() < deadline - 5:
        improved_mask = local_search(best_mask, deadline - time.time() - 5)
        size = int(np.sum(improved_mask))
        if size > best_size:
            best_mask = improved_mask
            best_size = size

    # Phase 3: More local search with different starting points
    if time.time() < deadline - 5:
        for i in range(50):
            if time.time() > deadline - 5:
                break
            
            mask = np.zeros(total_edges, dtype=bool)
            order = list(range(total_edges))
            random.shuffle(order)
            greedy_random(mask, order)
            greedy_repair(mask)
            
            size = int(np.sum(mask))
            if size > best_size:
                best_size = size
                best_mask = mask.copy()
            
            # Run local search on this set
            if time.time() < deadline - 10:
                time_left = deadline - time.time() - 10
                if time_left > 1:
                    improved_mask = local_search(mask, time_left)
                    size = int(np.sum(improved_mask))
                    if size > best_size:
                        best_mask = improved_mask
                        best_size = size

    # Convert to list of [u, v]
    result = []
    for e_idx in range(total_edges):
        if best_mask[e_idx]:
            u, v = all_edges[e_idx]
            result.append([u, v])

    return result
