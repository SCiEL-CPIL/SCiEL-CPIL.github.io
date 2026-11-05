import random
import time
import heapq

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin

    d = 7
    n = 2 ** d

    # 1. Precompute edges and faces
    edges = []
    edge_id_map = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edge_id = len(edges)
                edges.append((x, y))
                edge_id_map[(x, y)] = edge_id
                edge_id_map[(y, x)] = edge_id

    n_edges = len(edges)

    # Face ID: (base_vertex, i, j) where i < j
    face_edges = []
    face_list = []
    for v in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                face_id = len(face_list)
                face_list.append((v, i, j))

                v_i = v ^ (1 << i)
                v_j = v ^ (1 << j)
                v_ij = v ^ (1 << i) ^ (1 << j)

                e1 = edge_id_map[(min(v, v_i), max(v, v_i))]
                e2 = edge_id_map[(min(v_i, v_ij), max(v_i, v_ij))]
                e3 = edge_id_map[(min(v, v_j), max(v, v_j))]
                e4 = edge_id_map[(min(v_j, v_ij), max(v_j, v_ij))]

                face_edges.append([e1, e2, e3, e4])

    n_faces = len(face_edges)

    # Build edge_faces: for each edge, list of face indices
    edge_faces = [[] for _ in range(n_edges)]
    for f_idx, f_edges in enumerate(face_edges):
        for e_idx in f_edges:
            edge_faces[e_idx].append(f_idx)

    # 2. MCS-based initialization: build conflict graph, select edges in order of
    #    "least constrained" (fewest conflicts with already-selected edges),
    #    then repair into a valid hitting set (complement = removed edges).
    #    The conflict graph: two edges conflict if they share a 2-face.
    #    MCS ordering gives a global structure-aware selection.

    # Precompute conflict sets (edges sharing a face)
    # For speed, use bitsets via Python int (448 bits fits in int)
    conflict_mask = [0] * n_edges
    for f_idx, f_edges in enumerate(face_edges):
        for a in range(4):
            for b in range(a + 1, 4):
                ea, eb = f_edges[a], f_edges[b]
                conflict_mask[ea] |= (1 << eb)
                conflict_mask[eb] |= (1 << ea)

    def mcs_initialization(rng):
        """
        Use Maximum Cardinality Search to produce an ordering of edges.
        Then greedily select edges in that order, skipping any that would
        create a C4 (i.e., all 4 edges of some face present).
        The complement of selected edges is our hitting set R.
        """
        # MCS: iteratively pick vertex with max number of already-picked neighbors
        # For our purpose, we want the REVERSE: pick least-constrained first.
        # Actually, for maximum independent set, MCS ordering (most constrained last)
        # means we process in reverse MCS order (least constrained first).
        
        # Simpler approach: iterative greedy with dynamic degree
        # "degree" = number of conflicts with unprocessed vertices
        # Pick vertex with min degree (least constrained), add to independent set,
        # remove it and its neighbors from consideration.
        
        remaining = set(range(n_edges))
        # Initial degree = number of conflicts
        degree = [bin(conflict_mask[e]).count('1') for e in range(n_edges)]
        
        selected = set()
        
        while remaining:
            # Find vertex with minimum degree among remaining
            best_e = -1
            best_deg = float('inf')
            for e in remaining:
                if degree[e] < best_deg:
                    best_deg = degree[e]
                    best_e = e
                elif degree[e] == best_deg and rng.random() < 0.3:
                    best_e = e
            
            if best_e == -1:
                break
            
            selected.add(best_e)
            remaining.discard(best_e)
            
            # Remove neighbors of best_e from remaining, update degrees
            neighbors = conflict_mask[best_e]
            for e in list(remaining):
                if neighbors & (1 << e):
                    remaining.discard(e)
                    for other in remaining:
                        if conflict_mask[e] & (1 << other):
                            degree[other] -= 1
        
        # selected is a maximal independent set in the conflict graph
        # The complement is a hitting set (removes at least one edge from each face)
        R = set(range(n_edges)) - selected
        return R

    # 3. Heap-based greedy hitting set (faster than scanning all edges)
    def greedy_hitting_set_heap(rng):
        covered = [False] * n_faces
        R = set()
        benefit = [len(edge_faces[e]) for e in range(n_edges)]
        
        # Min-heap with negated benefit (max-heap)
        heap = [(-benefit[e], e) for e in range(n_edges)]
        heapq.heapify(heap)
        
        n_uncovered = n_faces
        while n_uncovered > 0:
            # Pop until we find a valid (not-in-R, positive-benefit) edge
            best_e = -1
            while heap:
                neg_ben, e = heapq.heappop(heap)
                if e not in R and -neg_ben > 0:
                    best_e = e
                    break
                elif e not in R and -neg_ben == 0:
                    # Zero benefit, try fallback
                    best_e = e
                    break
            
            if best_e == -1:
                # Fallback: find any edge covering an uncovered face
                for f in range(n_faces):
                    if not covered[f]:
                        for e in face_edges[f]:
                            if e not in R:
                                best_e = e
                                break
                        break
                if best_e == -1:
                    break
            
            R.add(best_e)
            
            for f in edge_faces[best_e]:
                if not covered[f]:
                    covered[f] = True
                    n_uncovered -= 1
                    for e in face_edges[f]:
                        if e != best_e and e not in R:
                            benefit[e] -= 1
                            heapq.heappush(heap, (-benefit[e], e))
            
            if time.time() - start_time > time_limit - 10:
                break

        return R

    # 4. Local search minimization with 1-removal and 2-opt swaps
    def minimize_hitting_set(R, rng, max_passes=4):
        R = set(R)
        cover_count = [0] * n_faces
        for e in R:
            for f in edge_faces[e]:
                cover_count[f] += 1

        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1

            # Pass 1: Try to remove redundant edges
            r_list = list(R)
            rng.shuffle(r_list)
            for e in r_list:
                if e not in R:
                    continue
                redundant = True
                for f in edge_faces[e]:
                    if cover_count[f] == 1:
                        redundant = False
                        break
                if redundant:
                    R.remove(e)
                    for f in edge_faces[e]:
                        cover_count[f] -= 1
                    improved = True
                    if time.time() - start_time > time_limit - 5:
                        return R

            # Pass 2: Try 2-opt swaps (remove A, add B)
            if time.time() - start_time < time_limit - 10 and len(R) > 10:
                r_list2 = list(R)
                rng.shuffle(r_list2)
                swaps_made = 0
                for e_remove in r_list2:
                    if e_remove not in R:
                        continue
                    newly_uncovered = []
                    for f in edge_faces[e_remove]:
                        if cover_count[f] == 1:
                            newly_uncovered.append(f)
                    if not newly_uncovered:
                        continue

                    best_e_add = -1
                    best_gain = 0
                    candidate_edges = set()
                    for f in newly_uncovered:
                        for ce in face_edges[f]:
                            if ce not in R:
                                candidate_edges.add(ce)

                    if len(candidate_edges) > 80:
                        candidate_edges = set(rng.sample(list(candidate_edges), 80))

                    for e_add in candidate_edges:
                        gain = 0
                        for f in edge_faces[e_add]:
                            if cover_count[f] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_e_add = e_add

                    if best_e_add != -1 and best_gain >= len(newly_uncovered):
                        R.remove(e_remove)
                        for f in edge_faces[e_remove]:
                            cover_count[f] -= 1
                        R.add(best_e_add)
                        for f in edge_faces[best_e_add]:
                            cover_count[f] += 1
                        improved = True
                        swaps_made += 1
                        if swaps_made >= 20:
                            break

                    if time.time() - start_time > time_limit - 5:
                        return R

            # Pass 3: 2-swap (remove A, B; add C) for larger moves
            if time.time() - start_time < time_limit - 10 and len(R) > 20:
                r_list3 = list(R)
                rng.shuffle(r_list3)
                two_swap_made = 0
                for idx, e_remove1 in enumerate(r_list3):
                    if e_remove1 not in R:
                        continue
                    for e_remove2 in r_list3[idx+1:]:
                        if e_remove2 not in R or e_remove2 == e_remove1:
                            continue
                        # Check if removing both leaves uncovered faces
                        newly_uncovered = []
                        for f in edge_faces[e_remove1]:
                            if cover_count[f] == 1:
                                newly_uncovered.append(f)
                            elif cover_count[f] == 2:
                                # Check if e_remove2 also covers it
                                if e_remove2 in face_edges[f]:
                                    newly_uncovered.append(f)
                        for f in edge_faces[e_remove2]:
                            if cover_count[f] == 1:
                                if f not in newly_uncovered:
                                    newly_uncovered.append(f)
                        if not newly_uncovered:
                            continue

                        candidate_edges = set()
                        for f in newly_uncovered:
                            for ce in face_edges[f]:
                                if ce not in R:
                                    candidate_edges.add(ce)

                        if len(candidate_edges) > 40:
                            candidate_edges = set(rng.sample(list(candidate_edges), 40))

                        best_e_add = -1
                        best_gain = 0
                        for e_add in candidate_edges:
                            gain = 0
                            for f in edge_faces[e_add]:
                                if cover_count[f] == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_e_add = e_add

                        if best_e_add != -1 and best_gain >= len(newly_uncovered):
                            R.discard(e_remove1)
                            R.discard(e_remove2)
                            for f in edge_faces[e_remove1]:
                                cover_count[f] -= 1
                            for f in edge_faces[e_remove2]:
                                cover_count[f] -= 1
                            R.add(best_e_add)
                            for f in edge_faces[best_e_add]:
                                cover_count[f] += 1
                            improved = True
                            two_swap_made += 1
                            break
                    if two_swap_made >= 5:
                        break
                    if time.time() - start_time > time_limit - 5:
                        return R

        return R

    # 5. Initialization with multiple methods
    best_R = None
    best_size = float('inf')

    # MCS-based initializations
    for seed in range(1, 16):
        if time.time() - start_time > time_limit - 20:
            break
        R_candidate = mcs_initialization(random.Random(seed))
        R_candidate = minimize_hitting_set(R_candidate, random.Random(seed))
        if len(R_candidate) < best_size:
            best_size = len(R_candidate)
            best_R = R_candidate

    # Heap-based greedy initializations
    for seed in range(1, 16):
        if time.time() - start_time > time_limit - 15:
            break
        R_candidate = greedy_hitting_set_heap(random.Random(seed))
        R_candidate = minimize_hitting_set(R_candidate, random.Random(seed))
        if len(R_candidate) < best_size:
            best_size = len(R_candidate)
            best_R = R_candidate

    if best_R is None:
        best_R = greedy_hitting_set_heap(random.Random(123))
        best_size = len(best_R)

    rng = random.Random(42)

    # 6. Simulated Annealing-style perturbation and repair
    while time.time() - start_time < time_limit - 8:
        current_R = set(best_R)

        # Perturb: remove random edges
        num_remove = rng.randint(1, 4)
        to_remove = rng.sample(list(current_R), min(num_remove, len(current_R)))
        for e in to_remove:
            current_R.remove(e)

        # Repair
        cover_count = [0] * n_faces
        for e in current_R:
            for f in edge_faces[e]:
                cover_count[f] += 1

        uncovered = [f for f in range(n_faces) if cover_count[f] == 0]
        while uncovered:
            f = rng.choice(uncovered)

            best_e = -1
            max_cover = -1
            for e in face_edges[f]:
                count = 0
                for ff in edge_faces[e]:
                    if cover_count[ff] == 0:
                        count += 1
                if count > max_cover:
                    max_cover = count
                    best_e = e
                elif count == max_cover:
                    if rng.random() < 0.5:
                        best_e = e

            if best_e != -1:
                current_R.add(best_e)
                for ff in edge_faces[best_e]:
                    if cover_count[ff] == 0:
                        uncovered.remove(ff)
                    cover_count[ff] += 1
            else:
                e = rng.choice(face_edges[f])
                current_R.add(e)
                for ff in edge_faces[e]:
                    if cover_count[ff] == 0:
                        uncovered.remove(ff)
                    cover_count[ff] += 1

        # Minimize
        current_R = minimize_hitting_set(current_R, rng)

        if len(current_R) < best_size:
            best_size = len(current_R)
            best_R = current_R

    # Convert R (edge IDs to remove) to result: all edges NOT in R
    result = []
    for e_id, (u, v) in enumerate(edges):
        if e_id not in best_R:
            result.append([u, v])

    return result
