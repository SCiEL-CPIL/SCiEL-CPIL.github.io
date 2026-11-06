def construct():
    import time
    import random
    import numpy as np

    start_time = time.time()
    deadline = start_time + 115.0

    d = 7
    nverts = 1 << d  # 128

    # 1. Build edges
    edges = []
    edge_map = {}
    for u in range(nverts):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_map[(u, v)] = idx
                edge_map[(v, u)] = idx

    num_edges = len(edges)  # 448

    # 2. Build faces
    face_edges_list = []
    edge_faces = [[] for _ in range(num_edges)]
    face_vertices_set = set()

    for i in range(d):
        for j in range(i + 1, d):
            for u in range(nverts):
                if (u & (1 << i)) != 0:
                    continue
                if (u & (1 << j)) != 0:
                    continue

                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)

                face_key = tuple(sorted([v1, v2, v3, v4]))
                if face_key in face_vertices_set:
                    continue

                face_vertices_set.add(face_key)
                f_idx = len(face_edges_list)

                e1 = edge_map[(min(v1, v2), max(v1, v2))]
                e2 = edge_map[(min(v1, v3), max(v1, v3))]
                e3 = edge_map[(min(v2, v4), max(v2, v4))]
                e4 = edge_map[(min(v3, v4), max(v3, v4))]

                face_edges_list.append((e1, e2, e3, e4))

                edge_faces[e1].append(f_idx)
                edge_faces[e2].append(f_idx)
                edge_faces[e3].append(f_idx)
                edge_faces[e4].append(f_idx)

    num_faces = len(face_edges_list)  # 336 faces
    face_edges_set = [set(f) for f in face_edges_list]

    # Precompute for each edge: which faces it belongs to, and for each face which other edges are in it
    # Also precompute the "opposite" edge in each face (the edge not sharing a vertex with this edge in the face)
    # Actually in a 4-cycle face, each edge is opposite to exactly one other edge.

    # For each edge e and each face f containing e, find the opposite edge in that face
    edge_opposite_in_face = {}  # (edge, face) -> opposite edge
    for f_idx, (e1, e2, e3, e4) in enumerate(face_edges_list):
        # In the face, edges are arranged as a cycle: e1-e2-e3-e4-e1
        # e1 is opposite to e3, e2 is opposite to e4
        edge_opposite_in_face[(e1, f_idx)] = e3
        edge_opposite_in_face[(e3, f_idx)] = e1
        edge_opposite_in_face[(e2, f_idx)] = e4
        edge_opposite_in_face[(e4, f_idx)] = e2

    # 3. Greedy initialization
    def random_greedy(seed=None):
        if seed is not None:
            random.seed(seed)

        selected = [False] * num_edges
        face_count = [0] * num_faces
        order = list(range(num_edges))
        random.shuffle(order)

        count = 0
        for e in order:
            ok = True
            for f in edge_faces[e]:
                if face_count[f] == 3:
                    ok = False
                    break
            if ok:
                selected[e] = True
                count += 1
                for f in edge_faces[e]:
                    face_count[f] += 1

        return selected, face_count, count

    # Try multiple random starts
    best_selected = None
    best_face_count = None
    best_count = 0
    t0 = time.time()
    while time.time() - t0 < 2.0:
        s, fc, c = random_greedy()
        if c > best_count:
            best_count = c
            best_selected = s
            best_face_count = fc

    selected = best_selected[:]
    face_count = best_face_count[:]
    count = best_count
    best_found = count
    best_selected_copy = selected[:]
    best_face_count_copy = face_count[:]

    # Local search with simulated annealing and 2-swap moves
    def try_add_edge(e):
        nonlocal selected, face_count, count, best_found, best_selected_copy, best_face_count_copy
        for f in edge_faces[e]:
            if face_count[f] == 3:
                return False
        selected[e] = True
        count += 1
        for f in edge_faces[e]:
            face_count[f] += 1
        if count > best_found:
            best_found = count
            best_selected_copy = selected[:]
            best_face_count_copy = face_count[:]
        return True

    def remove_edge(e):
        nonlocal selected, face_count, count
        selected[e] = False
        count -= 1
        for f in edge_faces[e]:
            face_count[f] -= 1
        return True

    # 2-swap: remove one selected edge r, add two unselected edges e1, e2
    # This can yield net +1 improvement
    def try_2swap(r, e1, e2):
        """Remove r, add e1 and e2. Returns True if successful."""
        nonlocal selected, face_count, count, best_found, best_selected_copy, best_face_count_copy
        # Check if removing r and adding e1, e2 is valid
        # First, temporarily remove r
        for f in edge_faces[r]:
            face_count[f] -= 1
        selected[r] = False
        count -= 1

        # Try to add e1
        ok1 = True
        for f in edge_faces[e1]:
            if face_count[f] == 3:
                ok1 = False
                break
        if not ok1:
            # Revert
            selected[r] = True
            count += 1
            for f in edge_faces[r]:
                face_count[f] += 1
            return False

        selected[e1] = True
        count += 1
        for f in edge_faces[e1]:
            face_count[f] += 1

        # Try to add e2
        ok2 = True
        for f in edge_faces[e2]:
            if face_count[f] == 3:
                ok2 = False
                break
        if not ok2:
            # Revert e1 and r
            selected[e1] = False
            count -= 1
            for f in edge_faces[e1]:
                face_count[f] -= 1
            selected[r] = True
            count += 1
            for f in edge_faces[r]:
                face_count[f] += 1
            return False

        selected[e2] = True
        count += 1
        for f in edge_faces[e2]:
            face_count[f] += 1

        if count > best_found:
            best_found = count
            best_selected_copy = selected[:]
            best_face_count_copy = face_count[:]
        return True

    # Main local search loop
    neutral_swap_count = 0
    max_neutral_swaps = 3000
    temperature = 1.0
    cooling_rate = 0.99995
    min_temperature = 0.01
    polish_phase = False

    # Track which edges are "hot" (participate in many tight faces)
    # A face is tight if it has exactly 3 selected edges

    while time.time() < deadline:
        # Phase 1: Greedy Add - try to add any edge that can be added
        added = False
        for e in range(num_edges):
            if selected[e]:
                continue
            can_add = True
            for f in edge_faces[e]:
                if face_count[f] == 3:
                    can_add = False
                    break
            if can_add:
                try_add_edge(e)
                added = True
                neutral_swap_count = 0
                break

        if added:
            continue

        # Phase 2: Try 1-swap (remove 1, add 1) or 2-swap (remove 1, add 2)
        # First try 2-swap for net +1 improvement
        if random.random() < 0.3:
            # 2-swap: pick a random selected edge to remove, then try to add 2 edges
            if count > 0:
                r = random.randint(0, num_edges - 1)
                if not selected[r]:
                    # Find a selected edge
                    sel_indices = [i for i in range(num_edges) if selected[i]]
                    if sel_indices:
                        r = sel_indices[random.randint(0, len(sel_indices) - 1)]
                    else:
                        continue

                # Find unselected edges that could be added after removing r
                # An edge e can be added after removing r if all faces containing e
                # have face_count <= 2 after removing r
                candidate_adds = []
                for e in range(num_edges):
                    if selected[e]:
                        continue
                    ok = True
                    for f in edge_faces[e]:
                        # After removing r, face_count[f] decreases by 1 if r is in f
                        new_fc = face_count[f] - (1 if r in face_edges_set[f] else 0)
                        if new_fc == 3:
                            ok = False
                            break
                    if ok:
                        candidate_adds.append(e)

                if len(candidate_adds) >= 2:
                    # Try pairs from candidate_adds
                    # Limit the number of pairs to try
                    max_pairs = min(20, len(candidate_adds))
                    pairs_tried = 0
                    random.shuffle(candidate_adds)
                    for i in range(min(max_pairs, len(candidate_adds))):
                        for j in range(i + 1, min(max_pairs, len(candidate_adds))):
                            e1 = candidate_adds[i]
                            e2 = candidate_adds[j]
                            if try_2swap(r, e1, e2):
                                neutral_swap_count = 0
                                break
                            pairs_tried += 1
                        if neutral_swap_count == 0:
                            break
                    else:
                        neutral_swap_count += 1
                else:
                    neutral_swap_count += 1
            else:
                neutral_swap_count += 1
        else:
            # 1-swap: pick an unselected edge, find a blocking face, remove an edge from that face
            e = random.randint(0, num_edges - 1)
            if selected[e]:
                continue

            b_faces = []
            for f in edge_faces[e]:
                if face_count[f] == 3:
                    b_faces.append(f)

            if not b_faces:
                try_add_edge(e)
                neutral_swap_count = 0
                continue

            # Find a candidate edge to remove from the blocking faces
            candidate_r = None
            first_f = b_faces[0]
            sel_in_first = [re for re in face_edges_list[first_f] if selected[re]]
            random.shuffle(sel_in_first)
            for r in sel_in_first:
                ok = True
                for f in b_faces[1:]:
                    if r not in face_edges_set[f]:
                        ok = False
                        break
                if ok:
                    candidate_r = r
                    break

            if candidate_r is None:
                neutral_swap_count += 1
                if neutral_swap_count > max_neutral_swaps:
                    # Restart with random greedy
                    t0 = time.time()
                    while time.time() - t0 < 1.0:
                        s, fc, c = random_greedy()
                        if c > best_count:
                            best_count = c
                            best_selected = s
                            best_face_count = fc
                    selected = best_selected[:]
                    face_count = best_face_count[:]
                    count = best_count
                    neutral_swap_count = 0
                    temperature = 1.0
                continue

            # Try the swap: remove candidate_r, add e
            remove_edge(candidate_r)
            can_add = True
            for f in edge_faces[e]:
                if face_count[f] == 3:
                    can_add = False
                    break
            if can_add:
                try_add_edge(e)
                neutral_swap_count += 1
                if neutral_swap_count > max_neutral_swaps:
                    t0 = time.time()
                    while time.time() - t0 < 1.0:
                        s, fc, c = random_greedy()
                        if c > best_count:
                            best_count = c
                            best_selected = s
                            best_face_count = fc
                    selected = best_selected[:]
                    face_count = best_face_count[:]
                    count = best_count
                    neutral_swap_count = 0
                    temperature = 1.0
            else:
                # Revert
                try_add_edge(candidate_r)
                neutral_swap_count += 1

        # Cooling
        if temperature > min_temperature:
            temperature *= cooling_rate

    # Final polish: try to add any remaining edges
    for e in range(num_edges):
        if selected[e]:
            continue
        can_add = True
        for f in edge_faces[e]:
            if face_count[f] == 3:
                can_add = False
                break
        if can_add:
            try_add_edge(e)

    # Return the best solution found
    result = []
    for i in range(num_edges):
        if best_selected_copy[i]:
            result.append(list(edges[i]))

    return result
