def construct():
    import time
    import random
    import numpy as np
    from itertools import combinations

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

    num_faces = len(face_edges_list)
    face_edges_set = [set(f) for f in face_edges_list]

    # 3. Greedy Initialization
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

    # Try a few random starts to find a good baseline
    best_selected, best_face_count, best_count = None, None, 0
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
    best_selected = selected[:]
    best_face_count = face_count[:]

    # 4. Local Search with Simulated Annealing-like swaps
    def try_add_edge(e):
        nonlocal selected, face_count, count, best_found, best_selected, best_face_count
        for f in edge_faces[e]:
            if face_count[f] == 3:
                return False
        selected[e] = True
        count += 1
        for f in edge_faces[e]:
            face_count[f] += 1
        if count > best_found:
            best_found = count
            best_selected = selected[:]
            best_face_count = face_count[:]
        return True

    def remove_edge(e):
        nonlocal selected, face_count, count
        selected[e] = False
        count -= 1
        for f in edge_faces[e]:
            face_count[f] -= 1
        return True

    # Main loop
    neutral_swap_count = 0
    max_neutral_swaps = 1000

    while time.time() < deadline:
        # Phase 1: Greedy Add
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

        # Phase 2: Swap (if no add possible)
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

        candidate_r = None
        if b_faces:
            first_f = b_faces[0]
            sel_in_first = [re for re in face_edges_list[first_f] if selected[re]]
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
            continue

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
        else:
            try_add_edge(candidate_r)
            neutral_swap_count += 1

    result = []
    for i in range(num_edges):
        if best_selected[i]:
            result.append(list(edges[i]))

    return result
