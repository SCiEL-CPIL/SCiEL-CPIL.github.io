def construct():
    import time
    import random

    deadline = time.time() + 115
    d = 7
    nverts = 1 << d
    num_edges = d * (1 << (d - 1))  # 7 * 64 = 448

    # 1. Build edges and mapping
    edges = []
    for u in range(nverts):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))

    # 2. Build faces
    edge_faces = [[] for _ in range(num_edges)]
    face_edges = []
    face_indices = {}

    for e_idx, (u, v) in enumerate(edges):
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            verts = frozenset([u, v, u2, v2])
            if verts not in face_indices:
                face_indices[verts] = len(face_edges)
                face_edges.append([])
            f_idx = face_indices[verts]
            face_edges[f_idx].append(e_idx)
            edge_faces[e_idx].append(f_idx)

    # Convert to tuples for speed
    face_edges_tuple = [tuple(fe) for fe in face_edges]
    num_faces = len(face_edges_tuple)

    # Precompute for each edge the list of (face, other_edge_in_face) pairs
    # This helps in checking blockers and performing swaps
    # Actually, let's keep it simple but optimize the inner loops.

    # 3. Initialize state
    def greedy_random():
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

    best_selected, best_face_count, best_count = greedy_random()

    # 4. Local Search
    current_selected = best_selected[:]
    current_face_count = best_face_count[:]
    current_count = best_count

    # Precompute for each edge e, the list of faces it belongs to
    # And for each face f, the list of edges it contains
    # We already have edge_faces and face_edges_tuple.

    # To speed up blocker checking, we can precompute for each edge e
    # the set of edges that share a face with e.
    # But let's stick to the current structure and just optimize the loop.

    # Helper to try adding an edge
    def try_add(e):
        nonlocal current_selected, current_face_count, current_count
        for f in edge_faces[e]:
            if current_face_count[f] == 3:
                return False
        current_selected[e] = True
        current_count += 1
        for f in edge_faces[e]:
            current_face_count[f] += 1
        return True

    # Helper to try removing an edge
    def try_remove(e):
        nonlocal current_selected, current_face_count, current_count
        current_selected[e] = False
        current_count -= 1
        for f in edge_faces[e]:
            current_face_count[f] -= 1
        return True

    # Main loop
    while time.time() < deadline:
        # 1. Try to add an edge
        added = False
        for e in range(num_edges):
            if current_selected[e]:
                continue
            can_add = True
            for f in edge_faces[e]:
                if current_face_count[f] == 3:
                    can_add = False
                    break
            if can_add:
                try_add(e)
                added = True

        if added:
            if current_count > best_count:
                best_count = current_count
                best_selected = current_selected[:]
                best_face_count = current_face_count[:]

        # 2. If no add, try swaps
        unselected = [e for e in range(num_edges) if not current_selected[e]]
        if not unselected:
            break

        e = random.choice(unselected)

        # Find blockers
        blockers = []
        for f in edge_faces[e]:
            if current_face_count[f] == 3:
                for r in face_edges_tuple[f]:
                    if current_selected[r] and r != e:
                        blockers.append(r)

        if not blockers:
            try_add(e)
            if current_count > best_count:
                best_count = current_count
                best_selected = current_selected[:]
                best_face_count = current_face_count[:]
            continue

        # Pick a random blocker to remove
        r = random.choice(blockers)

        # Check if swap is valid
        valid_swap = True
        for f in edge_faces[e]:
            if current_face_count[f] == 3:
                if r not in face_edges_tuple[f]:
                    valid_swap = False
                    break

        if valid_swap:
            try_remove(r)
            try_add(e)

    # Return best found
    result = [list(edges[i]) for i in range(num_edges) if best_selected[i]]
    return result
