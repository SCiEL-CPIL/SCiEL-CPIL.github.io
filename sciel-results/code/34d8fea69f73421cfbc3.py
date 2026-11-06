def construct():
    import time
    import random
    import numpy as np

    deadline = time.time() + 115
    d = 7
    nverts = 1 << d
    num_edges = d * (1 << (d - 1))  # 7 * 64 = 448

    # 1. Build edges and mapping
    edges = []
    edge_map = {}
    for u in range(nverts):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_map[(u, v)] = idx

    # 2. Build faces
    # A face is defined by two bits i, j and a fixed pattern for the other bits.
    # There are C(7,2)=21 pairs of bits.
    # For each pair, there are 2^(7-2)=16 faces.
    # Total 336 faces.

    edge_faces = [[] for _ in range(num_edges)]
    face_edges = []

    # We need to map each edge to the faces it belongs to.
    # And each face to its 4 edges.

    # Let's implement the face construction via edges.
    # This will be O(E * D) which is 448 * 6 = 2688 operations. Fast.

    face_indices = {}  # frozenset -> index
    face_edges_list = []

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
                face_indices[verts] = len(face_edges_list)
                face_edges_list.append([])
            f_idx = face_indices[verts]
            # Add this edge to the face
            face_edges_list[f_idx].append(e_idx)
            # Also record for the edge
            edge_faces[e_idx].append(f_idx)

    # Now face_edges_list has lists of edges. Each should have 4 edges.
    # Convert to tuples for immutability and speed
    face_edges_tuple = [tuple(fe) for fe in face_edges_list]
    num_faces = len(face_edges_tuple)

    # 3. Initialize state
    # Greedy random start
    def greedy_random():
        selected = [False] * num_edges
        face_count = [0] * num_faces
        order = list(range(num_edges))
        random.shuffle(order)
        count = 0
        for e in order:
            # Check if adding e is valid
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
    # We will try to improve best_count.
    # Moves:
    # 1. Add: If possible, add an edge.
    # 2. Swap: Remove one, add one (neutral).

    # To speed up, we can maintain a list of "available" edges (can be added).
    # But checking availability is O(6).

    current_selected = best_selected[:]
    current_face_count = best_face_count[:]
    current_count = best_count

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
    # We use a simple hill climbing with random restarts or just long random walks.
    # Given 115s, we can do many steps.

    # Let's try a simulated annealing style or just random swaps that don't decrease size.
    # Since Add increases size, we always take Add.
    # Swap keeps size same. We take Swap if it increases the number of "available" edges?
    # Or just take Swap randomly to explore.

    # Let's just do:
    # 1. Try to find any Add move. If found, take it.
    # 2. If no Add move, pick a random Swap move (1 remove, 1 add) that is valid.

    # To make it efficient, we can precompute for each unselected edge e, the set of edges that block it.
    # Blockers(e) = { r in S | exists f in edge_faces[e] such that r in face_edges[f] and face_count[f]==3 }
    # If Blockers(e) is empty, e is Add-able.
    # If not, we can remove one r in Blockers(e) and try to add e.
    # This is a Swap.

    # We will iterate through edges randomly.

    while time.time() < deadline:
        # 1. Try to add an edge
        added = False
        # Randomly sample some unselected edges to try adding
        # If we find one, add it and restart the scan (or just continue)

        # A more systematic approach:
        # Check all unselected edges? 448 is small.
        # Let's just iterate through all edges.

        for e in range(num_edges):
            if current_selected[e]:
                continue
            # Check if e can be added
            can_add = True
            for f in edge_faces[e]:
                if current_face_count[f] == 3:
                    can_add = False
                    break
            if can_add:
                try_add(e)
                added = True
                # Restart loop to check for more adds?
                # Or just continue? If we add e, it might block others, but might not.
                # It's safe to continue.

        if added:
            if current_count > best_count:
                best_count = current_count
                best_selected = current_selected[:]
                best_face_count = current_face_count[:]
            # If we added an edge, we might have new add opportunities?
            # Actually, adding an edge only increases face counts, so it only
            # makes it *harder* for other edges to be added.
            # So no new add opportunities are created by an add.
            # We should break out of the "try add" loop if we added, 
            # because we need to do swaps to potentially enable more adds.
            # Wait, if we add e, we might not be able to add f anymore.
            # But we won't be able to add anything new that we couldn't before.
            # So, if we added, we should switch to swap mode.

        # 2. If no add, try swaps
        # Pick a random unselected edge e
        # Find its blockers
        # If blockers exist, pick one blocker r, remove r, add e.

        # To avoid getting stuck, we can try multiple random e's.

        # Let's pick a random unselected edge
        unselected = [e for e in range(num_edges) if not current_selected[e]]
        if not unselected:
            break  # Perfect? No, max is 308 < 448.

        e = random.choice(unselected)

        # Find blockers
        blockers = []
        for f in edge_faces[e]:
            if current_face_count[f] == 3:
                # The 3 selected edges in this face are blockers
                for r in face_edges_tuple[f]:
                    if current_selected[r] and r != e:
                        blockers.append(r)

        if not blockers:
            # Should have been caught by the add loop, but just in case
            try_add(e)
            if current_count > best_count:
                best_count = current_count
                best_selected = current_selected[:]
                best_face_count = current_face_count[:]
            continue

        # Pick a random blocker to remove
        r = random.choice(blockers)

        # Perform swap: remove r, add e
        # We must ensure that after removing r, e is addable.
        # Removing r decreases face_count for faces containing r.
        # e is addable if for all f in edge_faces[e], face_count[f] < 3.
        # Before swap, face_count[f] == 3 for some f.
        # If r is in all such f, then after removal, face_count[f] becomes 2, so e is addable.
        # If r is not in some "full" face f', then face_count[f'] remains 3, so e is NOT addable.

        # So we only perform the swap if r is in ALL faces f where face_count[f]==3 and e in f.

        valid_swap = True
        for f in edge_faces[e]:
            if current_face_count[f] == 3:
                if r not in face_edges_tuple[f]:
                    valid_swap = False
                    break

        if valid_swap:
            try_remove(r)
            try_add(e)
            # Size unchanged.
            # Check if this swap enables new adds?
            # Removing r might lower face counts, potentially making other edges addable.
            # We don't check immediately, the next "try add" loop will catch it.

    # Return best found
    result = [list(edges[i]) for i in range(num_edges) if best_selected[i]]
    return result
