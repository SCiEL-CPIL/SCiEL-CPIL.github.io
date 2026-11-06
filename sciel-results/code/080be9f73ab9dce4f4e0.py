import random
import time


def construct():
    start = time.time()
    deadline = start + 115.0

    d = 7
    n = 1 << d  # 128 vertices

    # Precompute all edges with indices
    all_edges = []
    edge_index = {}
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(all_edges)
                all_edges.append((u, v))
                edge_index[(u, v)] = idx

    E = len(all_edges)  # 448

    # Precompute for each edge: which faces (C4s) it belongs to
    # A face is identified by (u, i, j) where u is the corner with bits i,j = 0
    # Each face has 4 edges
    edge_faces = [[] for _ in range(E)]  # edge_idx -> list of face_ids
    face_edges = {}  # face_id -> list of 4 edge indices

    face_id = 0
    for u in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                # Face with corner u, directions i and j
                # But we need u to have bit i = 0 and bit j = 0
                if (u >> i) & 1 or (u >> j) & 1:
                    continue
                u1 = u ^ (1 << i)
                u2 = u ^ (1 << j)
                u12 = u ^ (1 << i) ^ (1 << j)
                e1 = (min(u, u1), max(u, u1))
                e2 = (min(u1, u12), max(u1, u12))
                e3 = (min(u12, u2), max(u12, u2))
                e4 = (min(u2, u), max(u2, u))
                eid1 = edge_index[e1]
                eid2 = edge_index[e2]
                eid3 = edge_index[e3]
                eid4 = edge_index[e4]
                face_edges[face_id] = [eid1, eid2, eid3, eid4]
                for eid in [eid1, eid2, eid3, eid4]:
                    edge_faces[eid].append(face_id)
                face_id += 1

    num_faces = face_id

    def greedy_random(rng):
        """Random greedy: shuffle edges, add if no face would be completed."""
        chosen = [False] * E
        face_count = [0] * num_faces
        order = list(range(E))
        rng.shuffle(order)
        for idx in order:
            # Check if adding this edge would complete any face
            can_add = True
            for fid in edge_faces[idx]:
                if face_count[fid] == 3:
                    can_add = False
                    break
            if can_add:
                chosen[idx] = True
                for fid in edge_faces[idx]:
                    face_count[fid] += 1
        return chosen, face_count

    def local_search(chosen, face_count, rng, time_limit):
        """Local search: try swaps and additions."""
        best_chosen = chosen[:]
        best_fc = face_count[:]
        best_size = sum(best_chosen)
        current_size = best_size

        t = 0.5
        t_min = 0.01
        cooling = 0.998

        while time.time() < time_limit:
            # Try to add any unchosen edge that doesn't violate
            improved = False
            for idx in range(E):
                if not chosen[idx]:
                    can_add = True
                    for fid in edge_faces[idx]:
                        if face_count[fid] == 3:
                            can_add = False
                            break
                    if can_add:
                        chosen[idx] = True
                        current_size += 1
                        for fid in edge_faces[idx]:
                            face_count[fid] += 1
                        improved = True

            if current_size > best_size:
                best_size = current_size
                best_chosen = chosen[:]
                best_fc = face_count[:]

            if time.time() >= time_limit:
                break

            # Try a swap: remove one chosen, add one unchosen
            if rng.random() < 0.7:
                chosen_idx = [i for i in range(E) if chosen[i]]
                unchosen_idx = [i for i in range(E) if not chosen[i]]
                if chosen_idx and unchosen_idx:
                    rem = chosen_idx[rng.randrange(len(chosen_idx))]
                    add = unchosen_idx[rng.randrange(len(unchosen_idx))]

                    # Check if removing rem and adding add is feasible
                    # First check: after removing rem, can we add add?
                    # Temporarily remove rem
                    for fid in edge_faces[rem]:
                        face_count[fid] -= 1

                    can_add = True
                    for fid in edge_faces[add]:
                        if face_count[fid] == 3:
                            can_add = False
                            break

                    if can_add:
                        # Accept the swap
                        chosen[rem] = False
                        chosen[add] = True
                        for fid in edge_faces[add]:
                            face_count[fid] += 1
                        # Size stays the same (removed one, added one)
                        # But maybe we can now add more
                        for idx2 in range(E):
                            if not chosen[idx2]:
                                can_add2 = True
                                for fid in edge_faces[idx2]:
                                    if face_count[fid] == 3:
                                        can_add2 = False
                                        break
                                if can_add2:
                                    chosen[idx2] = True
                                    current_size += 1
                                    for fid in edge_faces[idx2]:
                                        face_count[fid] += 1

                        if current_size > best_size:
                            best_size = current_size
                            best_chosen = chosen[:]
                            best_fc = face_count[:]
                    else:
                        # Revert
                        for fid in edge_faces[rem]:
                            face_count[fid] += 1

            # Try 2-for-2 swap
            elif rng.random() < 0.3:
                chosen_idx = [i for i in range(E) if chosen[i]]
                unchosen_idx = [i for i in range(E) if not chosen[i]]
                if len(chosen_idx) >= 2 and len(unchosen_idx) >= 2:
                    rem1 = chosen_idx[rng.randrange(len(chosen_idx))]
                    rem2 = chosen_idx[rng.randrange(len(chosen_idx))]
                    if rem1 == rem2:
                        rem2 = chosen_idx[rng.randrange(len(chosen_idx))]
                    add1 = unchosen_idx[rng.randrange(len(unchosen_idx))]
                    add2 = unchosen_idx[rng.randrange(len(unchosen_idx))]
                    if add1 == add2:
                        add2 = unchosen_idx[rng.randrange(len(unchosen_idx))]

                    # Remove both
                    for fid in edge_faces[rem1]:
                        face_count[fid] -= 1
                    for fid in edge_faces[rem2]:
                        face_count[fid] -= 1

                    # Check if both can be added
                    can_add1 = True
                    for fid in edge_faces[add1]:
                        if face_count[fid] == 3:
                            can_add1 = False
                            break
                    can_add2 = True
                    for fid in edge_faces[add2]:
                        if face_count[fid] == 3:
                            can_add2 = False
                            break

                    if can_add1 and can_add2:
                        chosen[rem1] = False
                        chosen[rem2] = False
                        chosen[add1] = True
                        chosen[add2] = True
                        for fid in edge_faces[add1]:
                            face_count[fid] += 1
                        for fid in edge_faces[add2]:
                            face_count[fid] += 1
                        # Size unchanged, but try to add more
                        for idx2 in range(E):
                            if not chosen[idx2]:
                                can_add2 = True
                                for fid in edge_faces[idx2]:
                                    if face_count[fid] == 3:
                                        can_add2 = False
                                        break
                                if can_add2:
                                    chosen[idx2] = True
                                    current_size += 1
                                    for fid in edge_faces[idx2]:
                                        face_count[fid] += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_chosen = chosen[:]
                            best_fc = face_count[:]
                    else:
                        # Revert
                        for fid in edge_faces[rem1]:
                            face_count[fid] += 1
                        for fid in edge_faces[rem2]:
                            face_count[fid] += 1

            # Cooling
            t = max(t * cooling, t_min)

            # Random kick if stuck
            if rng.random() < 0.03:
                kick_size = rng.randint(2, 5)
                chosen_idx = [i for i in range(E) if chosen[i]]
                if len(chosen_idx) > kick_size:
                    to_remove = rng.sample(chosen_idx, kick_size)
                    for r in to_remove:
                        chosen[r] = False
                        current_size -= 1
                        for fid in edge_faces[r]:
                            face_count[fid] -= 1
                    # Refill greedily
                    for idx in range(E):
                        if not chosen[idx]:
                            can_add = True
                            for fid in edge_faces[idx]:
                                if face_count[fid] == 3:
                                    can_add = False
                                    break
                            if can_add:
                                chosen[idx] = True
                                current_size += 1
                                for fid in edge_faces[idx]:
                                    face_count[fid] += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_chosen = chosen[:]
                        best_fc = face_count[:]

        return best_chosen, best_size

    best_chosen = None
    best_size = 0
    rng = random.Random(42)

    # Multiple random greedy starts + local search
    num_starts = 20
    for s in range(num_starts):
        if time.time() > deadline - 30:
            break

        chosen, fc = greedy_random(rng)
        size = sum(chosen)
        if size > best_size:
            best_chosen = chosen[:]
            best_size = size

        # Local search from this start
        time_limit = deadline - 25
        chosen, size = local_search(chosen, fc, rng, time_limit)
        if size > best_size:
            best_chosen = chosen[:]
            best_size = size

        if time.time() > deadline - 25:
            break

    # Final: try more local search on best
    if best_chosen and time.time() < deadline - 10:
        fc = [0] * num_faces
        for idx in range(E):
            if best_chosen[idx]:
                for fid in edge_faces[idx]:
                    fc[fid] += 1
        best_chosen, best_size = local_search(best_chosen, fc, rng, deadline - 5)

    # Convert to list of [u, v]
    out = []
    for idx in range(E):
        if best_chosen[idx]:
            u, v = all_edges[idx]
            out.append([u, v])
    return out
