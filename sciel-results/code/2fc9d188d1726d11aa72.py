import random
import time
import math


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    d = 7
    n = 1 << d  # 128 vertices

    # Precompute all edges
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

    # Precompute faces and edge-face incidence
    edge_faces = [[] for _ in range(E)]
    face_edges = {}
    face_id = 0
    for u in range(n):
        for i in range(d):
            for j in range(i + 1, d):
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

    def greedy_random(rng, bias=None):
        """Random greedy with optional edge bias (higher = more likely early)."""
        chosen = [False] * E
        face_count = [0] * num_faces
        order = list(range(E))
        if bias is None:
            rng.shuffle(order)
        else:
            # Sort by bias descending with random tie-breaking
            scored = [(bias[i] + rng.random() * 0.1, i) for i in range(E)]
            scored.sort(key=lambda x: -x[0])
            order = [i for _, i in scored]
        for idx in order:
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

    def compute_bias(chosen, face_count):
        """Compute a bias score for each edge based on how 'free' it is."""
        bias = [0.0] * E
        for idx in range(E):
            if chosen[idx]:
                bias[idx] = -1.0
                continue
            # Count how many faces are blocked (face_count == 3)
            blocked = 0
            for fid in edge_faces[idx]:
                if face_count[fid] == 3:
                    blocked += 1
            bias[idx] = -blocked  # prefer edges with fewer blocked faces
        return bias

    def local_search(chosen, face_count, rng, time_limit, temperature_start=1.0):
        """Simulated annealing local search with 1-swap and 2-swap moves."""
        best_chosen = chosen[:]
        best_fc = face_count[:]
        best_size = sum(best_chosen)
        current_size = best_size

        t = temperature_start
        t_min = 0.001
        cooling = 0.997

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

            # 1-swap move
            chosen_idx = [i for i in range(E) if chosen[i]]
            unchosen_idx = [i for i in range(E) if not chosen[i]]
            if chosen_idx and unchosen_idx:
                rem = chosen_idx[rng.randrange(len(chosen_idx))]
                add = unchosen_idx[rng.randrange(len(unchosen_idx))]

                # Check feasibility after removing rem
                for fid in edge_faces[rem]:
                    face_count[fid] -= 1

                can_add = True
                for fid in edge_faces[add]:
                    if face_count[fid] == 3:
                        can_add = False
                        break

                if can_add:
                    # Accept if improves or with probability exp(-delta/t)
                    # Size stays same, so delta = 0, always accept
                    chosen[rem] = False
                    chosen[add] = True
                    for fid in edge_faces[add]:
                        face_count[fid] += 1
                    # Try to add more
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
                    for fid in edge_faces[rem]:
                        face_count[fid] += 1

            # 2-swap move
            elif rng.random() < 0.4:
                if len(chosen_idx) >= 2 and len(unchosen_idx) >= 2:
                    rem1 = chosen_idx[rng.randrange(len(chosen_idx))]
                    rem2 = chosen_idx[rng.randrange(len(chosen_idx))]
                    if rem1 == rem2:
                        rem2 = chosen_idx[rng.randrange(len(chosen_idx))]
                    add1 = unchosen_idx[rng.randrange(len(unchosen_idx))]
                    add2 = unchosen_idx[rng.randrange(len(unchosen_idx))]
                    if add1 == add2:
                        add2 = unchosen_idx[rng.randrange(len(unchosen_idx))]

                    for fid in edge_faces[rem1]:
                        face_count[fid] -= 1
                    for fid in edge_faces[rem2]:
                        face_count[fid] -= 1

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
                        for fid in edge_faces[rem1]:
                            face_count[fid] += 1
                        for fid in edge_faces[rem2]:
                            face_count[fid] += 1

            # Random restart kick
            if rng.random() < 0.02:
                kick_size = rng.randint(3, 7)
                chosen_idx_now = [i for i in range(E) if chosen[i]]
                if len(chosen_idx_now) > kick_size:
                    to_remove = rng.sample(chosen_idx_now, kick_size)
                    for r in to_remove:
                        chosen[r] = False
                        current_size -= 1
                        for fid in edge_faces[r]:
                            face_count[fid] -= 1
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

            t = max(t * cooling, t_min)

        return best_chosen, best_size

    best_chosen = None
    best_size = 0
    rng = random.Random(12345)

    # Phase 1: Multiple diverse greedy starts
    num_starts = 15
    for s in range(num_starts):
        if time.time() > deadline - 40:
            break

        # Vary the bias to explore different regions
        if s % 3 == 0:
            bias = None
        elif s % 3 == 1:
            # Prefer edges in lower-dimensional subcubes
            bias = [0.0] * E
            for idx in range(E):
                u, v = all_edges[idx]
                # Count set bits
                bias[idx] = bin(u).count('1') - 3.5
        else:
            # Random bias
            bias = [rng.gauss(0, 1) for _ in range(E)]

        chosen, fc = greedy_random(rng, bias)
        size = sum(chosen)
        if size > best_size:
            best_chosen = chosen[:]
            best_size = size

        # Local search
        time_limit = deadline - 35
        chosen, size = local_search(chosen, fc, rng, time_limit)
        if size > best_size:
            best_chosen = chosen[:]
            best_size = size

        if time.time() > deadline - 30:
            break

    # Phase 2: Polish the best solution with more intensive local search
    if best_chosen and time.time() < deadline - 15:
        fc = [0] * num_faces
        for idx in range(E):
            if best_chosen[idx]:
                for fid in edge_faces[idx]:
                    fc[fid] += 1
        best_chosen, best_size = local_search(best_chosen, fc, rng, deadline - 5, temperature_start=0.5)

    # Final greedy fill
    if best_chosen:
        fc = [0] * num_faces
        for idx in range(E):
            if best_chosen[idx]:
                for fid in edge_faces[idx]:
                    fc[fid] += 1
        for idx in range(E):
            if not best_chosen[idx]:
                can_add = True
                for fid in edge_faces[idx]:
                    if fc[fid] == 3:
                        can_add = False
                        break
                if can_add:
                    best_chosen[idx] = True
                    best_size += 1
                    for fid in edge_faces[idx]:
                        fc[fid] += 1

    # Convert to list of [u, v]
    out = []
    for idx in range(E):
        if best_chosen[idx]:
            u, v = all_edges[idx]
            out.append([u, v])
    return out
