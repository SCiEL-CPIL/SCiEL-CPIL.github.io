import random
import time
import numpy as np


def construct():
    start = time.time()
    deadline = start + 118.0  # safety margin before 150s kill

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

    # Precompute faces (2D squares) and edge-to-face mapping
    face_edges = []  # list of lists of 4 edge indices
    edge_to_faces = [[] for _ in range(E)]

    for i in range(d):
        for j in range(i + 1, d):
            for u in range(n):
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)

                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v4), max(v2, v4))
                e4 = (min(v3, v4), max(v3, v4))

                idxs = [edge_index[e1], edge_index[e2], edge_index[e3], edge_index[e4]]
                # Deduplicate: each face is counted 4 times (once per corner)
                # Use frozenset to deduplicate
                key = frozenset(idxs)
                if not any(frozenset(f) == key for f in face_edges[-4:]):  # quick check
                    # Better: use a set
                    pass

    # Redo face enumeration properly with deduplication
    face_edges = []
    seen_faces = set()
    for i in range(d):
        for j in range(i + 1, d):
            for u in range(n):
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)

                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v4), max(v2, v4))
                e4 = (min(v3, v4), max(v3, v4))

                idxs = tuple(sorted([edge_index[e1], edge_index[e2], edge_index[e3], edge_index[e4]]))
                if idxs not in seen_faces:
                    seen_faces.add(idxs)
                    face_edges.append(list(idxs))

    num_faces = len(face_edges)

    # Rebuild edge_to_faces
    edge_to_faces = [[] for _ in range(E)]
    for f_idx, face in enumerate(face_edges):
        for e in face:
            edge_to_faces[e].append(f_idx)

    def popcount(x):
        return bin(x).count("1")

    def greedy_random(rng):
        """Random greedy: shuffle edges, add if not blocked."""
        chosen_mask = 0
        face_counts = np.full(num_faces, 4, dtype=np.int32)
        order = list(range(E))
        rng.shuffle(order)
        for idx in order:
            # Check if adding idx would complete any face
            blocked = False
            for f_idx in edge_to_faces[idx]:
                if face_counts[f_idx] == 3:
                    blocked = True
                    break
            if not blocked:
                chosen_mask |= (1 << idx)
                for f_idx in edge_to_faces[idx]:
                    face_counts[f_idx] -= 1
        return chosen_mask

    def greedy_fill(chosen_mask, face_counts):
        """Add any unblocked edges; returns updated mask and face_counts."""
        for idx in range(E):
            if not (chosen_mask >> idx) & 1:
                blocked = False
                for f_idx in edge_to_faces[idx]:
                    if face_counts[f_idx] == 3:
                        blocked = True
                        break
                if not blocked:
                    chosen_mask |= (1 << idx)
                    for f_idx in edge_to_faces[idx]:
                        face_counts[f_idx] -= 1
        return chosen_mask, face_counts

    def simulated_annealing(chosen_mask, face_counts, rng, time_limit):
        """
        SA with proper save/restore: try to add unblocked edges,
        and perform 1-for-1 swaps with occasional non-improving moves.
        """
        best_mask = chosen_mask
        best_size = popcount(chosen_mask)
        current_size = best_size

        t = 0.8
        t_min = 0.02
        cooling = 0.995

        while time.time() < time_limit:
            # Phase 1: greedy add any unblocked edges
            chosen_mask, face_counts = greedy_fill(chosen_mask, face_counts)
            current_size = popcount(chosen_mask)

            if current_size > best_size:
                best_size = current_size
                best_mask = chosen_mask

            if time.time() >= time_limit:
                break

            # Phase 2: try a swap - save state before attempting
            chosen_list = [i for i in range(E) if (chosen_mask >> i) & 1]
            unchosen_list = [i for i in range(E) if not ((chosen_mask >> i) & 1)]
            if not unchosen_list:
                break
            rng.shuffle(chosen_list)
            rng.shuffle(unchosen_list)

            found_improve = False
            for _ in range(20):
                if time.time() >= time_limit:
                    break
                if not chosen_list or not unchosen_list:
                    break

                rem = chosen_list[rng.randrange(len(chosen_list))]
                add = unchosen_list[rng.randrange(len(unchosen_list))]

                # Save state
                saved_mask = chosen_mask
                saved_face_counts = face_counts.copy()

                # Remove rem
                trial_mask = chosen_mask & ~(1 << rem)
                trial_fc = face_counts.copy()
                for f_idx in edge_to_faces[rem]:
                    trial_fc[f_idx] += 1

                # Check if add is now unblocked
                blocked = False
                for f_idx in edge_to_faces[add]:
                    if trial_fc[f_idx] == 3:
                        blocked = True
                        break

                if not blocked:
                    trial_mask |= (1 << add)
                    for f_idx in edge_to_faces[add]:
                        trial_fc[f_idx] -= 1

                    # Greedy refill on trial
                    trial_mask, trial_fc = greedy_fill(trial_mask, trial_fc)
                    new_size = popcount(trial_mask)
                    delta = new_size - current_size

                    if delta >= 0:
                        # Accept
                        chosen_mask = trial_mask
                        face_counts = trial_fc
                        current_size = new_size
                        if new_size > best_size:
                            best_size = new_size
                            best_mask = chosen_mask
                        found_improve = True
                        chosen_list = [i for i in range(E) if (chosen_mask >> i) & 1]
                        unchosen_list = [i for i in range(E) if not ((chosen_mask >> i) & 1)]
                        break
                    else:
                        # Metropolis accept
                        if rng.random() < pow(2.718281828, delta / max(t, 1e-9)):
                            chosen_mask = trial_mask
                            face_counts = trial_fc
                            current_size = new_size
                            chosen_list = [i for i in range(E) if (chosen_mask >> i) & 1]
                            unchosen_list = [i for i in range(E) if not ((chosen_mask >> i) & 1)]
                            break
                # If blocked or rejected, state is already saved so we continue

            if found_improve:
                t = max(t * cooling, t_min)
                continue

            # Cool down
            t = max(t * cooling, t_min)

            # If stuck, do a random perturbation
            if rng.random() < 0.05:
                chosen_list = [i for i in range(E) if (chosen_mask >> i) & 1]
                if len(chosen_list) > 10:
                    to_remove = rng.sample(chosen_list, min(3, len(chosen_list) // 10))
                    for r in to_remove:
                        chosen_mask &= ~(1 << r)
                        for f_idx in edge_to_faces[r]:
                            face_counts[f_idx] += 1
                    chosen_mask, face_counts = greedy_fill(chosen_mask, face_counts)
                    current_size = popcount(chosen_mask)

        return best_mask, best_size

    def two_opt_search(chosen_mask, face_counts, rng, time_limit):
        """
        Two-opt: remove two edges, add two different edges.
        """
        best_mask = chosen_mask
        best_size = popcount(chosen_mask)
        current_size = best_size

        while time.time() < time_limit:
            # Greedy fill
            chosen_mask, face_counts = greedy_fill(chosen_mask, face_counts)
            current_size = popcount(chosen_mask)

            if current_size > best_size:
                best_size = current_size
                best_mask = chosen_mask

            if time.time() >= time_limit:
                break

            unchosen_list = [i for i in range(E) if not ((chosen_mask >> i) & 1)]
            if len(unchosen_list) < 2:
                break
            rng.shuffle(unchosen_list)

            improved = False
            for e1_idx in unchosen_list[:80]:
                if time.time() >= time_limit:
                    break

                # Check blocking faces for e1
                blocking1 = [f_idx for f_idx in edge_to_faces[e1_idx] if face_counts[f_idx] == 3]

                if not blocking1:
                    chosen_mask |= (1 << e1_idx)
                    for f_idx in edge_to_faces[e1_idx]:
                        face_counts[f_idx] -= 1
                    current_size += 1
                    improved = True
                    continue

                for e2_idx in unchosen_list[:80]:
                    if e2_idx == e1_idx:
                        continue
                    if time.time() >= time_limit:
                        break

                    blocking2 = [f_idx for f_idx in edge_to_faces[e2_idx] if face_counts[f_idx] == 3]

                    if not blocking2:
                        chosen_mask |= (1 << e2_idx)
                        for f_idx in edge_to_faces[e2_idx]:
                            face_counts[f_idx] -= 1
                        current_size += 1
                        improved = True
                        continue

                    # Try removing one edge from blocking1 and one from blocking2
                    if len(blocking1) == 1 and len(blocking2) == 1:
                        f1 = blocking1[0]
                        f2 = blocking2[0]
                        face1 = face_edges[f1]
                        face2 = face_edges[f2]

                        for rem1 in face1:
                            if rem1 == e1_idx:
                                continue
                            for rem2 in face2:
                                if rem2 == e2_idx or rem2 == rem1:
                                    continue

                                # Save state
                                saved_mask = chosen_mask
                                saved_fc = face_counts.copy()

                                # Remove rem1 and rem2
                                trial_mask = chosen_mask
                                trial_fc = face_counts.copy()
                                for f_idx in edge_to_faces[rem1]:
                                    trial_fc[f_idx] += 1
                                trial_mask &= ~(1 << rem1)
                                for f_idx in edge_to_faces[rem2]:
                                    trial_fc[f_idx] += 1
                                trial_mask &= ~(1 << rem2)

                                # Check if e1 and e2 are now unblocked
                                ok1 = all(trial_fc[f_idx] < 3 for f_idx in edge_to_faces[e1_idx])
                                ok2 = all(trial_fc[f_idx] < 3 for f_idx in edge_to_faces[e2_idx])

                                if ok1 and ok2:
                                    trial_mask |= (1 << e1_idx)
                                    for f_idx in edge_to_faces[e1_idx]:
                                        trial_fc[f_idx] -= 1
                                    trial_mask |= (1 << e2_idx)
                                    for f_idx in edge_to_faces[e2_idx]:
                                        trial_fc[f_idx] -= 1

                                    # Greedy refill
                                    trial_mask, trial_fc = greedy_fill(trial_mask, trial_fc)
                                    new_size = popcount(trial_mask)
                                    if new_size > current_size:
                                        chosen_mask = trial_mask
                                        face_counts = trial_fc
                                        current_size = new_size
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break

            if current_size > best_size:
                best_size = current_size
                best_mask = chosen_mask

            if not improved:
                # Random perturbation
                if time.time() < time_limit - 5:
                    chosen_list = [i for i in range(E) if (chosen_mask >> i) & 1]
                    if len(chosen_list) > 10:
                        to_remove = rng.sample(chosen_list, min(3, len(chosen_list) // 10))
                        for r in to_remove:
                            chosen_mask &= ~(1 << r)
                            for f_idx in edge_to_faces[r]:
                                face_counts[f_idx] += 1
                        chosen_mask, face_counts = greedy_fill(chosen_mask, face_counts)
                        current_size = popcount(chosen_mask)

        return best_mask, best_size

    best_mask = 0
    best_size = 0
    rng = random.Random(42)

    # Multiple random greedy starts + SA + two-opt
    num_starts = 12
    for s in range(num_starts):
        if time.time() > deadline - 50:
            break

        chosen_mask = greedy_random(rng)
        # Rebuild face_counts for this mask
        face_counts = np.full(num_faces, 4, dtype=np.int32)
        for idx in range(E):
            if (chosen_mask >> idx) & 1:
                for f_idx in edge_to_faces[idx]:
                    face_counts[f_idx] -= 1

        size = popcount(chosen_mask)
        if size > best_size:
            best_mask = chosen_mask
            best_size = size

        # SA from this start
        time_limit = deadline - 40
        chosen_mask, size = simulated_annealing(chosen_mask, face_counts, rng, time_limit)
        if size > best_size:
            best_mask = chosen_mask
            best_size = size

        # Rebuild face_counts for two-opt
        face_counts = np.full(num_faces, 4, dtype=np.int32)
        for idx in range(E):
            if (chosen_mask >> idx) & 1:
                for f_idx in edge_to_faces[idx]:
                    face_counts[f_idx] -= 1

        # Two-opt search
        if time.time() < deadline - 30:
            chosen_mask, size = two_opt_search(chosen_mask, face_counts, rng, deadline - 25)
            if size > best_size:
                best_mask = chosen_mask
                best_size = size

        if time.time() > deadline - 25:
            break

    # Final greedy fill on best
    if time.time() < deadline - 5:
        face_counts = np.full(num_faces, 4, dtype=np.int32)
        for idx in range(E):
            if (best_mask >> idx) & 1:
                for f_idx in edge_to_faces[idx]:
                    face_counts[f_idx] -= 1
        best_mask, _ = greedy_fill(best_mask, face_counts)

    # Convert to list of [u, v]
    out = []
    for idx in range(E):
        if (best_mask >> idx) & 1:
            u, v = all_edges[idx]
            out.append([u, v])
    return out
