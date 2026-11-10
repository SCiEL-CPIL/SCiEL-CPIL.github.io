import random
import time
import numpy as np

def construct():
    deadline = time.time() + 115
    d = 7
    n = 1 << d  # 128 vertices
    best_count = 0
    best_chosen = None

    # Build all edges
    all_edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v, i))

    num_edges = len(all_edges)  # 448

    # Map edge tuple to index
    edge_tuple_to_idx = {}
    for idx, (u, v, i) in enumerate(all_edges):
        edge_tuple_to_idx[(u, v)] = idx

    # Build faces: 672 faces, each with 4 edge indices
    faces = []
    for i in range(d):
        for j in range(i + 1, d):
            for mask in range(1 << d):
                if (mask & (1 << i)) or (mask & (1 << j)):
                    continue
                v1 = mask
                v2 = mask ^ (1 << i)
                v3 = mask ^ (1 << j)
                v4 = mask ^ (1 << i) ^ (1 << j)
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v2, v4), max(v2, v4))
                e3 = (min(v4, v3), max(v4, v3))
                e4 = (min(v3, v1), max(v3, v1))
                idx1 = edge_tuple_to_idx[e1]
                idx2 = edge_tuple_to_idx[e2]
                idx3 = edge_tuple_to_idx[e3]
                idx4 = edge_tuple_to_idx[e4]
                faces.append((idx1, idx2, idx3, idx4))

    num_faces = len(faces)  # 672

    # Map edge index to list of face indices
    edge_to_faces = [[] for _ in range(num_edges)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_to_faces[e_idx].append(f_idx)

    # Precompute face as numpy arrays for speed
    faces_np = np.array(faces, dtype=np.int32)  # shape (672, 4)

    def structured_init(seed):
        """Create initial solution by greedily adding edges in a structured order."""
        rng = random.Random(seed)
        chosen = np.zeros(num_edges, dtype=np.bool_)
        face_counts = np.zeros(num_faces, dtype=np.int32)
        count = 0

        # Try multiple orderings: by dimension, random, etc.
        order = list(range(num_edges))

        # Strategy: process edges grouped by dimension, with random order within groups
        by_dim = [[] for _ in range(d)]
        for idx in range(num_edges):
            dim = all_edges[idx][2]
            by_dim[dim].append(idx)

        rng.shuffle(by_dim)  # shuffle dimension order
        for dim_list in by_dim:
            rng.shuffle(dim_list)
            order.extend(dim_list)

        for idx in order:
            can_add = True
            for f_idx in edge_to_faces[idx]:
                if face_counts[f_idx] == 3:
                    can_add = False
                    break
            if can_add:
                chosen[idx] = True
                count += 1
                for f_idx in edge_to_faces[idx]:
                    face_counts[f_idx] += 1

        return count, chosen, face_counts

    def simulated_annealing(count, chosen, face_counts, deadline, seed):
        """Simulated annealing with add/remove/swap moves."""
        rng = random.Random(seed)
        best_count = count
        best_chosen = chosen.copy()
        best_face_counts = face_counts.copy()

        # Temperature schedule
        t0 = 1.0
        t_end = 0.01
        cooling = 0.995

        temperature = t0
        stuck = 0

        while time.time() < deadline and temperature > t_end:
            # Pick a random move type
            move_type = rng.random()
            improved_this_step = False

            if move_type < 0.3:
                # Try to add a random unchosen edge
                unchosen = np.where(~chosen)[0]
                if len(unchosen) == 0:
                    continue
                idx = unchosen[rng.randint(0, len(unchosen) - 1)]
                # Check if can add
                can_add = True
                for f_idx in edge_to_faces[idx]:
                    if face_counts[f_idx] == 3:
                        can_add = False
                        break
                if can_add:
                    chosen[idx] = True
                    count += 1
                    for f_idx in edge_to_faces[idx]:
                        face_counts[f_idx] += 1
                    improved_this_step = True
                # else: reject (no change)

            elif move_type < 0.5:
                # Try to remove a random chosen edge (to unblock others)
                chosen_edges = np.where(chosen)[0]
                if len(chosen_edges) == 0:
                    continue
                idx = chosen_edges[rng.randint(0, len(chosen_edges) - 1)]
                chosen[idx] = False
                count -= 1
                for f_idx in edge_to_faces[idx]:
                    face_counts[f_idx] -= 1
                improved_this_step = True  # count decreased, but may enable future adds

            else:
                # Try swap: remove one chosen, add one unchosen
                chosen_edges = np.where(chosen)[0]
                unchosen = np.where(~chosen)[0]
                if len(chosen_edges) == 0 or len(unchosen) == 0:
                    continue
                r_idx = chosen_edges[rng.randint(0, len(chosen_edges) - 1)]
                a_idx = unchosen[rng.randint(0, len(unchosen) - 1)]

                # Remove r_idx
                chosen[r_idx] = False
                count -= 1
                for f_idx in edge_to_faces[r_idx]:
                    face_counts[f_idx] -= 1

                # Check if a_idx can be added
                can_add = True
                for f_idx in edge_to_faces[a_idx]:
                    if face_counts[f_idx] == 3:
                        can_add = False
                        break

                if can_add:
                    chosen[a_idx] = True
                    count += 1
                    for f_idx in edge_to_faces[a_idx]:
                        face_counts[f_idx] += 1
                    improved_this_step = True
                else:
                    # Revert
                    chosen[r_idx] = True
                    count += 1
                    for f_idx in edge_to_faces[r_idx]:
                        face_counts[f_idx] += 1

            if count > best_count:
                best_count = count
                best_chosen = chosen.copy()
                best_face_counts = face_counts.copy()
                stuck = 0
            else:
                stuck += 1

            # Cooling
            temperature *= cooling

            # If stuck too long, reset temperature slightly
            if stuck > 200:
                temperature = max(temperature, 0.5)
                stuck = 0

        return best_count, best_chosen, best_face_counts

    def local_search_remove_add2(count, chosen, face_counts, deadline):
        """Local search: try remove 1, add 2 moves."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for r_idx in range(num_edges):
                if not chosen[r_idx]:
                    continue

                # Tentatively remove r_idx
                chosen[r_idx] = False
                count -= 1
                for f_idx in edge_to_faces[r_idx]:
                    face_counts[f_idx] -= 1

                # Collect candidate edges: unchosen edges sharing a face with r_idx
                candidate_set = set()
                for f_idx in edge_to_faces[r_idx]:
                    for e_idx in faces[f_idx]:
                        if e_idx != r_idx and not chosen[e_idx]:
                            candidate_set.add(e_idx)

                candidates = []
                for e_idx in candidate_set:
                    valid = True
                    for f_idx in edge_to_faces[e_idx]:
                        if face_counts[f_idx] == 3:
                            valid = False
                            break
                    if valid:
                        candidates.append(e_idx)

                # Try "remove 1, add 2" move
                added = []
                for i in range(len(candidates)):
                    e1 = candidates[i]
                    for j in range(i + 1, len(candidates)):
                        e2 = candidates[j]
                        compatible = True
                        for f_idx in edge_to_faces[e1]:
                            if e2 in faces[f_idx]:
                                if face_counts[f_idx] >= 2:
                                    compatible = False
                                    break
                        if compatible:
                            added.append(e1)
                            added.append(e2)
                            break
                    if len(added) >= 2:
                        break

                if len(added) >= 2:
                    for e_idx in added:
                        chosen[e_idx] = True
                        count += 1
                        for f_idx in edge_to_faces[e_idx]:
                            face_counts[f_idx] += 1
                    improved = True
                    break

                # Restore r_idx
                chosen[r_idx] = True
                count += 1
                for f_idx in edge_to_faces[r_idx]:
                    face_counts[f_idx] += 1

        return count, chosen, face_counts

    attempt = 0
    while time.time() < deadline:
        # Different initialization strategies
        if attempt % 4 == 0:
            count, chosen, face_counts = structured_init(attempt)
        elif attempt % 4 == 1:
            count, chosen, face_counts = structured_init(attempt + 100)
        elif attempt % 4 == 2:
            count, chosen, face_counts = structured_init(attempt + 200)
        else:
            count, chosen, face_counts = structured_init(attempt + 300)

        # Simulated annealing
        count, chosen, face_counts = simulated_annealing(count, chosen, face_counts, deadline, attempt + 500)

        # Local search polish
        count, chosen, face_counts = local_search_remove_add2(count, chosen, face_counts, deadline)

        if count > best_count:
            best_count = count
            best_chosen = chosen
        attempt += 1

    out = []
    for idx in range(num_edges):
        if best_chosen[idx]:
            u, v, i = all_edges[idx]
            out.append([u, v])
    return out
