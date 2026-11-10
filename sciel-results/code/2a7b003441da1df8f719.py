"""Structural rewrite: local search with fast face-conflict tracking."""
import time
import random
import math

def construct():
    deadline = time.time() + 115.0  # safety margin before 120s limit

    d = 7
    n = 1 << d  # 128 vertices
    total_edges = d * (1 << (d - 1))  # 448

    # Precompute all edges with unique indices
    edges = []  # list of (u, v) with u < v
    edge_index = {}  # (u,v) -> idx
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_index[(u, v)] = idx
                edge_index[(v, u)] = idx

    # For each edge, find all faces that contain it
    # Face identified by (x, i, j) where x < x^(1<<i) < x^(1<<i|1<<j)
    # Edges in face: (x, x^i), (x, x^j), (x^i, x^i^j), (x^j, x^i^j)
    edge_to_faces = [[] for _ in range(total_edges)]
    face_edges = []  # list of 4 edge indices per face
    face_id = {}  # (x, i, j) -> face_idx

    for x in range(n):
        for i in range(d):
            xi = x ^ (1 << i)
            for j in range(i + 1, d):
                xj = x ^ (1 << j)
                xij = x ^ (1 << i) ^ (1 << j)
                # The 4 vertices of the face
                # Edges: (x,xi), (x,xj), (xi,xij), (xj,xij)
                e1 = edge_index[(min(x, xi), max(x, xi))]
                e2 = edge_index[(min(x, xj), max(x, xj))]
                e3 = edge_index[(min(xi, xij), max(xi, xij))]
                e4 = edge_index[(min(xj, xij), max(xj, xij))]
                face_idx = len(face_edges)
                face_edges.append((e1, e2, e3, e4))
                for e in (e1, e2, e3, e4):
                    edge_to_faces[e].append(face_idx)

    num_faces = len(face_edges)  # C(7,2) * 128 / something... let's compute
    # Actually: for each x (128), C(7,2)=21 pairs -> 2688 faces total
    # But each face is counted 4 times (once per vertex), so 2688/4 = 672 unique faces

    def run_local_search(initial_edges, time_limit):
        """Run local search starting from initial edge set."""
        selected = [False] * total_edges
        face_count = [0] * num_faces  # number of selected edges in each face

        for e in initial_edges:
            selected[e] = True
            for f in edge_to_faces[e]:
                face_count[f] += 1

        best_count = sum(selected)
        best_set = set(initial_edges)

        # Local search: try to add random unselected edges, remove if needed
        rng = random.Random(42)
        unselected = [e for e in range(total_edges) if not selected[e]]

        while time.time() < time_limit:
            # Try adding a random unselected edge
            if not unselected:
                break
            e = rng.choice(unselected)
            if selected[e]:
                unselected.remove(e)
                continue

            # Check if adding e creates any face with count=4
            can_add = True
            for f in edge_to_faces[e]:
                if face_count[f] >= 3:
                    can_add = False
                    break

            if can_add:
                selected[e] = True
                for f in edge_to_faces[e]:
                    face_count[f] += 1
                unselected.remove(e)
                cur_count = sum(selected)
                if cur_count > best_count:
                    best_count = cur_count
                    best_set = set(selected.index(True) for _ in range(total_edges) if selected[_])
                    # More efficient:
                    best_set = set(e2 for e2 in range(total_edges) if selected[e2])
            else:
                # Try to remove one edge from a conflicting face, then add e
                # Find a face where count>=3 and one of its edges is not e
                for f in edge_to_faces[e]:
                    if face_count[f] >= 3:
                        # Remove a random edge from this face (not e)
                        candidates = [fe for fe in face_edges[f] if selected[fe] and fe != e]
                        if candidates:
                            rem = rng.choice(candidates)
                            selected[rem] = False
                            for ff in edge_to_faces[rem]:
                                face_count[ff] -= 1
                            unselected.append(rem)
                            # Now try to add e
                            can_add2 = True
                            for ff in edge_to_faces[e]:
                                if face_count[ff] >= 3:
                                    can_add2 = False
                                    break
                            if can_add2:
                                selected[e] = True
                                for ff in edge_to_faces[e]:
                                    face_count[ff] += 1
                                unselected.remove(e)
                                cur_count = sum(selected)
                                if cur_count > best_count:
                                    best_count = cur_count
                                    best_set = set(e2 for e2 in range(total_edges) if selected[e2])
                        break

        return best_set, best_count

    # Phase 1: Greedy initialization with randomization
    best_overall = 0
    best_result = None

    # Run greedy with different random orderings
    for trial in range(200):
        if time.time() > deadline:
            break
        rng = random.Random(trial * 12345 + 678)
        order = list(range(total_edges))
        rng.shuffle(order)

        selected = [False] * total_edges
        face_count = [0] * num_faces
        count = 0

        for e in order:
            can_add = True
            for f in edge_to_faces[e]:
                if face_count[f] >= 3:
                    can_add = False
                    break
            if can_add:
                selected[e] = True
                for f in edge_to_faces[e]:
                    face_count[f] += 1
                count += 1

        if count > best_overall:
            best_overall = count
            init_set = set(e for e in range(total_edges) if selected[e])
            if time.time() < deadline - 10:
                # Try to improve with local search
                ls_set, ls_count = run_local_search(list(init_set), deadline - 5)
                if ls_count > best_overall:
                    best_overall = ls_count
                    best_result = ls_set
            else:
                best_result = init_set

    # Phase 2: More local search from best found
    if best_result is not None and time.time() < deadline - 10:
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            start_set = list(best_result)
            # Perturb: randomly remove some edges
            rng = random.Random(999 + trial)
            perturbed = set(start_set)
            num_remove = rng.randint(5, 20)
            to_remove = rng.sample(list(perturbed), min(num_remove, len(perturbed)))
            for e in to_remove:
                perturbed.discard(e)
            ls_set, ls_count = run_local_search(list(perturbed), deadline - 5)
            if ls_count > best_overall:
                best_overall = ls_count
                best_result = ls_set

    if best_result is None:
        # Fallback: return empty or minimal valid set
        best_result = set()

    # Convert to list of [u, v]
    result = []
    for e in sorted(best_result):
        u, v = edges[e]
        result.append([u, v])

    return result
