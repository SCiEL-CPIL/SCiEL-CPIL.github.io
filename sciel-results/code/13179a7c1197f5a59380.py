import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin under 150s kill

    d = 7
    n = 1 << d
    num_edges = d * (1 << (d - 1))  # 448

    # 1. Precompute edges
    idx_to_edge = []
    edge_to_idx = {}
    idx = 0
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = idx
                idx_to_edge.append((u, v))
                idx += 1

    # 2. Precompute faces (each 2D face = 4 edges)
    seen_faces = set()
    face_edge_list = []
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
                idxs = [edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]]
                key = frozenset(idxs)
                if key not in seen_faces:
                    seen_faces.add(key)
                    face_edge_list.append(idxs)

    num_faces = len(face_edge_list)

    # Precompute edge -> faces mapping
    edge_to_faces = [[] for _ in range(num_edges)]
    for f_idx, face in enumerate(face_edge_list):
        for e in face:
            edge_to_faces[e].append(f_idx)

    # 3. Conflict graph adjacency (edges share a face) -> for spectral init
    # Build sparse conflict graph: edge e conflicts with edge e' if they share a face
    # We'll build the Laplacian implicitly via face incidence.
    # Incidence: face f has 4 edges. Conflict degree of e = sum over faces of (4-1)=3 * (#faces e in)
    # Build adjacency matrix in CSR-like form using numpy.
    # adjacency[e][e'] = number of shared faces (0 or 1, since two edges share at most one face)
    adj = np.zeros((num_edges, num_edges), dtype=np.int8)
    for face in face_edge_list:
        for a in range(4):
            for b in range(a + 1, 4):
                adj[face[a], face[b]] = 1
                adj[face[b], face[a]] = 1

    # 4. Spectral initialization: rank edges by eigenvector centrality of conflict graph.
    # Remove edges with LOW centrality first (they are "peripheral" and cheaper to drop).
    def spectral_init(rng, time_limit):
        """Greedy removal ordered by spectral centrality (low first), then local repair."""
        active = np.ones(num_edges, dtype=np.uint8)
        face_counts = np.full(num_faces, 4, dtype=np.int32)

        # Compute eigenvector centrality of conflict graph (use power iteration, cheap).
        # For a 448x448 sparse-ish matrix, a few power iterations suffice for ordering.
        vec = np.ones(num_edges, dtype=np.float64)
        vec /= np.linalg.norm(vec)
        for _ in range(40):
            new_vec = adj.astype(np.float64) @ vec
            norm = np.linalg.norm(new_vec)
            if norm == 0:
                break
            new_vec /= norm
            if np.linalg.norm(new_vec - vec) < 1e-9:
                vec = new_vec
                break
            vec = new_vec
        centrality = vec

        # Order edges by ascending centrality (remove peripheral first)
        order = np.argsort(centrality)

        # Greedy: walk through order, remove edge if it is in any all-4 face
        for e in order:
            if time.time() > time_limit:
                break
            # Check if e is in any face with count 4
            in_conflict = False
            for f_idx in edge_to_faces[e]:
                if face_counts[f_idx] == 4:
                    in_conflict = True
                    break
            if in_conflict:
                active[e] = 0
                for f_idx in edge_to_faces[e]:
                    face_counts[f_idx] -= 1

        return active, face_counts

    # 5. Classic greedy hitting set (fallback / diversity)
    def greedy_hit(rng):
        active = np.ones(num_edges, dtype=np.uint8)
        face_counts = np.full(num_faces, 4, dtype=np.int32)
        while True:
            active_faces = np.where(face_counts == 4)[0]
            if len(active_faces) == 0:
                break
            scores = np.zeros(num_edges, dtype=np.int32)
            for f_idx in active_faces:
                for e in face_edge_list[f_idx]:
                    scores[e] += 1
            max_score = np.max(scores)
            if max_score == 0:
                break
            candidates = np.where(scores == max_score)[0]
            # tie-break: pick edge with most active-face degree
            best_e = candidates[0]
            best_deg = -1
            for e in candidates:
                deg = sum(1 for ff in edge_to_faces[e] if face_counts[ff] == 4)
                if deg > best_deg:
                    best_deg = deg
                    best_e = e
            active[best_e] = 0
            for f_idx in edge_to_faces[best_e]:
                face_counts[f_idx] -= 1
        return active, face_counts

    # 6. Helpers
    def count_violations(active):
        v = 0
        for face in face_edge_list:
            if active[face[0]] and active[face[1]] and active[face[2]] and active[face[3]]:
                v += 1
        return v

    def try_add_edge(active, face_counts, e):
        for f_idx in edge_to_faces[e]:
            if face_counts[f_idx] == 3:
                return False, face_counts
        new_counts = face_counts.copy()
        for f_idx in edge_to_faces[e]:
            new_counts[f_idx] += 1
        return True, new_counts

    def recompute_face_counts(active):
        fc = np.full(num_faces, 4, dtype=np.int32)
        for e in range(num_edges):
            if active[e] == 0:
                for f_idx in edge_to_faces[e]:
                    fc[f_idx] -= 1
        return fc

    # 7. SA refine with kicks + local-search swaps
    def sa_refine(active, face_counts, rng, time_limit):
        best_active = active.copy()
        best_violations = count_violations(active)
        best_size = int(np.sum(active))
        current_violations = best_violations

        removed_edges = np.where(active == 0)[0]
        T = 1.0
        T_min = 0.005
        cooling = 0.9998

        while time.time() < time_limit:
            if len(removed_edges) == 0:
                break

            # Choose move type: add-removed (60%), swap (40%)
            move_type = rng.random()
            if move_type < 0.6 or len(removed_edges) == 0:
                # Try to add a removed edge
                e = removed_edges[rng.integers(0, len(removed_edges))]
                success, new_counts = try_add_edge(active, face_counts, e)
                if success:
                    active[e] = 1
                    face_counts = new_counts
                    removed_edges = np.where(active == 0)[0]
                    current_violations = count_violations(active)
                    current_size = int(np.sum(active))
                    if current_violations < best_violations or (
                        current_violations == best_violations and current_size > best_size):
                        best_violations = current_violations
                        best_size = current_size
                        best_active = active.copy()
                else:
                    # Kick: remove 1-2 edges from a blocking face, then add e
                    problem_faces = [f for f in edge_to_faces[e] if face_counts[f] == 3]
                    if problem_faces:
                        f_idx = problem_faces[rng.integers(0, len(problem_faces))]
                        face = face_edge_list[f_idx]
                        other_edges = [x for x in face if x != e]
                        num_remove = rng.integers(1, 3)
                        edges_to_remove = list(rng.choice(other_edges, size=min(num_remove, len(other_edges)), replace=False))
                        old_active = active.copy()
                        old_fc = face_counts.copy()
                        old_removed = removed_edges.copy()
                        for re in edges_to_remove:
                            active[re] = 0
                            for ff in edge_to_faces[re]:
                                face_counts[ff] -= 1
                        success2, new_counts2 = try_add_edge(active, face_counts, e)
                        if success2:
                            active[e] = 1
                            face_counts = new_counts2
                            removed_edges = np.where(active == 0)[0]
                            current_violations = count_violations(active)
                            current_size = int(np.sum(active))
                            if current_violations < best_violations or (
                                current_violations == best_violations and current_size > best_size):
                                best_violations = current_violations
                                best_size = current_size
                                best_active = active.copy()
                        else:
                            active = old_active
                            face_counts = old_fc
                            removed_edges = old_removed
            else:
                # Swap: remove an active edge, add a removed edge (net neutral)
                if len(removed_edges) == 0:
                    continue
                e_add = removed_edges[rng.integers(0, len(removed_edges))]
                # Find active edges that block e_add
                blocking = [f for f in edge_to_faces[e_add] if face_counts[f] == 3]
                if not blocking:
                    # e_add is addable directly (shouldn't happen in this branch, but handle)
                    active[e_add] = 1
                    for f in edge_to_faces[e_add]:
                        face_counts[f] += 1
                    removed_edges = np.where(active == 0)[0]
                    continue
                f_idx = blocking[rng.integers(0, len(blocking))]
                face = face_edge_list[f_idx]
                candidates_remove = [x for x in face if x != e_add and active[x] == 1]
                if not candidates_remove:
                    continue
                e_rem = candidates_remove[rng.integers(0, len(candidates_remove))]
                # Execute swap
                old_active = active.copy()
                old_fc = face_counts.copy()
                old_removed = removed_edges.copy()
                # remove e_rem
                active[e_rem] = 0
                for ff in edge_to_faces[e_rem]:
                    face_counts[ff] -= 1
                # add e_add
                can_add = True
                for f in edge_to_faces[e_add]:
                    if face_counts[f] == 3:
                        can_add = False
                        break
                if can_add:
                    active[e_add] = 1
                    for f in edge_to_faces[e_add]:
                        face_counts[f] += 1
                    removed_edges = np.where(active == 0)[0]
                    current_violations = count_violations(active)
                    current_size = int(np.sum(active))
                    if current_violations < best_violations or (
                        current_violations == best_violations and current_size > best_size):
                        best_violations = current_violations
                        best_size = current_size
                        best_active = active.copy()
                else:
                    active = old_active
                    face_counts = old_fc
                    removed_edges = old_removed

            T *= cooling
            if T < T_min:
                break

        return best_active, best_violations, best_size

    # 8. Final polish: repeatedly try to add any removable edge (greedy add-back)
    def polish(active, face_counts):
        changed = True
        while changed:
            changed = False
            for e in range(num_edges):
                if active[e] == 0:
                    success, new_fc = try_add_edge(active, face_counts, e)
                    if success:
                        active[e] = 1
                        face_counts = new_fc
                        changed = True
        return active, face_counts

    # 9. Main search
    rng = np.random.default_rng(42)

    # Build a set of diverse initializations
    inits = []
    # Spectral init (primary)
    a, fc = spectral_init(rng, deadline - 5)
    inits.append((a, fc))
    # Greedy init
    a, fc = greedy_hit(rng)
    inits.append((a, fc))
    # A couple of random-perturbed spectral inits for diversity
    for k in range(2):
        a, fc = spectral_init(rng, deadline - 5)
        # Perturb: randomly toggle a few edges
        for _ in range(rng.integers(8, 20)):
            e = rng.integers(0, num_edges)
            a[e] ^= 1
        fc = recompute_face_counts(a)
        inits.append((a, fc))

    best_active = None
    best_size = -1
    best_violations = 10**9

    for (init_active, init_fc) in inits:
        if time.time() > deadline - 8:
            break
        # Ensure init is a valid starting point (repair violations via greedy removal)
        a = init_active.copy()
        fc = init_fc.copy()
        # Repair: remove edges from all-4 faces
        while True:
            active_faces = np.where(fc == 4)[0]
            if len(active_faces) == 0:
                break
            f_idx = active_faces[0]
            face = face_edge_list[f_idx]
            # remove a random edge from the face
            e = face[rng.integers(0, 4)]
            a[e] = 0
            for ff in edge_to_faces[e]:
                fc[ff] -= 1
        # Run SA
        sa_limit = deadline - 6
        res_active, res_viol, res_size = sa_refine(a, fc, rng, sa_limit)
        if res_size > best_size or (res_size == best_size and res_viol < best_violations):
            best_size = res_size
            best_violations = res_viol
            best_active = res_active

    if best_active is None:
        best_active, _ = greedy_hit(rng)
        best_size = int(np.sum(best_active))
        best_violations = count_violations(best_active)

    # Final polish pass
    if time.time() < deadline - 2:
        fc = recompute_face_counts(best_active)
        best_active, fc = polish(best_active, fc)
        best_size = int(np.sum(best_active))
        best_violations = count_violations(best_active)

    # Convert to list of edges
    result = []
    for i in range(num_edges):
        if best_active[i] == 1:
            result.append(list(idx_to_edge[i]))

    return result
