import time
import random
from collections import defaultdict

def construct():
    """
    C4-free edge set maximizer for Q7.
    Combines a strong deterministic structured seed (bipartite even-parity pattern)
    with local-search repair and addition, plus multiple randomized restarts.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    d = 7
    n_vertices = 1 << d  # 128
    n_edges_total = d * (1 << (d - 1))  # 448

    # Precompute all edges
    edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))

    n_edges = len(edges)
    edge_to_idx = {e: i for i, e in enumerate(edges)}

    # Build face list: each face is a tuple of 4 edge indices
    faces = []
    for base in range(n_vertices):
        for d1 in range(d):
            if (base >> d1) & 1:
                continue
            for d2 in range(d1 + 1, d):
                if (base >> d2) & 1:
                    continue
                e1 = (base, base ^ (1 << d1))
                e2 = (base, base ^ (1 << d2))
                e3 = (base ^ (1 << d1), base ^ (1 << d1) ^ (1 << d2))
                e4 = (base ^ (1 << d2), base ^ (1 << d1) ^ (1 << d2))
                e1 = (min(e1), max(e1))
                e2 = (min(e2), max(e2))
                e3 = (min(e3), max(e3))
                e4 = (min(e4), max(e4))
                face_edges = (edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4])
                faces.append(face_edges)

    n_faces = len(faces)

    # For each edge, list of faces it belongs to
    edge_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        for ei in face:
            edge_faces[ei].append(fi)

    def time_left():
        return deadline - time.time()

    def is_valid(sol):
        for face in faces:
            if all(sol[e] for e in face):
                return False
        return True

    def count_edges(sol):
        return sum(1 for e in range(n_edges) if sol[e])

    def compute_face_counts(sol):
        fc = [0] * n_faces
        for ei in range(n_edges):
            if sol[ei]:
                for fi in edge_faces[ei]:
                    fc[fi] += 1
        return fc

    def structured_seed():
        """
        Deterministic C4-free seed: include an edge (u,v) if the parity of
        the differing bit index is even. This creates a bipartite-like structure
        that avoids C4s.
        """
        sol = [False] * n_edges
        for ei, (u, v) in enumerate(edges):
            # Find which bit differs
            diff = u ^ v
            bit = diff.bit_length() - 1
            if bit % 2 == 0:  # even-indexed bit
                sol[ei] = True
        return sol

    def greedy_initial(seed=None):
        """Randomized greedy: add edges in random order if they don't complete a face."""
        rng = random.Random(seed)
        sol = [False] * n_edges
        fc = [0] * n_faces
        order = list(range(n_edges))
        rng.shuffle(order)
        for ei in order:
            can_add = True
            for fi in edge_faces[ei]:
                if fc[fi] == 3:
                    can_add = False
                    break
            if can_add:
                sol[ei] = True
                for fi in edge_faces[ei]:
                    fc[fi] += 1
        return sol

    def perturbed_seed(sol, rng, frac=0.1):
        """Randomly flip some edges to escape local optima."""
        sol = list(sol)
        n_flip = int(n_edges * frac)
        for _ in range(n_flip):
            ei = rng.randint(0, n_edges - 1)
            sol[ei] = not sol[ei]
        return sol

    def local_search(sol, seed=None):
        """Iteratively repair bad faces and add safe edges until no improvement."""
        rng = random.Random(seed)
        sol = list(sol)
        fc = compute_face_counts(sol)

        # Find bad faces (all 4 edges present)
        bad_faces = set(fi for fi in range(n_faces) if fc[fi] == 4)

        improved = True
        while improved and time_left() > 5:
            improved = False

            if bad_faces:
                # Remove edge from bad face that fixes the most other bad faces
                bad_list = list(bad_faces)
                rng.shuffle(bad_list)
                best_removal = None
                best_score = -1
                for fi in bad_list:
                    for ei in faces[fi]:
                        if sol[ei]:
                            cnt = sum(1 for f2 in edge_faces[ei] if f2 in bad_faces)
                            if cnt > best_score:
                                best_score = cnt
                                best_removal = ei
                if best_removal is not None:
                    ei = best_removal
                    sol[ei] = False
                    for fi2 in edge_faces[ei]:
                        fc[fi2] -= 1
                        if fc[fi2] == 3 and fi2 in bad_faces:
                            bad_faces.discard(fi2)
                    improved = True
            else:
                # No bad faces: try to add edges that don't complete any face
                addable = []
                for ei in range(n_edges):
                    if not sol[ei]:
                        ok = True
                        for fi in edge_faces[ei]:
                            if fc[fi] >= 3:
                                ok = False
                                break
                        if ok:
                            addable.append(ei)
                if addable:
                    rng.shuffle(addable)
                    for ei in addable:
                        if time_left() < 5:
                            break
                        ok = True
                        for fi in edge_faces[ei]:
                            if fc[fi] >= 3:
                                ok = False
                                break
                        if ok:
                            sol[ei] = True
                            for fi in edge_faces[ei]:
                                fc[fi] += 1
                            improved = True
                else:
                    break
        return sol

    best_count = 0
    best_solution = None

    # Attempt 0: structured seed
    if time_left() > 15:
        sol = structured_seed()
        sol = local_search(sol, seed=12345)
        if is_valid(sol):
            cnt = count_edges(sol)
            if cnt > best_count:
                best_count = cnt
                best_solution = list(sol)

    # Multiple randomized restarts
    attempts = 0
    while time_left() > 10:
        attempts += 1
        seed = attempts * 7919 + 1

        # Alternate between greedy and perturbed structured seed
        if attempts % 2 == 1:
            sol = greedy_initial(seed)
        else:
            base = structured_seed()
            sol = perturbed_seed(base, random.Random(seed), frac=0.05 + 0.05 * (attempts % 4))

        sol = local_search(sol, seed)

        if is_valid(sol):
            cnt = count_edges(sol)
            if cnt > best_count:
                best_count = cnt
                best_solution = list(sol)

    # Fallback: simple greedy if nothing found
    if best_solution is None or best_count < 200:
        chosen = set()
        out = []
        for x in range(n_vertices):
            for i in range(d):
                y = x ^ (1 << i)
                if x < y:
                    ok = True
                    for j in range(d):
                        if j != i:
                            c = x ^ (1 << j)
                            e = y ^ (1 << j)
                            if all(p in chosen for p in ((min(x, c), max(x, c)), (min(y, e), max(y, e)), (min(c, e), max(c, e)))):
                                ok = False
                                break
                    if ok:
                        chosen.add((x, y))
                        out.append([x, y])
        return out

    result = []
    for ei in range(n_edges):
        if best_solution[ei]:
            result.append(list(edges[ei]))
    return result
