import time

def construct():
    # Time budget: 150s wall-clock, keep safety margin
    t0 = time.time()
    deadline = 140.0

    # Neighbors in triangular lattice (axial coordinates)
    neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

    def count_edges(pts):
        """Count the number of pairs at distance exactly 1."""
        pset = set(pts)
        edges = 0
        for a, b in pts:
            for da, db in neighbors:
                if (a + da, b + db) in pset:
                    edges += 1
        return edges // 2

    def hex_dist(a, b):
        return (abs(a) + abs(b) + abs(a + b)) // 2

    # Candidate pool: all points within hex_dist <= 8
    all_candidates = []
    for a in range(-8, 9):
        for b in range(-8, 9):
            if hex_dist(a, b) <= 8:
                all_candidates.append((a, b))

    # Precompute neighbors for each candidate
    candidate_neighbors = {}
    cand_set = set(all_candidates)
    for pt in all_candidates:
        a, b = pt
        nbs = []
        for da, db in neighbors:
            nb = (a + da, b + db)
            if nb in cand_set:
                nbs.append(nb)
        candidate_neighbors[pt] = nbs

    def edge_delta_remove_add(cur_set, remove_pt, add_pt):
        """Calculate change in edge count when removing remove_pt and adding add_pt."""
        lost = 0
        for nb in candidate_neighbors[remove_pt]:
            if nb in cur_set and nb != add_pt:
                lost += 1

        gained = 0
        for nb in candidate_neighbors[add_pt]:
            if nb in cur_set and nb != remove_pt:
                gained += 1

        return gained - lost

    # Greedy spiral construction: add points that maximize edges to existing set
    current = [(0, 0)]
    current_set = {(0, 0)}

    for _ in range(31):
        best_pt = None
        best_score = -1
        best_dist = 999
        for pt in all_candidates:
            if pt in current_set:
                continue
            score = 0
            for nb in candidate_neighbors[pt]:
                if nb in current_set:
                    score += 1
            d = hex_dist(pt[0], pt[1])
            if score > best_score or (score == best_score and d < best_dist):
                best_score = score
                best_dist = d
                best_pt = pt
        if best_pt:
            current.append(best_pt)
            current_set.add(best_pt)

    best_points = list(current)
    best_edges = count_edges(best_points)

    if best_edges >= 76:
        return [list(p) for p in best_points]

    # Local Search: Best-Improvement single-swap
    def local_search(pts, max_iters=10000):
        cur_pts = list(pts)
        cur_set = set(pts)
        cur_edges = count_edges(cur_pts)
        pool = [p for p in all_candidates if p not in cur_set]

        for _ in range(max_iters):
            if time.time() > deadline:
                break

            improved = False
            best_delta = 0
            best_move = None

            for i in range(32):
                old = cur_pts[i]
                old_neighbors = candidate_neighbors[old]

                for new in pool:
                    if new == old:
                        continue
                    delta = edge_delta_remove_add(cur_set - {old}, old, new)
                    if delta > best_delta:
                        best_delta = delta
                        best_move = (i, old, new)

            if best_move:
                i, old, new = best_move
                cur_pts[i] = new
                cur_set.discard(old)
                cur_set.add(new)
                cur_edges += best_delta

                # Update pool
                pool_idx = pool.index(old)
                pool[pool_idx] = new
                improved = True

            if not improved:
                break

        return cur_pts, cur_set, cur_edges

    # Try local search from greedy start
    res_pts, res_set, res_edges = local_search(best_points)
    if res_edges > best_edges:
        best_edges = res_edges
        best_points = res_pts

    if best_edges >= 76:
        return [list(p) for p in best_points]

    # ILS: Iterate Local Search with random perturbations
    import random
    rng = random.Random(42)

    for k in range(100):
        if time.time() > deadline:
            break
        if best_edges >= 76:
            break

        # Perturb best solution: randomly swap 2-3 points
        perturbed = list(best_points)
        perturbed_set = set(perturbed)
        pool_for_perturb = [p for p in all_candidates if p not in perturbed_set]

        if len(pool_for_perturb) < 3:
            continue

        num_swaps = rng.randint(2, 3)
        for _ in range(num_swaps):
            i1 = rng.randint(0, 31)
            i2 = rng.randint(0, 31)
            if i1 == i2:
                continue

            j1 = rng.randint(0, len(pool_for_perturb) - 1)
            j2 = rng.randint(0, len(pool_for_perturb) - 1)
            if j1 == j2:
                continue

            n1 = pool_for_perturb[j1]
            n2 = pool_for_perturb[j2]

            old1 = perturbed[i1]
            old2 = perturbed[i2]

            perturbed[i1] = n1
            perturbed[i2] = n2

            perturbed_set.discard(old1)
            perturbed_set.discard(old2)
            perturbed_set.add(n1)
            perturbed_set.add(n2)

            # Update pool
            pool_for_perturb[j1] = old1
            pool_for_perturb[j2] = old2

        # Run local search from perturbed state
        res_pts, res_set, res_edges = local_search(perturbed)
        if res_edges > best_edges:
            best_edges = res_edges
            best_points = res_pts

    return [list(p) for p in best_points]
