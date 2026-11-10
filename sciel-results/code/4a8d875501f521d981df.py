import time
import random

def construct():
    t0 = time.time()
    deadline = 145.0

    # Neighbors in triangular lattice (axial coordinates)
    neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

    def count_edges(pts):
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

    # Greedy spiral construction
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
            for da, db in neighbors:
                if (pt[0] + da, pt[1] + db) in current_set:
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

    def local_search(pts, max_iters=2000):
        cur_pts = list(pts)
        cur_set = set(pts)
        cur_edges = count_edges(cur_pts)
        pool = [p for p in all_candidates if p not in cur_set]

        improved = True
        while improved and (time.time() - t0) < deadline:
            improved = False
            best_delta = 0
            best_move = None
            for i in range(32):
                old = cur_pts[i]
                for new in pool:
                    # Calculate edge delta for removing old and adding new
                    temp_set = cur_set - {old}
                    lost = 0
                    for da, db in neighbors:
                        if (old[0] + da, old[1] + db) in cur_set and (old[0] + da, old[1] + db) != old:
                            lost += 1
                    gained = 0
                    for da, db in neighbors:
                        nb = (new[0] + da, new[1] + db)
                        if nb in temp_set:
                            gained += 1
                    # Check if new creates edge with old (shouldn't since old is removed)
                    delta = gained - lost
                    if delta > best_delta:
                        best_delta = delta
                        best_move = (i, old, new)

            if best_move:
                i, old, new = best_move
                cur_pts[i] = new
                cur_set.discard(old)
                cur_set.add(new)
                cur_edges += best_delta
                pool.remove(old)
                pool.append(new)
                improved = True

        return cur_pts, cur_set, cur_edges

    # Run local search
    cur_pts, cur_set, cur_edges = local_search(best_points)
    if cur_edges > best_edges:
        best_edges = cur_edges
        best_points = cur_pts

    if best_edges >= 76:
        return [list(p) for p in best_points]

    # ILS: Iterated Local Search
    random.seed(42)
    for k in range(200):
        if (time.time() - t0) > deadline:
            break
        if best_edges >= 76:
            break

        # Perturb: swap 2 random points with 2 random pool points
        perturbed = list(best_points)
        perturbed_set = set(best_points)
        pool_for_perturb = [p for p in all_candidates if p not in perturbed_set]

        if len(pool_for_perturb) < 2:
            break

        i1 = random.randint(0, 31)
        i2 = random.randint(0, 31)
        if i1 == i2:
            continue

        j1 = random.randint(0, len(pool_for_perturb) - 1)
        j2 = random.randint(0, len(pool_for_perturb) - 1)
        if j1 == j2:
            continue

        n1 = pool_for_perturb[j1]
        n2 = pool_for_perturb[j2]

        perturbed[i1] = n1
        perturbed[i2] = n2

        # Run local search from perturbed state
        res_pts, res_set, res_edges = local_search(perturbed)

        if res_edges > best_edges:
            best_edges = res_edges
            best_points = res_pts

    return [list(p) for p in best_points]
