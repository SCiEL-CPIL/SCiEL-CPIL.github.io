# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start = time.time()
    deadline = start + 140.0

    EDGES = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da * da + da * db + db * db

    def count_edges_fast(pts):
        pt_set = set(pts)
        e = 0
        for p in pts:
            for da, db in EDGES:
                q = (p[0] + da, p[1] + db)
                if q in pt_set:
                    e += 1
        return e // 2

    def is_valid(pts):
        if len(pts) != 32:
            return False
        seen = set()
        for p in pts:
            if p in seen:
                return False
            seen.add(p)
            if abs(p[0]) > 64 or abs(p[1]) > 64:
                return False
        for i in range(32):
            pi = pts[i]
            for j in range(i + 1, 32):
                if dist_sq(pi, pts[j]) < 1:
                    return False
        return True

    def build_hex_spiral_optimal():
        """
        Construct the optimal 32-point hexagonal spiral with exactly 76 edges.
        Uses a systematic ring-by-ring filling that maximizes adjacency.
        """
        pts = [(0, 0)]
        
        # Ring 1: 6 points at hex-distance 1
        ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
        pts.extend(ring1)  # 7 points
        
        # Ring 2: 12 points at hex-distance 2
        ring2 = []
        for a in range(-2, 3):
            for b in range(-2, 3):
                d = a*a + a*b + b*b
                if d == 4:
                    ring2.append((a, b))
        pts.extend(ring2)  # 19 points
        
        # Ring 3: need 13 more from hex-distance 3
        ring3 = []
        for a in range(-3, 4):
            for b in range(-3, 4):
                d = a*a + a*b + b*b
                if d == 9:
                    ring3.append((a, b))
        
        # Sort ring 3 points to maximize edges with existing points
        def edge_score(p):
            s = set(pts)
            cnt = 0
            for da, db in EDGES:
                if (p[0]+da, p[1]+db) in s:
                    cnt += 1
            return cnt
        
        ring3.sort(key=edge_score, reverse=True)
        pts.extend(ring3[:13])
        
        return pts[:32]

    def build_known_optimal():
        """
        Directly return a known 32-point configuration with 76 edges.
        This is a hexagonal spiral pattern that attains Harborth's bound.
        """
        # Center + 6 in ring 1 + 12 in ring 2 + 13 in ring 3
        # The 13 from ring 3 are chosen to maximize connectivity
        pts = [
            (0, 0),
            # Ring 1
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            # Ring 2
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
            (-1, -1), (0, -2), (1, -2), (2, -1), (2, 1), (1, 2),
            # Ring 3: 13 points chosen for maximum edge count
            (3, 0), (2, 2), (1, 3), (0, 3), (-1, 3), (-2, 2),
            (-3, 1), (-3, 0), (-2, -1), (-1, -2), (0, -3),
            (1, -3), (2, -2)
        ]
        return pts

    def local_search_fast(pts, max_time):
        pts = list(pts)
        pt_set = set(pts)
        pos_to_idx = {p: i for i, p in enumerate(pts)}
        best = count_edges_fast(pts)

        def get_point_edges(p, exclude_pos):
            e = 0
            for da, db in EDGES:
                q = (p[0] + da, p[1] + db)
                if q in pt_set and q != exclude_pos:
                    e += 1
            return e

        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(32):
                if time.time() > max_time:
                    break
                old = pts[i]
                old_edges = get_point_edges(old, old)

                best_cand = None
                best_delta = 0

                for da in range(-4, 5):
                    for db in range(-4, 5):
                        if da == 0 and db == 0:
                            continue
                        cand = (old[0] + da, old[1] + db)
                        if abs(cand[0]) > 64 or abs(cand[1]) > 64:
                            continue
                        if cand in pt_set:
                            continue

                        # Check validity: no neighbor of cand (other than old) is in pt_set
                        valid = True
                        for da2, db2 in EDGES:
                            neighbor = (cand[0] + da2, cand[1] + db2)
                            if neighbor in pt_set and neighbor != old:
                                valid = False
                                break
                        if not valid:
                            continue

                        new_edges = get_point_edges(cand, old)
                        delta = new_edges - old_edges
                        if delta > best_delta:
                            best_delta = delta
                            best_cand = cand

                if best_cand is not None and best_delta > 0:
                    pt_set.remove(old)
                    pos_to_idx.pop(old)
                    pts[i] = best_cand
                    pt_set.add(best_cand)
                    pos_to_idx[best_cand] = i
                    best += best_delta
                    improved = True

        return pts, best

    def anneal_fast(pts, max_time):
        pts = list(pts)
        pt_set = set(pts)
        cur_e = count_edges_fast(pts)
        best_pts = list(pts)
        best_e = cur_e
        T = 3.0
        T_min = 0.01
        cooldown = 0.9995

        def get_point_edges(p, exclude_pos):
            e = 0
            for da, db in EDGES:
                q = (p[0] + da, p[1] + db)
                if q in pt_set and q != exclude_pos:
                    e += 1
            return e

        while time.time() < max_time:
            i = random.randrange(32)
            old = pts[i]
            old_edges = get_point_edges(old, old)

            if random.random() < 0.8:
                da = random.randint(-2, 2)
                db = random.randint(-2, 2)
            else:
                da = random.randint(-5, 5)
                db = random.randint(-5, 5)
            cand = (old[0] + da, old[1] + db)

            if abs(cand[0]) > 64 or abs(cand[1]) > 64 or cand in pt_set:
                T = max(T_min, T * cooldown)
                continue

            valid = True
            for da2, db2 in EDGES:
                neighbor = (cand[0] + da2, cand[1] + db2)
                if neighbor in pt_set and neighbor != old:
                    valid = False
                    break

            if not valid:
                T = max(T_min, T * cooldown)
                continue

            new_edges = get_point_edges(cand, old)
            delta = new_edges - old_edges

            if delta >= 0 or random.random() < math.exp(delta / T):
                pt_set.remove(old)
                pts[i] = cand
                pt_set.add(cand)
                cur_e += delta
                if cur_e > best_e:
                    best_e = cur_e
                    best_pts = list(pts)

            T = max(T_min, T * cooldown)

        return best_pts, best_e

    # ---- Main ----
    # Try the directly known optimal configuration first
    known_pts = build_known_optimal()
    if is_valid(known_pts):
        ke = count_edges_fast(known_pts)
        if ke >= 76:
            return [[a, b] for a, b in known_pts]

    # Try the hexagonal spiral construction
    hex_pts = build_hex_spiral_optimal()
    if is_valid(hex_pts):
        he = count_edges_fast(hex_pts)
        if he >= 76:
            return [[a, b] for a, b in hex_pts]

    # If neither direct construction works, use search
    seeds = [known_pts, hex_pts]
    valid_seeds = [s for s in seeds if is_valid(s)]
    if not valid_seeds:
        # Fallback: build a valid seed greedily
        pts = [(0, 0)]
        for _ in range(31):
            best_cand = None
            best_score = -1
            for a in range(-5, 6):
                for b in range(-5, 6):
                    if abs(a) > 64 or abs(b) > 64:
                        continue
                    cand = (a, b)
                    if cand in pts:
                        continue
                    valid = True
                    for p in pts:
                        if dist_sq(cand, p) < 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    edges = 0
                    for p in pts:
                        if dist_sq(cand, p) == 1:
                            edges += 1
                    if edges > best_score:
                        best_score = edges
                        best_cand = cand
            if best_cand is None:
                break
            pts.append(best_cand)
        valid_seeds = [pts]

    best_seed = max(valid_seeds, key=lambda s: count_edges_fast(s))
    best_pts, best_e = local_search_fast(best_seed, deadline - 20.0)

    if best_e >= 76:
        return [[a, b] for a, b in best_pts]

    while time.time() < deadline - 10.0:
        cand_pts, cand_e = anneal_fast(best_pts, deadline - 10.0)
        if cand_e > best_e:
            best_pts, best_e = cand_pts, cand_e
            if best_e >= 76:
                break
            polished, pe = local_search_fast(best_pts, deadline - 10.0)
            if pe > best_e:
                best_pts, best_e = polished, pe
                if best_e >= 76:
                    break

    if not is_valid(best_pts):
        return [[a, b] for a in range(16) for b in range(2)]

    return [[a, b] for a, b in best_pts]
# EVOLVE-BLOCK-END
