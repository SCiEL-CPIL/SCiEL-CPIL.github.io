import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 140.0
    
    D = 1
    max_coord = 100
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            candidates.append((x, y))
            
    n_cand = len(candidates)
    cand_to_idx = {p: i for i, p in enumerate(candidates)}
    
    dist2 = [[0]*n_cand for _ in range(n_cand)]
    is_sq = [[False]*n_cand for _ in range(n_cand)]
    
    for i in range(n_cand):
        xi, yi = candidates[i]
        for j in range(i+1, n_cand):
            xj, yj = candidates[j]
            dx = xi - xj
            dy = yi - yj
            d2 = dx*dx + dy*dy
            dist2[i][j] = d2
            dist2[j][i] = d2
            r = int(d2**0.5)
            if r*r == d2:
                is_sq[i][j] = True
                is_sq[j][i] = True

    def check_collinear_idx(i, j, k):
        x1, y1 = candidates[i]
        x2, y2 = candidates[j]
        x3, y3 = candidates[k]
        return (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1) != 0

    def check_concyclic_idx(i, j, k, l):
        x1, y1 = candidates[i]
        x2, y2 = candidates[j]
        x3, y3 = candidates[k]
        x4, y4 = candidates[l]
        a1 = x1*x1 + y1*y1
        a2 = x2*x2 + y2*y2
        a3 = x3*x3 + y3*y3
        a4 = x4*x4 + y4*y4
        
        d1 = a1 - a4
        d2 = a2 - a4
        d3 = a3 - a4
        dx1 = x1 - x4
        dx2 = x2 - x4
        dx3 = x3 - x4
        dy1 = y1 - y4
        dy2 = y2 - y4
        dy3 = y3 - y4
        
        det = d1*(dx2*dy3 - dy2*dx3) - dx1*(d2*dy3 - dy2*d3) + dy1*(d2*dx3 - dx2*d3)
        return det != 0

    def check_validity(indices):
        n = len(indices)
        if len(set(indices)) != n:
            return False
        for i in range(n):
            for j in range(i+1, n):
                if not is_sq[indices[i]][indices[j]]:
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if not check_collinear_idx(indices[i], indices[j], indices[k]):
                        return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if not check_concyclic_idx(indices[i], indices[j], indices[k], indices[l]):
                            return False
        return True

    def try_build(target_size, start_set=None):
        current = start_set.copy() if start_set else []
        # If start_set is not valid, return None
        if start_set and not check_validity(current):
            return None
            
        while len(current) < target_size and time.time() < deadline:
            added = False
            # Try all candidates
            for idx in range(n_cand):
                if idx in current:
                    continue
                new_set = current + [idx]
                # Optimized check: only check constraints involving the new point
                # 1. Distances to all existing
                ok = True
                for existing in current:
                    if not is_sq[idx][existing]:
                        ok = False
                        break
                if not ok:
                    continue
                # 2. Collinearity with pairs of existing
                if ok:
                    for i in range(len(current)):
                        for j in range(i+1, len(current)):
                            if not check_collinear_idx(idx, current[i], current[j]):
                                ok = False
                                break
                        if not ok:
                            break
                if not ok:
                    continue
                # 3. Concyclicity with triples of existing
                if ok:
                    for i in range(len(current)):
                        for j in range(i+1, len(current)):
                            for k in range(j+1, len(current)):
                                if not check_concyclic_idx(idx, current[i], current[j], current[k]):
                                    ok = False
                                    break
                                if not ok:
                                    break
                                if not ok:
                                    break
                        if not ok:
                            break
                        if not ok:
                            break
                if ok:
                    current = new_set
                    added = True
                    break
            if not added:
                return current # Stuck, return what we have
        
        return current

    best_set = []
    
    # Try to extend seed
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    seed_indices = []
    for p in seed_points:
        if p in cand_to_idx:
            seed_indices.append(cand_to_idx[p])
            
    if check_validity(seed_indices):
        ext_set = try_build(12, seed_indices)
        if len(ext_set) > len(best_set):
            best_set = ext_set

    # If we didn't get more than 5, try random restarts to find a set of 6, 7, etc.
    # We want to find ANY set of size > 5.
    if len(best_set) <= 5:
        for _ in range(50): # 50 restarts
            if time.time() > deadline:
                break
            # Pick 3 random points
            # Ensure they are not collinear and have rational distances
            # This is hard to pick randomly.
            # Instead, pick a random point, then find neighbors with rational distance.
            # Graph approach: Nodes are points. Edges if rational distance.
            # We want a clique of size N with no 3 collinear and no 4 concyclic.
            
            # Let's just try to build from a random starting triple.
            # Find a valid triple first.
            valid_triple = None
            for _ in range(1000):
                i, j, k = random.sample(range(n_cand), 3)
                if is_sq[i][j] and is_sq[j][k] and is_sq[i][k]:
                    if check_collinear_idx(i, j, k):
                        valid_triple = [i, j, k]
                        break
            if valid_triple:
                built = try_build(12, valid_triple)
                if len(built) > len(best_set):
                    best_set = built
                    if len(best_set) >= 8: # Good enough?
                        break

    # Convert best_set to output format
    points_out = []
    for idx in best_set:
        x, y = candidates[idx]
        points_out.append([[[x, 1], [y, 1]]])
        
    return {'D': D, 'points': points_out}
