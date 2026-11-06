# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin before 150s kill

    # Triangular lattice: point = a + b*omega, omega = exp(i*pi/3)
    # Squared distance between (a1,b1) and (a2,b2):
    #   da^2 + da*db + db^2
    # Unit-distance (edge) offsets (dist_sq == 1):
    EDGES = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da * da + da * db + db * db

    def count_edges(pts):
        e = 0
        n = len(pts)
        for i in range(n):
            pi = pts[i]
            for j in range(i + 1, n):
                if dist_sq(pi, pts[j]) == 1:
                    e += 1
        return e

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

    def build_deterministic_seed():
        """
        Deterministic greedy construction.
        Start with (0,0).
        Repeatedly add the valid candidate point that maximizes:
        1. Number of edges to existing points.
        2. Minimizes distance to centroid (for compactness).
        """
        pts = [(0, 0)]
        
        # Precompute centroid sum for tie-breaking
        sum_a = 0
        sum_b = 0
        
        for _ in range(31):
            best_cand = None
            best_edges = -1
            best_dist_centroid = float('inf')
            
            # Determine search range
            min_a = min(p[0] for p in pts)
            max_a = max(p[0] for p in pts)
            min_b = min(p[1] for p in pts)
            max_b = max(p[1] for p in pts)
            
            # Expand range slightly to allow filling gaps
            search_range = 3
            
            for a in range(min_a - search_range, max_a + search_range + 1):
                for b in range(min_b - search_range, max_b + search_range + 1):
                    if abs(a) > 64 or abs(b) > 64:
                        continue
                    cand = (a, b)
                    if cand in pts:
                        continue
                    
                    # Check validity (distance >= 1)
                    valid = True
                    edges = 0
                    for p in pts:
                        d = dist_sq(cand, p)
                        if d < 1:
                            valid = False
                            break
                        if d == 1:
                            edges += 1
                    
                    if not valid:
                        continue
                    
                    # Calculate distance to centroid
                    # Centroid is (sum_a/len, sum_b/len)
                    # Minimize (a - sum_a/len)^2 + ... roughly equivalent to minimizing a^2+a*b+b^2 if centered, 
                    # but let's use actual centroid distance squared for accuracy.
                    # Actually, since we want compactness, distance to current centroid is good.
                    da_c = a - sum_a / len(pts)
                    db_c = b - sum_b / len(pts)
                    # In triangular coords, distance squared is da^2 + da*db + db^2
                    dist_c = da_c * da_c + da_c * db_c + db_c * db_c
                    
                    if edges > best_edges or (edges == best_edges and dist_c < best_dist_centroid):
                        best_edges = edges
                        best_dist_centroid = dist_c
                        best_cand = cand
            
            if best_cand is None:
                # Fallback: expand search or break
                # If we can't find a neighbor, we might need to jump, but greedy usually finds one.
                # Let's try a wider range if we are stuck
                found = False
                for a in range(min_a - 10, max_a + 11):
                    if found: break
                    for b in range(min_b - 10, max_b + 11):
                        if abs(a) > 64 or abs(b) > 64: continue
                        cand = (a, b)
                        if cand in pts: continue
                        valid = True
                        edges = 0
                        for p in pts:
                            d = dist_sq(cand, p)
                            if d < 1:
                                valid = False
                                break
                            if d == 1:
                                edges += 1
                        if valid:
                            da_c = a - sum_a / len(pts)
                            db_c = b - sum_b / len(pts)
                            dist_c = da_c * da_c + da_c * db_c + db_c * db_c
                            if edges > best_edges or (edges == best_edges and dist_c < best_dist_centroid):
                                best_edges = edges
                                best_dist_centroid = dist_c
                                best_cand = cand
                if best_cand is None:
                    break
                found = True

            pts.append(best_cand)
            sum_a += best_cand[0]
            sum_b += best_cand[1]
            
        return pts[:32]

    def local_search_fast(pts, max_time):
        pts = list(pts)
        best_e = count_edges(pts)
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(32):
                if time.time() > max_time:
                    break
                old = pts[i]
                best_delta = 0
                best_cand = None
                
                # Try all 6 neighbors and distance 2
                candidates = []
                for da, db in EDGES:
                    c = (old[0] + da, old[1] + db)
                    if abs(c[0]) <= 64 and abs(c[1]) <= 64:
                        candidates.append(c)
                for da in range(-2, 3):
                    for db in range(-2, 3):
                        if da == 0 and db == 0: continue
                        c = (old[0] + da, old[1] + db)
                        if abs(c[0]) <= 64 and abs(c[1]) <= 64:
                            candidates.append(c)
                
                for cand in candidates:
                    if cand in pts:
                        continue
                    valid = True
                    new_edges = 0
                    for j in range(32):
                        if j == i: continue
                        d = dist_sq(cand, pts[j])
                        if d < 1:
                            valid = False
                            break
                        if d == 1:
                            new_edges += 1
                    if not valid:
                        continue
                    
                    # Calculate old edges for point i
                    old_edges = 0
                    for j in range(32):
                        if j == i: continue
                        if dist_sq(old, pts[j]) == 1:
                            old_edges += 1
                    
                    delta = new_edges - old_edges
                    if delta > best_delta:
                        best_delta = delta
                        best_cand = cand
                
                if best_cand and best_delta > 0:
                    pts[i] = best_cand
                    best_e += best_delta
                    improved = True
        
        return pts, best_e

    # Main
    seed = build_deterministic_seed()
    
    if not is_valid(seed):
        # Fallback to a simple valid configuration if greedy failed
        # This should not happen, but just in case
        seed = [(i, j) for i in range(8) for j in range(4)]
        # Check validity of fallback, might need adjustment
        if not is_valid(seed):
             # Force valid: spread out
             seed = [(i, 0) for i in range(32)]
             if not is_valid(seed):
                 return [[a, b] for a in range(16) for b in range(2)]

    best_pts, best_e = local_search_fast(seed, deadline - 10.0)
    
    if best_e >= 76:
        return [[a, b] for a, b in best_pts]
    
    # If not optimal, try a few random perturbations + local search
    best_pts, best_e = best_pts, best_e
    for _ in range(5):
        if time.time() > deadline - 5.0:
            break
        # Perturb
        perturbed = list(best_pts)
        for _ in range(5):
            idx = random.randrange(32)
            da = random.randint(-3, 3)
            db = random.randint(-3, 3)
            new_p = (perturbed[idx][0] + da, perturbed[idx][1] + db)
            if abs(new_p[0]) <= 64 and abs(new_p[1]) <= 64 and new_p not in perturbed:
                valid = True
                for j in range(32):
                    if j == idx: continue
                    if dist_sq(new_p, perturbed[j]) < 1:
                        valid = False
                        break
                if valid:
                    perturbed[idx] = new_p
        
        p_pts, p_e = local_search_fast(perturbed, deadline - 5.0)
        if p_e > best_e:
            best_pts, best_e = p_pts, p_e
            if best_e >= 76:
                break

    if not is_valid(best_pts):
        return [[a, b] for a in range(16) for b in range(2)]

    return [[a, b] for a, b in best_pts]
# EVOLVE-BLOCK-END
