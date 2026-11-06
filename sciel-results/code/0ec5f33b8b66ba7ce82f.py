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

    # ---- Build a hexagonal/spiral seed ----
    def build_hex_seed():
        """Build a compact hexagonal cluster of 32 points."""
        pts = [(0, 0)]
        
        # Ring 1: 6 points
        ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
        pts.extend(ring1)
        
        # Ring 2: 12 points at distance 2 from origin
        ring2 = []
        for a in range(-2, 3):
            for b in range(-2, 3):
                if a == 0 and b == 0:
                    continue
                if dist_sq((a,b), (0,0)) == 4:  # distance 2
                    ring2.append((a,b))
        # Filter to only those that are valid (not too close to existing)
        valid_ring2 = []
        for p in ring2:
            valid = True
            for q in pts:
                if dist_sq(p, q) < 1:
                    valid = False
                    break
            if valid:
                valid_ring2.append(p)
        pts.extend(valid_ring2[:12])  # Take up to 12 from ring 2
        
        # Ring 3: need 32 - len(pts) more points
        needed = 32 - len(pts)
        ring3 = []
        for a in range(-3, 4):
            for b in range(-3, 4):
                if dist_sq((a,b), (0,0)) == 9:  # distance 3
                    ring3.append((a,b))
        valid_ring3 = []
        for p in ring3:
            valid = True
            for q in pts:
                if dist_sq(p, q) < 1:
                    valid = False
                    break
            if valid:
                valid_ring3.append(p)
        # Sort by distance to center for compactness
        valid_ring3.sort(key=lambda p: p[0]*p[0] + p[0]*p[1] + p[1]*p[1])
        pts.extend(valid_ring3[:needed])
        
        return pts[:32]

    def build_greedy_seed():
        pts = [(0, 0)]
        
        for _ in range(31):
            best_cand = None
            best_score = -1
            best_dist_center = 999
            
            min_a = min(p[0] for p in pts)
            max_a = max(p[0] for p in pts)
            min_b = min(p[1] for p in pts)
            max_b = max(p[1] for p in pts)
            
            for a in range(min_a - 2, max_a + 3):
                for b in range(min_b - 2, max_b + 3):
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
                    
                    d_center = a*a + a*b + b*b
                    
                    if edges > best_score or (edges == best_score and d_center < best_dist_center):
                        best_score = edges
                        best_dist_center = d_center
                        best_cand = cand
            
            if best_cand is None:
                for _ in range(100):
                    da = random.randint(-5, 5)
                    db = random.randint(-5, 5)
                    cand = (pts[-1][0] + da, pts[-1][1] + db)
                    if abs(cand[0]) <= 64 and abs(cand[1]) <= 64 and cand not in pts:
                        valid = True
                        for p in pts:
                            if dist_sq(cand, p) < 1:
                                valid = False
                                break
                        if valid:
                            best_cand = cand
                            break
            
            if best_cand:
                pts.append(best_cand)
            else:
                break
                
        return pts[:32]

    # ---- Fast local search with precomputed neighbor counts ----
    def fast_local_search(pts, max_time):
        pts = list(pts)
        best = count_edges(pts)
        improved = True
        
        while improved and time.time() < max_time:
            improved = False
            for i in range(32):
                if time.time() > max_time:
                    break
                old = pts[i]
                
                # Generate candidates: all lattice points within radius 3 that are valid
                candidates = []
                for da in range(-3, 4):
                    for db in range(-3, 4):
                        if da == 0 and db == 0:
                            continue
                        c = (old[0] + da, old[1] + db)
                        if abs(c[0]) <= 64 and abs(c[1]) <= 64:
                            candidates.append(c)
                
                best_delta = 0
                best_cand = None
                
                for cand in candidates:
                    if cand in pts:
                        continue
                    # Check validity
                    valid = True
                    for j in range(32):
                        if j == i:
                            continue
                        if dist_sq(cand, pts[j]) < 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Count edges for old position
                    old_e = 0
                    for j in range(32):
                        if j != i and dist_sq(old, pts[j]) == 1:
                            old_e += 1
                    
                    # Count edges for new position
                    new_e = 0
                    for j in range(32):
                        if j != i and dist_sq(cand, pts[j]) == 1:
                            new_e += 1
                    
                    delta = new_e - old_e
                    if delta > best_delta:
                        best_delta = delta
                        best_cand = cand
                
                if best_cand is not None and best_delta > 0:
                    pts[i] = best_cand
                    best += best_delta
                    improved = True
        
        return pts, best

    # ---- Simulated annealing with adaptive cooling ----
    def anneal(pts, max_time):
        pts = list(pts)
        cur_e = count_edges(pts)
        best_pts = list(pts)
        best_e = cur_e
        T = 1.0
        T_min = 1e-6
        cooldown = 0.9995
        no_improve = 0
        
        while time.time() < max_time:
            i = random.randrange(32)
            old = pts[i]
            
            # Pick a random candidate within radius 3
            found = False
            for _ in range(20):
                da = random.randint(-3, 3)
                db = random.randint(-3, 3)
                if da == 0 and db == 0:
                    continue
                cand = (old[0] + da, old[1] + db)
                if abs(cand[0]) > 64 or abs(cand[1]) > 64:
                    continue
                if cand in pts:
                    continue
                # Check validity
                valid = True
                for j in range(32):
                    if j == i:
                        continue
                    if dist_sq(cand, pts[j]) < 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                old_e = 0
                for j in range(32):
                    if j != i and dist_sq(old, pts[j]) == 1:
                        old_e += 1
                
                new_e = 0
                for j in range(32):
                    if j != i and dist_sq(cand, pts[j]) == 1:
                        new_e += 1
                
                delta = new_e - old_e
                
                # Accept if improvement or with Boltzmann probability
                if delta >= 0:
                    accept = True
                    no_improve = 0
                else:
                    accept = random.random() < math.exp(delta / max(T, 1e-10))
                    if not accept:
                        no_improve += 1
                
                if accept:
                    pts[i] = cand
                    cur_e += delta
                    if cur_e > best_e:
                        best_e = cur_e
                        best_pts = list(pts)
                        no_improve = 0
                    found = True
                    break
            
            if not found:
                no_improve += 1
            
            # Adaptive cooling
            T = max(T_min, T * cooldown)
            
            # Early stop if stuck for too long
            if no_improve > 5000:
                break
        
        return best_pts, best_e

    # ---- Main ----
    # Try both seeds and use the better one
    seed1 = build_hex_seed()
    seed2 = build_greedy_seed()
    
    if not is_valid(seed1):
        seed1 = seed2
    if not is_valid(seed2):
        seed2 = seed1
    
    e1 = count_edges(seed1)
    e2 = count_edges(seed2)
    
    if e1 >= e2:
        seed = seed1
    else:
        seed = seed2
    
    best_pts, best_e = fast_local_search(seed, deadline - 20.0)
    
    if best_e == 76:
        return [[a, b] for a, b in best_pts]

    # Multiple annealing passes with polishing
    while time.time() < deadline - 15.0:
        # Animate from current best
        cand_pts, cand_e = anneal(best_pts, deadline - 15.0)
        if cand_e > best_e:
            best_pts, best_e = cand_pts, cand_e
            if best_e == 76:
                break
            # Polish with fast local search
            polished, pe = fast_local_search(best_pts, deadline - 15.0)
            if pe > best_e:
                best_pts, best_e = polished, pe
                if best_e == 76:
                    break
        else:
            # Try fresh seed occasionally
            if random.random() < 0.3:
                fresh = build_greedy_seed()
                if is_valid(fresh):
                    fp, fe = fast_local_search(fresh, deadline - 15.0)
                    if fe > best_e:
                        best_pts, best_e = fp, fe
                        if best_e == 76:
                            break

    if not is_valid(best_pts):
        return [[a, b] for a in range(16) for b in range(2)]

    return [[a, b] for a, b in best_pts]
# EVOLVE-BLOCK-END
