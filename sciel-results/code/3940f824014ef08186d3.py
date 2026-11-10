"""No-isosceles set: maximize points in [0,64)^2 with no three forming an isosceles triangle."""
import time
import random
import math

N = 64
TIME_LIMIT = 280  # Safety margin below 330s

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    random.seed(42)
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    def dist2(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy

    def fast_is_valid_addition(p, pts, dist_counts):
        """
        Check if adding p to pts creates any isosceles triangle.
        pts: list of points
        dist_counts: list of dicts, dist_counts[i] maps distance to count of points in pts at that distance from pts[i]
        """
        k = len(pts)
        if k < 2:
            return True, {}
        
        dists_from_p = {}
        
        for i, a in enumerate(pts):
            d = dist2(p, a)
            
            # Check vertex a: if any point in pts is at distance d from a
            if dist_counts[i].get(d, 0) >= 1:
                return False, {}
            
            # Check vertex p: if any other point in pts is at distance d from p
            if d in dists_from_p:
                return False, {}
            dists_from_p[d] = 1
            
        return True, dists_from_p

    def update_dist_counts(pts, dist_counts, new_p):
        """
        Update dist_counts after adding new_p to pts.
        Returns new dist_counts list.
        """
        # Copy existing dicts
        new_dist_counts = [dict(dc) for dc in dist_counts]
        
        # Update existing points
        for i, a in enumerate(pts):
            d = dist2(new_p, a)
            new_dist_counts[i][d] = new_dist_counts[i].get(d, 0) + 1
            
        # Create dict for new_p
        new_dc = {}
        for a in pts:
            d = dist2(new_p, a)
            new_dc[d] = new_dc.get(d, 0) + 1
        new_dist_counts.append(new_dc)
        
        return new_dist_counts

    def full_validity_check(pts):
        k = len(pts)
        if k < 3:
            return True
        for i in range(k):
            for j in range(i+1, k):
                for l in range(j+1, k):
                    a, b, c = pts[i], pts[j], pts[l]
                    d_ab = dist2(a, b)
                    d_bc = dist2(b, c)
                    d_ca = dist2(c, a)
                    if d_ab == d_bc or d_bc == d_ca or d_ca == d_ab:
                        return False
        return True

    best_overall = []
    best_overall_len = 0

    # Run multiple restarts
    for restart in range(100):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        pts = []
        dist_counts = []
        candidates = all_points[:]
        random.shuffle(candidates)
        
        # Greedy phase
        for p in candidates:
            if len(pts) >= 30:
                break
            
            valid, dists_from_p = fast_is_valid_addition(p, pts, dist_counts)
            if valid:
                pts.append(p)
                dist_counts = update_dist_counts(pts[:-1], dist_counts, p)
        
        if len(pts) > best_overall_len:
            best_overall = pts[:]
            best_overall_len = len(pts)
            
        # Local search phase
        improved = True
        while improved and time.time() - start_time < TIME_LIMIT - 5:
            improved = False
            
            # Try to add a point
            for p in all_points:
                if p in pts:
                    continue
                valid, _ = fast_is_valid_addition(p, pts, dist_counts)
                if valid:
                    pts.append(p)
                    dist_counts = update_dist_counts(pts[:-1], dist_counts, p)
                    improved = True
                    break # Restart loop to try adding more
            
            # Try to swap a point
            if not improved:
                # Random swap
                if len(pts) > 0:
                    idx = random.randint(0, len(pts) - 1)
                    old_p = pts[idx]
                    new_p = random.choice(all_points)
                    if new_p not in pts:
                        # Check if swapping old_p with new_p is valid
                        # Remove old_p, add new_p
                        # This requires checking validity of new_p with pts without old_p
                        # And validity of pts without old_p (which is valid)
                        
                        # To check validity of new_p with pts\{old_p}, we need dist_counts for pts\{old_p}
                        # This is expensive to compute.
                        # Let's skip swaps for now and rely on greedy + add.
                        pass

    # Final verification
    if not full_validity_check(best_overall):
        return [(0, 0)]
    
    return best_overall
# EVOLVE-BLOCK-END
