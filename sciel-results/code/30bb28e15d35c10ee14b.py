"""No-isosceles set construction with vectorized validation and local search."""
import numpy as np
import time
import random

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 290 second deadline to be safe
    
    best_points = []
    best_count = 0
    
    # Fast vectorized validity check
    def is_valid_addition(points, p):
        if not points:
            return True
        
        px, py = p
        P = np.array(points, dtype=np.int64)
        
        # Squared distances from p to all points in P
        d_p = (P[:, 0] - px)**2 + (P[:, 1] - py)**2
        
        n = len(P)
        
        # Check 1: d(p, a) == d(p, b) for any a != b
        # This is equivalent to checking if there are duplicate values in d_p
        if len(set(d_p)) != n:
            return False
            
        # Squared distances between all pairs in P
        P_norm_sq = np.sum(P**2, axis=1)
        Dot = P @ P.T
        D_sq = P_norm_sq[:, np.newaxis] + P_norm_sq[np.newaxis, :] - 2 * Dot
        
        # Check 2 and 3: d(p, a) == d(a, b) or d(p, b) == d(a, b)
        # For each pair (i, j) with i < j:
        #   d_p[i] != D_sq[i, j] and d_p[j] != D_sq[i, j]
        
        for i in range(n - 1):
            js = np.arange(i + 1, n)
            if np.any(D_sq[i, js] == d_p[i]):
                return False
            if np.any(D_sq[i, js] == d_p[js]):
                return False
                
        return True

    # Greedy construction
    def greedy_construct():
        pts = []
        candidates = [(x, y) for x in range(N) for y in range(N)]
        random.shuffle(candidates)
        
        for p in candidates:
            if time.time() > deadline:
                break
            if is_valid_addition(pts, p):
                pts.append(p)
        return pts

    # Local search to improve
    def local_search(initial_pts):
        pts = list(initial_pts)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Try adding any valid point
            added = False
            for x in range(N):
                for y in range(N):
                    if time.time() > deadline:
                        return pts
                    p = (x, y)
                    if p not in pts and is_valid_addition(pts, p):
                        pts.append(p)
                        improved = True
                        added = True
                        break
                if added:
                    break
            
            # Phase 2: If no point could be added, try swapping
            if not improved:
                for i in range(len(pts)):
                    if time.time() > deadline:
                        return pts
                    old_pt = pts[i]
                    pts_temp = pts[:i] + pts[i+1:]
                    
                    # Try to find a replacement
                    for x in range(N):
                        for y in range(N):
                            if time.time() > deadline:
                                return pts
                            p = (x, y)
                            if p not in pts_temp and p != old_pt and is_valid_addition(pts_temp, p):
                                pts = pts_temp + [p]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                        
        return pts

    # Run multiple greedy constructions
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        
        pts = greedy_construct()
        if len(pts) > best_count:
            best_points = pts
            best_count = len(pts)
            
        # Apply local search if we have time and potential
        if len(pts) > best_count - 2 and time.time() < deadline - 20:
            improved_pts = local_search(pts)
            if len(improved_pts) > best_count:
                best_points = improved_pts
                best_count = len(improved_pts)
                
    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
