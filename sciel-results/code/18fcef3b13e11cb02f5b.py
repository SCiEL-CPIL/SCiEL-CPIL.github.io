"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # 20s safety margin before 160s limit
    
    N = 32
    MAX_COORD = 64
    
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da * da + db * db + da * db
    
    def count_edges(points):
        """Count pairs at distance exactly 1."""
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count
    
    def is_valid(points):
        """Check all pairs have distance >= 1 (i.e., dist_sq >= 1)."""
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if dist_sq(points[i], points[j]) < 1:
                    return False
        return True
    
    def hex_cluster(r):
        """Generate hex cluster of radius r centered at origin."""
        pts = set()
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                # Hex cluster: |a| + |b| + |a+b| <= 2r... actually for triangular lattice
                # The hex cluster in axial coords: max(|a|,|b|,|a+b|) <= r
                # But our coords are a + b*omega, so let's use: |a|<=r, |b|<=r, |a+b|<=r... 
                # Actually for triangular lattice with omega = exp(i*pi/3):
                # Point a+b*omega. The 6 neighbors are (a±1,b), (a,b±1), (a+1,b-1), (a-1,b+1)
                # Wait, let me reconsider. omega = exp(i*pi/3) = 1/2 + i*sqrt(3)/2
                # So a + b*omega = a + b/2 + i*b*sqrt(3)/2
                # Distance between (a1,b1) and (a2,b2):
                # dx = (a1-a2) + (b1-b2)/2
                # dy = (b1-b2)*sqrt(3)/2
                # d^2 = dx^2 + dy^2 = (a1-a2)^2 + (a1-a2)(b1-b2) + (b1-b2)^2
                # So dist_sq = da^2 + da*db + db^2 where da=a1-a2, db=b1-b2
                # Neighbors at dist_sq=1: (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)
                # Hex cluster: use the condition that max of certain coords <= r
                # In the triangular lattice, a hex of radius r contains all points within
                # "hex distance" r. The hex distance is max(|a|,|b|,|a+b|) in some coord system.
                # Let me just use: for the hex cluster, I'll enumerate and check.
                # Actually simpler: generate all points in a bounding box and keep those
                # whose "hex distance" from origin <= r.
                # Hex distance in our coords: for point (a,b), the distance to origin in terms
                # of number of unit steps. This is max(|a|,|b|,|a+b|) ... let me verify.
                # Origin (0,0). Point (1,0): dist=1. max(1,0,1)=1. OK.
                # Point (1,1): dist_sq = 1+1+1=3, so dist=sqrt(3). Not a unit neighbor.
                # Point (1,-1): dist_sq=1-1+1=1. Unit neighbor. max(1,-1,0)=1. OK.
                # Point (2,0): dist_sq=4. dist=2. max(2,0,2)=2. OK.
                # Point (2,-1): dist_sq=4-2+1=3. dist=sqrt(3). max(2,-1,1)=2. Hmm.
                # Actually the hex distance (minimum number of unit steps) for (a,b) is:
                # It's the minimum of |a|+|b|+|a+b| ... no.
                # Let me just use: a point (a,b) is within hex radius r if
                # |a| <= r and |b| <= r and |a+b| <= r... let me check.
                # (2,-1): |2|=2, |-1|=1, |2+(-1)|=1. If r=2, all <=2. But dist=sqrt(3)~1.73.
                # Is (2,-1) within 2 steps of origin? Path: (0,0)->(1,0)->(2,-1)? 
                # (1,0) to (2,-1): da=1,db=-1, dist_sq=1. Yes! So it IS within 2 steps.
                # (2,0): |2|=2,|0|=0,|2|=2. Within r=2. dist=2. Yes, 2 steps.
                # (1,1): |1|=1,|1|=1,|2|=2. Within r=2? dist_sq=3, dist=sqrt(3)~1.73.
                # Can we reach (1,1) in 2 steps? (0,0)->(1,0)->(1,1)? (1,0) to (1,1): da=0,db=1,dist_sq=1. Yes!
                # So (1,1) is within 2 steps. And max(1,1,2)=2. OK.
                # (0,2): |0|=0,|2|=2,|2|=2. dist_sq=4, dist=2. 2 steps. OK.
                # So the condition max(|a|,|b|,|a+b|) <= r seems to give the hex cluster.
                if max(abs(a), abs(b), abs(a + b)) <= r:
                    pts.add((a, b))
        return list(pts)
    
    # Generate initial hex cluster of radius 3 (37 points), then remove 5
    full_hex = hex_cluster(3)
    assert len(full_hex) == 37, f"Expected 37, got {len(full_hex)}"
    
    random.seed(42)
    
    best_points = None
    best_edges = -1
    
    # Try multiple random removals of 5 points from the 37-point hex
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        idxs = list(range(37))
        random.shuffle(idxs)
        remove_set = set(idxs[:5])
        points = [full_hex[i] for i in range(37) if i not in remove_set]
        
        if not is_valid(points):
            continue
        
        edges = count_edges(points)
        if edges > best_edges:
            best_edges = edges
            best_points = points[:]
        
        if best_edges >= 76:
            break
    
    # Local search / simulated annealing
    if best_points is not None:
        points = best_points[:]
        current_edges = best_edges
        
        # Neighbor offsets for the 6 unit-distance neighbors
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
        
        # Generate candidate positions near the current configuration
        # For each point, generate candidate positions within a small radius
        def get_candidates(point, radius=3):
            """Generate candidate lattice positions within radius of point."""
            cands = []
            for da in range(-radius, radius + 1):
                for db in range(-radius, radius + 1):
                    na, nb = point[0] + da, point[1] + db
                    if abs(na) <= MAX_COORD and abs(nb) <= MAX_COORD:
                        cands.append((na, nb))
            return cands
        
        # Precompute set of occupied positions for quick lookup
        def make_occ_set(points):
            return set(tuple(p) for p in points)
        
        # Try to improve via single-point moves
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                orig = points[i]
                # Try moving point i to a nearby position
                cands = get_candidates(orig, radius=2)
                random.shuffle(cands)
                for cand in cands:
                    if time.time() > deadline - 5:
                        break
                    if cand == tuple(orig):
                        continue
                    if cand in make_occ_set(points[:i] + points[i+1:]):
                        continue
                    # Check validity: new position must be >=1 from all other points
                    valid = True
                    for j in range(N):
                        if j == i:
                            continue
                        if dist_sq(cand, points[j]) < 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Compute new edge count
                    new_points = points[:]
                    new_points[i] = cand
                    new_edges = count_edges(new_points)
                    if new_edges > current_edges:
                        points = new_points
                        current_edges = new_edges
                        best_edges = max(best_edges, current_edges)
                        best_points = points[:]
                        improved = True
                        if current_edges >= 76:
                            break
                if current_edges >= 76:
                    break
        
        # Try swaps
        if current_edges < 76 and time.time() < deadline - 5:
            for it in range(5000):
                if time.time() > deadline - 5:
                    break
                i, j = random.sample(range(N), 2)
                pi, pj = points[i], points[j]
                points[i], points[j] = pj, pi
                if is_valid(points):
                    new_edges = count_edges(points)
                    if new_edges > current_edges:
                        current_edges = new_edges
                        best_edges = max(best_edges, current_edges)
                        best_points = points[:]
                    else:
                        points[i], points[j] = pi, pj
                else:
                    points[i], points[j] = pi, pj
        
        # Final local search with more aggressive moves
        if current_edges < 76 and time.time() < deadline - 3:
            # Try moving each point to many candidate positions
            for i in range(N):
                if time.time() > deadline - 3:
                    break
                orig = points[i]
                cands = get_candidates(orig, radius=4)
                random.shuffle(cands)
                for cand in cands:
                    if time.time() > deadline - 3:
                        break
                    if cand == tuple(orig):
                        continue
                    others = [points[j] for j in range(N) if j != i]
                    if tuple(cand) in set(tuple(p) for p in others):
                        continue
                    valid = all(dist_sq(cand, p) >= 1 for p in others)
                    if not valid:
                        continue
                    new_points = points[:]
                    new_points[i] = cand
                    new_edges = count_edges(new_points)
                    if new_edges > current_edges:
                        points = new_points
                        current_edges = new_edges
                        best_edges = max(best_edges, current_edges)
                        best_points = points[:]
    
    if best_points is not None:
        return [[p[0], p[1]] for p in best_points]
    
    # Fallback: return the 16x2 strip
    return [[a, b] for a in range(16) for b in range(2)]
# EVOLVE-BLOCK-END
