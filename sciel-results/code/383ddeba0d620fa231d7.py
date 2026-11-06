import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_score = 0
    best_points = None
    
    def calc_score(pts):
        # pts is a list of tuples or lists
        n = len(pts)
        areas = []
        # Optimization: Unpack for speed?
        # Just use local variables
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i+1, n):
                x2, y2 = pts[j]
                for k in range(j+1, n):
                    x3, y3 = pts[k]
                    det = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
                    adet = abs(det)
                    if adet > 0:
                        areas.append(adet)
        if not areas:
            return 0
        return max(Counter(areas).values())
        
    def local_search(initial_pts, time_limit):
        # initial_pts: list of tuples
        current_pts = [list(p) for p in initial_pts]
        current_set = set(map(tuple, current_pts))
        current_score = calc_score(current_pts)
        
        best_local_score = current_score
        best_local_pts = [tuple(p) for p in current_pts]
        
        radius = 3
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Randomize order of points
            indices = list(range(30))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > time_limit:
                    break
                x, y = current_pts[idx]
                old_pt = (x, y)
                
                # Generate neighbors
                neighbors = []
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if -30 <= nx <= 30 and -30 <= ny <= 30:
                            new_pt = (nx, ny)
                            # Check collision
                            # If new_pt is in current_set, and new_pt != old_pt, it's a collision
                            # Wait, if new_pt == old_pt, we skipped dx=0,dy=0
                            # So if new_pt in current_set, it must be another point
                            if new_pt in current_set:
                                continue
                            neighbors.append(new_pt)
                
                # Shuffle neighbors to avoid bias
                random.shuffle(neighbors)
                
                for new_pt in neighbors:
                    if time.time() > time_limit:
                        break
                    
                    # Apply move
                    current_pts[idx] = list(new_pt)
                    current_set.remove(old_pt)
                    current_set.add(new_pt)
                    
                    new_score = calc_score(current_pts)
                    
                    if new_score > current_score:
                        current_score = new_score
                        if current_score > best_local_score:
                            best_local_score = current_score
                            best_local_pts = [tuple(p) for p in current_pts]
                        improved = True
                        # Accept move and break to next point?
                        # Usually in hill climbing, we accept the first improvement or best improvement.
                        # Accepting first improvement is faster.
                        break 
                    else:
                        # Revert
                        current_pts[idx] = [x, y]
                        current_set.remove(new_pt)
                        current_set.add(old_pt)
                        
        return best_local_score, best_local_pts

    # Initial Candidates
    candidates = []
    # 5x6
    candidates.append([(i, j) for i in range(5) for j in range(6)])
    # 6x5
    candidates.append([(i, j) for i in range(6) for j in range(5)])
    # Centered 5x6
    candidates.append([(i-2, j-2) for i in range(5) for j in range(6)])
    # Scaled
    candidates.append([(i*2, j*2) for i in range(5) for j in range(6)])
    
    # Random ones
    for _ in range(10):
        pts = []
        seen = set()
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x,y) not in seen:
                seen.add((x,y))
                pts.append((x,y))
        candidates.append(pts)
        
    # Run local search on each candidate
    # Allocate time based on number of candidates
    # Let's just run them sequentially until deadline
    
    for i, init_pts in enumerate(candidates):
        if time.time() > deadline - 5: # Leave 5s for safety
            break
            
        # Allocate remaining time to this candidate?
        # Or just run until deadline?
        # If we have many candidates, we should split time.
        # But hill climbing converges fast.
        # Let's give each candidate a chunk of time, or just run them all in a loop.
        
        # To be safe, let's limit each candidate to a portion of the time if many candidates.
        # But 14 candidates * 1s each = 14s. We have 85s.
        # Let's just run them. If one takes too long, it will hit the global deadline.
        
        score, pts = local_search(init_pts, deadline)
        if score > best_score:
            best_score = score
            best_points = pts
            
    if best_points is None:
        best_points = candidates[0]
        
    return best_points

BET: {mechanism: "Hill climbing from multiple grid initializations with optimized neighbor search and vectorized-ish score calculation (Python loops are fast enough for N=30) allows escaping local optima of the baseline grid, significantly increasing the max area multiplicity by finding configurations with higher symmetry or structured area repetitions.", q: 0.9, delta: [0.01, 0.1]}
