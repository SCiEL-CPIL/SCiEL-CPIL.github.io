# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Helper: squared distance between two lattice points
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + da*db + db*db

    # Count edges (pairs at distance exactly 1, i.e., dist_sq == 1)
    def count_edges(pts):
        edges = 0
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(pts[i], pts[j]) == 1:
                    edges += 1
        return edges
    
    # Check validity: all distinct, all at mutual distance >= 1 (dist_sq >= 1)
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
            for j in range(i+1, 32):
                if dist_sq(pts[i], pts[j]) < 1:
                    return False
        return True

    # Generate initial hexagonal spiral
    def generate_initial():
        points = []
        points.append((0, 0))
        
        def ring_points(r):
            if r == 0:
                return [(0, 0)]
            pts = []
            dirs = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
            pos = (r, 0)
            for d in dirs:
                for _ in range(r):
                    pts.append(pos)
                    pos = (pos[0] + d[0], pos[1] + d[1])
            return pts
        
        for r in range(1, 5):
            ring = ring_points(r)
            for p in ring:
                if len(points) < 32:
                    points.append(p)
                else:
                    break
            if len(points) >= 32:
                break
        return points[:32]

    # Simulated Annealing
    def simulated_annealing(initial_pts, max_time):
        pts = list(initial_pts)
        best_pts = list(pts)
        current_edges = count_edges(pts)
        best_edges = current_edges
        
        # Precompute neighbors for fast delta calculation? 
        # Or just recalculate delta edges for the moved point.
        
        # To speed up, we can maintain a set of points for O(1) lookup?
        # But we need to check distance to all others for validity.
        
        T = 1.0
        alpha = 0.9995
        min_T = 0.01
        
        # Directions for moves
        # We can try moving to any of the 6 neighbors, or a random offset within radius 2-3
        move_offsets = []
        for r in range(1, 4):
            for a in range(-r, r+1):
                for b in range(-r, r+1):
                    if a == 0 and b == 0: continue
                    if dist_sq((0,0), (a,b)) <= r*r: # Approximate bounding box
                        move_offsets.append((a,b))
        # Filter to valid lattice points within radius
        valid_offsets = []
        for a, b in move_offsets:
            if dist_sq((0,0), (a,b)) <= 4: # Radius 2
                valid_offsets.append((a,b))
        
        # Ensure we have some offsets
        if not valid_offsets:
            valid_offsets = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]

        while time.time() < max_time:
            if T < min_T:
                break
            
            i = random.randint(0, 31)
            old = pts[i]
            
            # Pick a random offset
            da, db = random.choice(valid_offsets)
            new = (old[0] + da, old[1] + db)
            
            # Check bounds
            if abs(new[0]) > 64 or abs(new[1]) > 64:
                continue
                
            # Check if new position is occupied
            # We can use a set for points for O(1) check, but we need to update it.
            # For simplicity and speed, let's just check against list. n=32 is small.
            occupied = False
            for j in range(32):
                if j != i and pts[j] == new:
                    occupied = True
                    break
            
            if occupied:
                continue
            
            # Check validity: distance >= 1 from all others
            # Since new is a lattice point and distinct from others, dist >= 1 is guaranteed 
            # UNLESS new coincides with another point (handled above).
            # Wait, distinct lattice points always have dist >= 1. 
            # So validity is just "distinct" and "bounds".
            
            # Calculate change in edges
            # Remove edges involving old, add edges involving new
            old_edges = 0
            new_edges = 0
            for j in range(32):
                if j == i:
                    continue
                if dist_sq(old, pts[j]) == 1:
                    old_edges += 1
                if dist_sq(new, pts[j]) == 1:
                    new_edges += 1
            
            delta_edges = new_edges - old_edges
            delta_E = -delta_edges # Energy decreases if edges increase
            
            if delta_E < 0:
                # Improvement
                pts[i] = new
                current_edges += delta_edges
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_pts = list(pts)
            else:
                # Worse or equal
                # Accept with probability exp(-delta_E / T)
                if random.random() < math.exp(-delta_E / T):
                    pts[i] = new
                    current_edges += delta_edges
            
            T *= alpha
        
        return best_pts, best_edges

    # Run SA
    initial = generate_initial()
    max_time = deadline - 2.0
    
    # Multiple restarts to escape local optima
    best_overall = []
    best_overall_edges = 0
    
    # We can run SA multiple times with different seeds or initial states
    # Since we have time, let's run it a few times.
    
    # First run with default seed
    random.seed(42)
    res_pts, res_edges = simulated_annealing(initial, max_time / 3.0)
    if res_edges > best_overall_edges:
        best_overall_edges = res_edges
        best_overall = res_pts
        
    # Second run with different seed
    random.seed(123)
    res_pts, res_edges = simulated_annealing(initial, max_time / 3.0)
    if res_edges > best_overall_edges:
        best_overall_edges = res_edges
        best_overall = res_pts
        
    # Third run with different seed
    random.seed(999)
    res_pts, res_edges = simulated_annealing(initial, max_time / 3.0)
    if res_edges > best_overall_edges:
        best_overall_edges = res_edges
        best_overall = res_pts

    # Final local search on the best result to polish
    def local_search_polish(pts, max_time):
        pts = list(pts)
        best_edges = count_edges(pts)
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(32):
                if time.time() > max_time:
                    break
                old = pts[i]
                # Try all 6 neighbors
                for da, db in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
                    cand = (old[0]+da, old[1]+db)
                    if abs(cand[0]) > 64 or abs(cand[1]) > 64:
                        continue
                    if cand in pts:
                        continue
                    
                    # Check validity (distinctness is handled by 'in pts', dist>=1 is automatic)
                    
                    old_edges = 0
                    new_edges = 0
                    for j in range(32):
                        if j == i:
                            continue
                        if dist_sq(old, pts[j]) == 1:
                            old_edges += 1
                        if dist_sq(cand, pts[j]) == 1:
                            new_edges += 1
                            
                    delta = new_edges - old_edges
                    if delta > 0:
                        pts[i] = cand
                        best_edges += delta
                        improved = True
                        break # Restart scan
        return pts, best_edges

    # Polish the best result
    final_pts, final_edges = local_search_polish(best_overall, max_time)
    
    if not is_valid(final_pts):
        return [[a, b] for a in range(16) for b in range(2)]
        
    return [[a, b] for a, b in final_pts]
# EVOLVE-BLOCK-END
