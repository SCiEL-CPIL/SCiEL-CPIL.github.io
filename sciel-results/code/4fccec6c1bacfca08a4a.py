"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Distance squared on triangular lattice
    # Points are a + b*omega, omega = exp(i*pi/3) = 1/2 + i*sqrt(3)/2
    # (a1 + b1*omega) - (a2 + b2*omega) = (a1-a2) + (b1-b2)*omega
    # |z|^2 = (a1-a2)^2 + (b1-b2)^2 + (a1-a2)(b1-b2)
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + db*db + da*db
    
    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count
    
    def hex_dist(p):
        # Hexagonal distance from origin for point (a, b) on triangular lattice
        # The six neighbors of origin are: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # Hex distance is the minimum number of steps to reach origin
        a, b = p
        # Convert to axial coordinates for hex distance
        # In triangular lattice with basis (1,0) and (1/2, sqrt(3)/2),
        # the hex distance formula: max(|a|, |b|, |a+b|) ... let me think more carefully
        # Actually for triangular lattice, the distance from origin to (a,b) is:
        # The six unit vectors are: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # Hex distance = max(|a|, |b|, |a+b|) ... no that's not right either
        # Let me use the standard formula. In a triangular lattice with coordinates (a,b),
        # the hex distance (number of unit steps) from (0,0) to (a,b) is:
        # d = max(|a|, |b|, |a+b|) is for cube coordinates in hex grids
        # For our basis, let's think: the neighbors are (±1,0), (0,±1), (1,-1), (-1,1)
        # This is equivalent to a hex grid. The distance is max(|a|, |b|, |a+b|)
        # Wait, let me verify: (1,0) -> max(1,0,1)=1 ✓
        # (0,1) -> max(0,1,1)=1 ✓
        # (1,1) -> max(1,1,2)=2. Is (1,1) at distance 2? 
        # (1,1) - (1,0) = (0,1) which is a unit step, and (1,0) is distance 1, so (1,1) is distance 2. ✓
        # (-1,1) -> max(1,1,0)=1. Is (-1,1) a neighbor? Yes! ✓
        # (1,-1) -> max(1,1,0)=1. Is (1,-1) a neighbor? Yes! ✓
        return max(abs(a), abs(b), abs(a+b))
    
    # Generate hexagonal spiral points
    def hex_spiral(n_points):
        """Generate the first n_points of the hexagonal spiral on triangular lattice."""
        points = [(0, 0)]
        if n_points <= 1:
            return points[:n_points]
        
        # Directions in order for the spiral
        # The six directions: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        directions = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        
        ring = 1
        while len(points) < n_points:
            # Each ring has 6*ring points
            for side in range(6):
                dx, dy = directions[side]
                for step in range(ring):
                    if len(points) >= n_points:
                        break
                    # Starting point for this side
                    # For ring r, side s, the starting corner is:
                    # corner 0: (r, 0)
                    # corner 1: (0, r)  
                    # corner 2: (-r, r)
                    # corner 3: (-r, 0)
                    # corner 4: (0, -r)
                    # corner 5: (r, -r)
                    # And we move along the side direction for 'ring' steps
                    # Actually, let me use a simpler approach:
                    # The ring consists of 6 segments. For ring r:
                    # Segment 0: from (r,0) to (0,r) - direction (-1,1), r steps
                    # Segment 1: from (0,r) to (-r,r) - direction (-1,0), r steps
                    # Segment 2: from (-r,r) to (-r,0) - direction (0,-1), r steps
                    # Segment 3: from (-r,0) to (0,-r) - direction (1,-1), r steps
                    # Segment 4: from (0,-r) to (r,-r) - direction (1,0), r steps
                    # Segment 5: from (r,-r) to (r,0) - direction (0,1), r steps
                    # But we only need r-1 steps per segment to avoid duplicating corners
                    # Actually for a proper spiral, each segment has r points (including start, excluding end)
                    # Let me just generate all points in the ring by going around
                    
                    pass
            
            # Simpler approach: generate all points at hex distance == ring
            for a in range(-ring, ring+1):
                for b in range(-ring, ring+1):
                    if hex_dist((a, b)) == ring:
                        points.append((a, b))
                        if len(points) >= n_points:
                            return points
            ring += 1
        
        return points[:n_points]
    
    # Get initial hex spiral solution
    best_points = hex_spiral(32)
    best_score = count_edges(best_points)
    
    # Local search to try to improve
    # The hex spiral should already give 76 edges (Harborth bound), but let's verify
    # and try some local moves just in case
    
    if time.time() < deadline - 10:
        # Try local search: swap points or move points
        for iteration in range(5000):
            if time.time() > deadline - 10:
                break
            
            # Pick a random point and try to move it to a nearby location
            idx = random.randint(0, 31)
            old_a, old_b = best_points[idx]
            
            # Try moving to a nearby position
            moved = False
            for da in range(-3, 4):
                for db in range(-3, 4):
                    if da == 0 and db == 0:
                        continue
                    new_a, new_b = old_a + da, old_b + db
                    if abs(new_a) > 64 or abs(new_b) > 64:
                        continue
                    
                    # Check if new position is already occupied
                    if [new_a, new_b] in best_points:
                        continue
                    
                    # Check if new position maintains minimum distance 1
                    valid = True
                    for j, (pa, pb) in enumerate(best_points):
                        if j == idx:
                            continue
                        if dist_sq([new_a, new_b], [pa, pb]) < 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Try the move
                    best_points[idx] = [new_a, new_b]
                    new_score = count_edges(best_points)
                    
                    if new_score > best_score:
                        best_score = new_score
                        moved = True
                    else:
                        best_points[idx] = [old_a, old_b]
            
            if best_score == 76:
                break
    
    return [[a, b] for a, b in best_points]
# EVOLVE-BLOCK-END
