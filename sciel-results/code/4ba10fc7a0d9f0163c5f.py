"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _dist2(a, b):
    """Squared distance between two lattice points (a,b) and (c,d)."""
    da = a[0] - b[0]
    db = a[1] - b[1]
    return da*da + da*db + db*db

def _edge_count(points):
    """Count pairs at distance exactly 1 (dist2 == 1)."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if _dist2(points[i], points[j]) == 1:
                count += 1
    return count

def _neighbors(a, b):
    """The 6 neighbors of (a,b) on the triangular lattice at distance 1."""
    return [(a+1, b), (a, b+1), (a-1, b+1), (a-1, b), (a, b-1), (a+1, b-1)]

def _hex_dist(a, b):
    """Hexagonal distance from origin in axial coordinates (a,b) for triangular lattice."""
    # Distance d satisfies d^2 = a^2 + ab + b^2 is NOT the hex distance.
    # Hex distance (radius) r is the minimum number of steps to reach origin.
    # In axial coords (q, r) for hex grid, distance is (|q| + |r| + |q+r|) / 2.
    # Our coords: a + b*omega. 
    # Neighbors: (1,0), (0,1), (-1,1), etc.
    # This corresponds to axial coordinates where the basis vectors are at 60 deg.
    # The hex distance is max(|a|, |b|, |a+b|)? 
    # Let's check (1,0): max(1,0,1)=1. Correct.
    # (0,1): max(0,1,1)=1. Correct.
    # (1,1): dist2 = 1+1+1=3. dist = sqrt(3). 
    # Hex dist of (1,1): max(1,1,2)=2. 
    # Path (0,0)->(1,0)->(1,1) is 2 steps. Correct.
    return max(abs(a), abs(b), abs(a+b))

def _generate_hexagonal_points(n):
    """
    Generate n points forming a 'hexagonal' cluster.
    1. Add all points with hex_dist < R.
    2. Add points with hex_dist == R until n is reached.
    To maximize edges, we should add the boundary points in a way that keeps the shape convex.
    """
    points = []
    # Find max radius needed
    # 1 + 6 + 12 + ...
    # R=0: 1
    # R=1: 7
    # R=2: 19
    # R=3: 37
    # For n=32, we need R=2 full, and 13 from R=3.
    
    # Get all points with hex_dist <= 2
    for a in range(-3, 4):
        for b in range(-3, 4):
            if _hex_dist(a, b) <= 2:
                points.append((a, b))
    
    if len(points) >= n:
        return points[:n]
        
    needed = n - len(points)
    
    # Get points with hex_dist == 3
    boundary_points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if _hex_dist(a, b) == 3:
                boundary_points.append((a, b))
                
    # Sort boundary points by angle to pick a contiguous arc
    # Angle = atan2(y, x) in cartesian coords
    # Point (a,b) -> (a + b/2, b*sqrt(3)/2)
    def angle(p):
        a, b = p
        x = a + b * 0.5
        y = b * math.sqrt(3) / 2
        return math.atan2(y, x)
        
    boundary_points.sort(key=angle)
    
    # Pick 'needed' points from the sorted list.
    # To be safe against specific tie-breaking issues that might reduce edges,
    # we can try a few starting indices in the circular list and pick the one with max edges.
    # But for n=32, any contiguous arc of 13 from 18 should be optimal.
    # Let's just take the first 'needed'.
    points.extend(boundary_points[:needed])
    
    return points

def _local_search(points, max_iters=1000):
    """
    Simple local search to ensure we hit the optimum if the initial guess was suboptimal.
    Tries to swap a point with a neighbor of a neighbor to improve edge count.
    """
    n = len(points)
    best_edges = _edge_count(points)
    current_points = [tuple(p) for p in points]
    used = set(current_points)
    
    for _ in range(max_iters):
        improved = False
        # Try moving each point to a nearby empty spot
        for i in range(n):
            p = current_points[i]
            # Check neighbors of neighbors
            for np1 in _neighbors(p[0], p[1]):
                for np2 in _neighbors(np1[0], np1[1]):
                    if np2 in used or np2 == p:
                        continue
                    # Check if moving i to np2 is valid (dist >= 1 from all others)
                    valid = True
                    for j in range(n):
                        if j == i:
                            continue
                        if _dist2(np2, current_points[j]) < 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Calculate new edge count
                    # Remove edges involving i
                    old_edges_i = 0
                    for j in range(n):
                        if j == i:
                            continue
                        if _dist2(p, current_points[j]) == 1:
                            old_edges_i += 1
                            
                    # Add edges involving i at new pos
                    new_edges_i = 0
                    for j in range(n):
                        if j == i:
                            continue
                        if _dist2(np2, current_points[j]) == 1:
                            new_edges_i += 1
                            
                    delta = new_edges_i - old_edges_i
                    if delta > 0:
                        # Perform move
                        used.remove(p)
                        used.add(np2)
                        current_points[i] = np2
                        best_edges += delta
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
            
    return current_points, best_edges

def construct():
    deadline = time.time() + 140.0
    n = 32
    
    # 1. Deterministic Construction
    points = _generate_hexagonal_points(n)
    
    # 2. Evaluate
    edges = _edge_count(points)
    
    # 3. If not optimal, run local search
    if edges < 76:
        points, edges = _local_search(points)
        
    # 4. Final check
    if edges < 76:
        # Fallback to random search if deterministic fails (unlikely)
        for _ in range(1000):
            if time.time() > deadline:
                break
            random_points = []
            used = set()
            # Greedy random
            start = (random.randint(-5,5), random.randint(-5,5))
            random_points.append(start)
            used.add(start)
            for _ in range(n-1):
                candidates = set()
                for p in random_points:
                    for nb in _neighbors(p[0], p[1]):
                        if nb not in used:
                            candidates.add(nb)
                if not candidates: break
                best_c = None
                best_e = -1
                for c in candidates:
                    valid = all(_dist2(c, p) >= 1 for p in random_points)
                    if not valid: continue
                    e = sum(1 for p in random_points if _dist2(c, p) == 1)
                    if e > best_e:
                        best_e = e
                        best_c = c
                if best_c:
                    random_points.append(best_c)
                    used.add(best_c)
                else:
                    break
            if len(random_points) == n:
                e = _edge_count(random_points)
                if e > edges:
                    edges = e
                    points = random_points
                if edges >= 76:
                    break

    return [[a, b] for (a, b) in points]
# EVOLVE-BLOCK-END
