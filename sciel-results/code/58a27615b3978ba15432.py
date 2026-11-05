import math
import random
import time
from itertools import combinations

def is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def is_collinear(p1, p2, p3):
    # Cross product of (p2-p1) and (p3-p1)
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

def is_concyclic(p1, p2, p3, p4):
    # Check if 4 points are concyclic
    # Determinant of the matrix:
    # x1^2+y1^2, x1, y1, 1
    # x2^2+y2^2, x2, y2, 1
    # x3^2+y3^2, x3, y3, 1
    # x4^2+y4^2, x4, y4, 1
    # If det == 0, they are concyclic.
    # We use exact integer arithmetic.
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    r1 = x1*x1 + y1*y1
    r2 = x2*x2 + y2*y2
    r3 = x3*x3 + y3*y3
    r4 = x4*x4 + y4*y4
    
    # Calculate determinant of 4x4 matrix
    # Using Sarrus-like expansion or cofactor expansion
    # For speed and simplicity, we can use the formula for the determinant of:
    # | A B C D |
    # | E F G H |
    # | I J K L |
    # | M N O P |
    
    # Let's use a generic determinant calculation for 4x4 to be safe, 
    # or specific expansion.
    # det = A*(F*K*P - F*L*O - G*J*P + G*L*N + H*J*O - H*K*N)
    #     - B*(E*K*P - E*L*O - G*I*P + G*L*M + H*I*O - H*K*M)
    #     + C*(E*J*P - E*L*N - F*I*P + F*L*M + H*I*N - H*J*M)
    #     - D*(E*J*O - E*K*N - F*I*O + F*K*M + G*I*N - G*J*M)
    
    A, B, C, D = r1, x1, y1, 1
    E, F, G, H = r2, x2, y2, 1
    I, J, K, L = r3, x3, y3, 1
    M, N, O, P = r4, x4, y4, 1
    
    det = (
        A * (F * K * P - F * L * O - G * J * P + G * L * N + H * J * O - H * K * N)
        - B * (E * K * P - E * L * O - G * I * P + G * L * M + H * I * O - H * K * M)
        + C * (E * J * P - E * L * N - F * I * P + F * L * M + H * I * N - H * J * M)
        - D * (E * J * O - E * K * N - F * I * O + F * K * M + G * I * N - G * J * M)
    )
    return det == 0

def check_validity(points):
    n = len(points)
    # Check distinct
    if len(set(points)) != n:
        return False
    # Check distances
    for i in range(n):
        for j in range(i+1, n):
            if not is_square(dist_sq(points[i], points[j])):
                return False
    # Check collinearity
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if is_collinear(points[i], points[j], points[k]):
                    return False
    # Check concyclicity
    for quad in combinations(range(n), 4):
        if is_concyclic(points[quad[0]], points[quad[1]], points[quad[2]], points[quad[3]]):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    # Seed points
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_points = seed_points
    best_size = len(seed_points)
    
    # Try to find a larger set using local search / growth
    # We will try to grow a set from 2 points up to 6 or 7.
    # This is a heuristic search.
    
    # Strategy: Randomized local search on N points.
    # We'll try N=6 first. If found, try N=7.
    
    def local_search(N, max_iters=10000):
        # Initialize with random points in a box
        # Or start with a known valid small set and add random points
        # Let's start with a random set
        pts = [(random.randint(-50, 50), random.randint(-50, 50)) for _ in range(N)]
        # Ensure distinct
        while len(set(pts)) < N:
            pts[random.randint(0, N-1)] = (random.randint(-50, 50), random.randint(-50, 50))
            
        # Calculate initial violations
        def get_violations(pts):
            v_dist = 0
            v_coll = 0
            v_conc = 0
            for i in range(N):
                for j in range(i+1, N):
                    if not is_square(dist_sq(pts[i], pts[j])):
                        v_dist += 1
            for i in range(N):
                for j in range(i+1, N):
                    for k in range(j+1, N):
                        if is_collinear(pts[i], pts[j], pts[k]):
                            v_coll += 1
            for quad in combinations(range(N), 4):
                if is_concyclic(pts[quad[0]], pts[quad[1]], pts[quad[2]], pts[quad[3]]):
                    v_conc += 1
            return v_dist * 1000 + v_coll * 10 + v_conc
        
        current_viol = get_violations(pts)
        if current_viol == 0:
            return pts
            
        best_pts = pts
        best_viol = current_viol
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            # Pick a random point to move
            idx = random.randint(0, N-1)
            old_pt = pts[idx]
            # Move to a nearby point
            new_pt = (old_pt[0] + random.randint(-5, 5), old_pt[1] + random.randint(-5, 5))
            pts[idx] = new_pt
            
            new_viol = get_violations(pts)
            if new_viol < best_viol:
                best_viol = new_viol
                best_pts = pts[:]
                if new_viol == 0:
                    return pts
            else:
                # Revert if not better (simple hill climbing)
                # Or accept with prob exp(-delta/T) (simulated annealing)
                # For simplicity, revert
                pts[idx] = old_pt
                
        return best_pts if best_viol == 0 else None

    # Try to find sets of size 6, 7, 8
    for N in range(6, 9):
        if time.time() > deadline:
            break
        found = False
        # Multiple restarts
        for restart in range(20):
            if time.time() > deadline:
                break
            result = local_search(N, max_iters=2000)
            if result:
                if check_validity(result):
                    if N > best_size:
                        best_points = result
                        best_size = N
                    found = True
                    break
        if found:
            # If we found N, maybe try N+1? 
            # But let's be conservative and just return the largest found.
            # Actually, if we found 6, we should try 7.
            # The loop above does this.
            pass

    # If no better set found, return seed
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
