"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def _dist2(a, b):
    """Squared distance between two lattice points (a,b) and (c,d).
    Point = a + b*omega where omega = exp(i*pi/3) = 1/2 + i*sqrt(3)/2.
    So point = (a + b/2) + i*b*sqrt(3)/2.
    Distance squared between (a,b) and (c,d):
    dx = (a-c) + (b-d)/2
    dy = (b-d)*sqrt(3)/2
    dist2 = dx^2 + dy^2 = (a-c+(b-d)/2)^2 + 3(b-d)^2/4
    = (a-c)^2 + (a-c)(b-d) + (b-d)^2/4 + 3(b-d)^2/4
    = (a-c)^2 + (a-c)(b-d) + (b-d)^2
    """
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

def _valid(points):
    """Check all pairs have distance >= 1."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            if _dist2(points[i], points[j]) < 1:
                return False
    return True

def _neighbors(a, b):
    """The 6 neighbors of (a,b) on the triangular lattice at distance 1."""
    # (a,b) + (da, db) where da^2 + da*db + db^2 == 1
    # Solutions: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    return [(a+1, b), (a, b+1), (a-1, b+1), (a-1, b), (a, b-1), (a+1, b-1)]

def _harborth(n):
    return math.floor(3*n - math.sqrt(12*n - 3))

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    n = 32
    target_edges = _harborth(n)  # 76
    
    best_points = None
    best_edges = 0
    
    # Strategy: Greedy construction with multiple random restarts
    # Build a set of 32 points maximizing edge count
    
    for restart in range(50000):
        if time.time() > deadline:
            break
        
        # Random starting point near origin
        start_a = random.randint(-5, 5)
        start_b = random.randint(-5, 5)
        
        points = [(start_a, start_b)]
        used = set(points)
        
        # Greedily add points that maximize edge count
        for step in range(1, n):
            if time.time() > deadline:
                break
            
            # Candidate positions: neighbors of existing points, plus some random nearby
            candidates = set()
            for (a, b) in points:
                for (na, nb) in _neighbors(a, b):
                    if (na, nb) not in used:
                        candidates.add((na, nb))
            
            # Also add some random candidates in a wider area
            for _ in range(20):
                ra = start_a + random.randint(-8, 8)
                rb = start_b + random.randint(-8, 8)
                if (ra, rb) not in used:
                    candidates.add((ra, rb))
            
            if not candidates:
                break
            
            # Score each candidate by number of edges it would add
            best_score = -1
            best_candidate = None
            for cand in candidates:
                # Check minimum distance constraint
                valid = True
                for p in points:
                    if _dist2(cand, p) < 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Count edges this candidate adds
                edges_added = 0
                for p in points:
                    if _dist2(cand, p) == 1:
                        edges_added += 1
                
                if edges_added > best_score:
                    best_score = edges_added
                    best_candidate = cand
            
            if best_candidate is None:
                # Try to find any valid candidate
                for cand in candidates:
                    valid = True
                    for p in points:
                        if _dist2(cand, p) < 1:
                            valid = False
                            break
                    if valid:
                        best_candidate = cand
                        break
            
            if best_candidate is None:
                break
            
            points.append(best_candidate)
            used.add(best_candidate)
        
        if len(points) < n:
            continue
        
        # Evaluate
        edges = _edge_count(points)
        if edges > best_edges:
            best_edges = edges
            best_points = points[:]
        
        if best_edges >= target_edges:
            break
    
    if best_points is None:
        # Fallback: simple grid
        best_points = [(a, b) for a in range(16) for b in range(2)]
    
    # Convert to list of lists
    return [[a, b] for (a, b) in best_points]
# EVOLVE-BLOCK-END
