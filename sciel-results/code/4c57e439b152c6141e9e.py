"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    if n == 0:
        return True
    r = int(n ** 0.5)
    if r * r == n:
        return True
    # Handle large integers properly
    for r in (r - 1, r, r + 1):
        if r >= 0 and r * r == n:
            return True
    return False

def construct():
    start = time.time()
    deadline = start + 145  # safety margin
    
    def elapsed():
        return time.time() - start
    
    def collinear(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) == (q[1] - p[1]) * (r[0] - p[0])
    
    def concyclic(a, b, c, d):
        xa, ya = a
        xb, yb = b
        xc, yc = c
        xd, yd = d
        
        A = xa * xa + ya * ya
        B = xb * xb + yb * yb
        C = xc * xc + yc * yc
        D_ = xd * xd + yd * yd
        
        def det3(m):
            return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
                  - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
                  + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
        
        M1 = [[xb, yb, 1], [xc, yc, 1], [xd, yd, 1]]
        M2 = [[B, yb, 1], [C, yc, 1], [D_, yd, 1]]
        M3 = [[B, xb, 1], [C, xc, 1], [D_, xd, 1]]
        M4 = [[B, xb, yb], [C, xc, yc], [D_, xd, yd]]
        
        return A * det3(M1) - xa * det3(M2) + ya * det3(M3) - det3(M4)
    
    def all_distances_rational(pts):
        """Check that all pairwise squared distances are perfect squares."""
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                d2 = dx*dx + dy*dy
                if not _is_square(d2):
                    return False
        return True
    
    def check_set(pts):
        # Check all pairwise distances are rational
        if not all_distances_rational(pts):
            return False
        # Check no three collinear
        for a, b, c in combinations(pts, 3):
            if collinear(a, b, c):
                return False
        # Check no four concyclic
        for a, b, c, d in combinations(pts, 4):
            if concyclic(a, b, c, d) == 0:
                return False
        return True
    
    # Strategy: Find 6 integer points with integer pairwise distances
    # (i.e., all squared distances are perfect squares)
    # This is a "rational distance clique" problem
    
    # Start with a known valid 3-point configuration: right triangle
    # (0,0), (3,4), (0,5) - distances: 5, 5, 3... wait, let me use (0,0), (3,4), (6,0)
    # distances: 5, 6, 5 - all integers!
    
    # Let's try to build up from small valid sets
    # First, find all pairs of points in a bounded region with integer distance
    # Then find cliques of size 6
    
    # Use a smaller search space for efficiency
    R = 20  # radius
    
    # Generate candidate points
    candidates = []
    for x in range(-R, R+1):
        if elapsed() > deadline:
            break
        for y in range(-R, R+1):
            candidates.append((x, y))
    
    # Build adjacency: two points are adjacent if their distance is integer
    n = len(candidates)
    adj = [set() for _ in range(n)]
    for i in range(n):
        if elapsed() > deadline:
            break
        for j in range(i+1, n):
            dx = candidates[i][0] - candidates[j][0]
            dy = candidates[i][1] - candidates[j][1]
            d2 = dx*dx + dy*dy
            if _is_square(d2):
                adj[i].add(j)
                adj[j].add(i)
    
    # Now find a clique of size 6 with no three collinear and no four concyclic
    # Use a greedy/backtracking approach
    
    def find_clique(min_size=6):
        """Find a clique of the given size with geometric constraints."""
        # Try all starting points, then extend greedily
        for start in range(n):
            if elapsed() > deadline:
                return None
            # Get neighbors of start
            neighbors = list(adj[start])
            # Try to build a clique from start + subset of neighbors
            # Use a simple backtracking
            def backtrack(current_clique, available):
                if len(current_clique) >= min_size:
                    # Verify geometric constraints
                    pts = [candidates[i] for i in current_clique]
                    if check_set(pts):
                        return pts
                    return None
                
                if not available:
                    return None
                
                # Try each available point
                for idx, node in enumerate(available):
                    # Check if node is adjacent to all in current_clique
                    if all(node in adj[c] for c in current_clique):
                        new_clique = current_clique + [node]
                        # New available: intersection of available and adj[node]
                        new_available = [a for a in available[idx+1:] if a in adj[node]]
                        result = backtrack(new_clique, new_available)
                        if result is not None:
                            return result
                return None
            
            result = backtrack([start], neighbors)
            if result is not None:
                return result
        
        return None
    
    # Try to find a 6-clique
    result = find_clique(6)
    
    if result is not None:
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in result]}
    
    # If 6 not found, try 5
    result = find_clique(5)
    
    if result is not None:
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in result]}
    
    # Fallback: return a known valid 3-point set
    # (0,0), (3,4), (6,0) - distances: 5, 6, 5 - all integers
    # Check: no three collinear (they form a triangle), no four concyclic (only 3 points)
    pts = [(0, 0), (3, 4), (6, 0)]
    if check_set(pts):
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
    
    # Last resort: single point (trivially valid)
    return {'D': 1, 'points': [[[0, 1], [0, 1]]]}
# EVOLVE-BLOCK-END
