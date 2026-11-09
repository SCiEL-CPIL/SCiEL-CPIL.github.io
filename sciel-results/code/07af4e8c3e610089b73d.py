import time
import math

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 150.0
    
    def get_points_hex(R):
        # Generate points in a hexagon of radius R
        # Using axial coordinates or similar?
        # Let's use the definition: a point (a,b) is in H_R if 
        # the minimum number of steps to origin using steps (1,0), (0,1), (1,-1) etc is <= R.
        # This is equivalent to max(|a|, |b|, |a+b|) <= R? 
        # Let's test R=1.
        # Points: (0,0), (1,0), (0,1), (1,-1), (-1,0), (0,-1), (-1,1).
        # Check (1,1): max(1,1,2)=2 > 1. Correct.
        # Check (1,0): max(1,0,1)=1 <= 1. Correct.
        # Check (-1,1): max(-1,1,0) -> abs: 1,1,1. Max 1. Correct.
        # So condition is max(abs(a), abs(b), abs(a+b)) <= R.
        pts = []
        for a in range(-R, R+1):
            for b in range(-R, R+1):
                if max(abs(a), abs(b), abs(a+b)) <= R:
                    pts.append((a,b))
        return pts

    def count_edges(pts):
        pt_set = set(pts)
        edges = 0
        # Neighbors: (1,0), (0,1), (1,-1), (-1,0), (0,-1), (-1,1)
        dirs = [(1,0), (0,1), (1,-1), (-1,0), (0,-1), (-1,1)]
        for p in pts:
            for d in dirs:
                q = (p[0]+d[0], p[1]+d[1])
                if q in pt_set:
                    edges += 1
        return edges // 2 # Each edge counted twice

    def get_degrees(pts):
        pt_set = set(pts)
        degrees = {}
        dirs = [(1,0), (0,1), (1,-1), (-1,0), (0,-1), (-1,1)]
        for p in pts:
            deg = 0
            for d in dirs:
                q = (p[0]+d[0], p[1]+d[1])
                if q in pt_set:
                    deg += 1
            degrees[p] = deg
        return degrees

    # Strategy 1: Hexagon R=3, prune 5 lowest degree points
    hex_pts = get_points_hex(3)
    # Verify count
    if len(hex_pts) == 37:
        degrees = get_degrees(hex_pts)
        # Sort by degree, then by distance from center (to keep shape compact)
        # Actually, just sort by degree. If tie, arbitrary.
        sorted_pts = sorted(hex_pts, key=lambda p: degrees[p])
        # Remove 5
        to_remove = set(sorted_pts[:5])
        candidate_pts = [p for p in hex_pts if p not in to_remove]
        
        if len(candidate_pts) == 32:
            e_count = count_edges(candidate_pts)
            if e_count >= 76:
                # Check bounds
                if all(abs(a) <= 64 and abs(b) <= 64 for a,b in candidate_pts):
                    return [[a, b] for a, b in candidate_pts]
    
    # Strategy 2: Fallback to known optimal strip (Seed)
    # 16x2 strip
    return [[a, b] for a in range(16) for b in range(2)]
