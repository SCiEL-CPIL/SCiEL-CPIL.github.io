"""Improved program: two-lattice structure with simulated annealing for max area multiplicity."""
import random
import time
import numpy as np
from collections import Counter

def _score(points):
    """Compute the largest multiplicity of a positive doubled area among all triples."""
    pts = np.array(points, dtype=np.int64)
    n = len(pts)
    
    # Precompute all pairwise differences
    # For triple (i,j,k): det = (xj-xi)*(yk-yi) - (yj-yi)*(xk-xi)
    # Use vectorized computation with broadcasting for pairs
    
    # Generate all unique triples i<j<k
    i_idx = []
    j_idx = []
    k_idx = []
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    
    if not i_idx:
        return 0
    
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    
    dx_ij = pj[:, 0] - pi[:, 0]
    dy_ij = pj[:, 1] - pi[:, 1]
    dx_ik = pk[:, 0] - pi[:, 0]
    dy_ik = pk[:, 1] - pi[:, 1]
    
    dets = dx_ij * dy_ik - dy_ij * dx_ik
    
    non_zero_dets = dets[dets != 0]
    
    if len(non_zero_dets) == 0:
        return 0
        
    values, counts = np.unique(non_zero_dets, return_counts=True)
    return int(np.max(counts))

def _make_lattice_points(base_x, base_y, size, shift_x=0, shift_y=0):
    """Create a triangular lattice subset of given size with optional shift."""
    # Triangular lattice: points (a, b) where a+b <= size-1
    # This gives a right triangle shape
    pts = []
    for a in range(size):
        for b in range(size - a):
            pts.append((a + shift_x, b + shift_y))
            if len(pts) >= 15:
                return pts
    return pts[:15]

def _initial_two_lattice(shift_x=0, shift_y=0, scale=1):
    """Create 30 points as union of two congruent triangular lattices."""
    # First lattice at origin
    lat1 = _make_lattice_points(0, 0, 6, 0, 0)
    # Second lattice shifted
    lat2 = _make_lattice_points(0, 0, 6, shift_x, shift_y)
    
    # Scale if needed
    if scale != 1:
        lat1 = [(x*scale, y*scale) for x, y in lat1]
        lat2 = [(x*scale, y*scale) for x, y in lat2]
    
    pts = lat1 + lat2
    return pts

def construct():
    deadline = time.time() + 55
    
    rng = random.Random(12345)
    
    # Try multiple initial configurations
    best_score = 0
    best_points = None
    
    # Initial candidates: different shift vectors
    initial_shifts = [
        (5, 0), (0, 5), (5, 5), (7, 3), (3, 7),
        (6, 2), (2, 6), (4, 4), (8, 1), (1, 8),
        (10, 0), (0, 10), (10, 5), (5, 10),
        (7, 7), (9, 3), (3, 9), (11, 2), (2, 11),
        (12, 0), (0, 12), (12, 4), (4, 12),
    ]
    
    for sx, sy in initial_shifts:
        if time.time() > deadline - 30:
            break
        pts = _initial_two_lattice(sx, sy)
        
        # Ensure all points are in range and distinct
        seen = set()
        valid_pts = []
        for p in pts:
            if -30 <= p[0] <= 30 and -30 <= p[1] <= 30 and p not in seen:
                seen.add(p)
                valid_pts.append(p)
        
        if len(valid_pts) < 30:
            continue
        
        # Pad with random points if needed
        while len(valid_pts) < 30:
            p = (rng.randint(-30, 30), rng.randint(-30, 30))
            if p not in seen:
                seen.add(p)
                valid_pts.append(p)
        
        valid_pts = valid_pts[:30]
        s = _score(valid_pts)
        if s > best_score:
            best_score = s
            best_points = valid_pts[:]
    
    if best_points is None:
        # Fallback: simple grid
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score = _score(best_points)
    
    # Local search with simulated annealing
    current_points = best_points[:]
    current_score = best_score
    
    temp = 50.0
    cooling = 0.998
    steps_since_improvement = 0
    max_steps_without_improvement = 2000
    
    def _random_point_near(p, radius=3):
        """Generate a random point near p within radius."""
        dx = rng.randint(-radius, radius)
        dy = rng.randint(-radius, radius)
        return (p[0] + dx, p[1] + dy)
    
    def _valid(points):
        """Check all points are distinct and in range."""
        seen = set()
        for p in points:
            if p in seen:
                return False
            seen.add(p)
            if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
                return False
        return True
    
    while time.time() < deadline - 3:
        # Pick a random point to move
        idx = rng.randint(0, 29)
        old_pt = current_points[idx]
        
        # Try moving nearby first (local refinement)
        if rng.random() < 0.7:
            new_pt = _random_point_near(old_pt, 2)
        else:
            new_pt = (rng.randint(-30, 30), rng.randint(-30, 30))
        
        current_points[idx] = new_pt
        
        if not _valid(current_points):
            current_points[idx] = old_pt
            continue
        
        new_score = _score(current_points)
        
        if new_score >= current_score:
            current_score = new_score
            steps_since_improvement = 0
            if new_score > best_score:
                best_score = new_score
                best_points = current_points[:]
        else:
            # Accept with annealing probability
            delta = current_score - new_score
            if rng.random() < temp / max(1, delta):
                pass  # Keep the move
            else:
                current_points[idx] = old_pt
            steps_since_improvement += 1
        
        temp *= cooling
        if temp < 0.1:
            # Reset: go back to best and restart with higher temp
            temp = 30.0
            current_points = best_points[:]
            current_score = best_score
            steps_since_improvement = 0
        
        if steps_since_improvement > max_steps_without_improvement:
            # Restart from best
            current_points = best_points[:]
            current_score = best_score
            steps_since_improvement = 0
            temp = max(temp, 20.0)
    
    return best_points
