<<<<<<< SEARCH
"""Improved program: local search to maximize max multiplicity of doubled triangle area."""
import random
import time
import numpy as np
from collections import Counter

def _score(points):
    """Compute the largest multiplicity of a positive doubled area among all triples using NumPy."""
    # points is a list of tuples, convert to numpy array for vectorization
    pts = np.array(points)
    n = len(pts)
    
    # Generate indices for all triples i < j < k
    # This creates arrays of indices for i, j, k
    i_idx, j_idx, k_idx = np.indices((n, n, n)).reshape(3, -1)
    
    # Filter for i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    # Get coordinates
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    
    # Calculate determinants (doubled areas)
    # det = (xj - xi)*(yk - yi) - (yj - yi)*(xk - xi)
    dx_ij = pj[:, 0] - pi[:, 0]
    dy_ij = pj[:, 1] - pi[:, 1]
    dx_ik = pk[:, 0] - pi[:, 0]
    dy_ik = pk[:, 1] - pi[:, 1]
    
    dets = dx_ij * dy_ik - dy_ij * dx_ik
    
    # Filter out zero areas
    non_zero_dets = dets[dets != 0]
    
    if len(non_zero_dets) == 0:
        return 0
        
    # Count multiplicities
    # np.unique with return_counts is efficient
    values, counts = np.unique(non_zero_dets, return_counts=True)
    
    return int(np.max(counts))
=======
"""Improved program: local search to maximize max multiplicity of doubled triangle area."""
import random
import time
import numpy as np
from collections import Counter

def _score(points):
    """Compute the largest multiplicity of a positive doubled area among all triples using NumPy."""
    pts = np.array(points, dtype=np.int32)
    n = len(pts)
    
    # Precompute all pairwise differences and their cross products for speed
    # For triples i<j<k, area = 0.5 * |cross(Pj-Pi, Pk-Pi)|
    # We can compute this more efficiently
    
    # Generate all triples i < j < k
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    
    i_idx = np.array(i_idx, dtype=np.int32)
    j_idx = np.array(j_idx, dtype=np.int32)
    k_idx = np.array(k_idx, dtype=np.int32)
    
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    
    dx_ij = pj[:, 0].astype(np.int32) - pi[:, 0].astype(np.int32)
    dy_ij = pj[:, 1].astype(np.int32) - pi[:, 1].astype(np.int32)
    dx_ik = pk[:, 0].astype(np.int32) - pi[:, 0].astype(np.int32)
    dy_ik = pk[:, 1].astype(np.int32) - pi[:, 1].astype(np.int32)
    
    dets = dx_ij * dy_ik - dy_ij * dx_ik
    
    # Filter out zero areas
    non_zero_dets = dets[dets != 0]
    
    if len(non_zero_dets) == 0:
        return 0
        
    # Count multiplicities
    values, counts = np.unique(non_zero_dets, return_counts=True)
    
    return int(np.max(counts))
>>>>>>> REPLACE
<<<<<<< SEARCH
def construct():
    # Set deadline to 50s to be safe against the 60s kill
    deadline = time.time() + 50
    
    # Start with a reasonable initial configuration
    # Use a grid-like structure that might produce repeated areas
    def _initial_points():
        # 6x5 grid
        pts = [(x, y) for x in range(6) for y in range(5)]
        return pts
    
    points = _initial_points()
    best_score = _score(points)
    best_points = points[:]
    
    # Local search: randomly perturb one point, keep if improves
    rng = random.Random(42)
    
    def _random_point():
        return (rng.randint(-30, 30), rng.randint(-30, 30))
    
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
    
    # Simulated annealing style local search
    current_score = best_score
    current_points = best_points[:]
    temp = 100.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = rng.randint(0, 29)
        old_pt = current_points[idx]
        new_pt = _random_point()
        
        # Try the move
        current_points[idx] = new_pt
        
        if not _valid(current_points):
            current_points[idx] = old_pt
            continue
        
        new_score = _score(current_points)
        
        if new_score >= current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points[:]
        else:
            # Accept with some probability (annealing)
            if rng.random() < temp:
                # Keep the move
                pass
            else:
                current_points[idx] = old_pt
        
        temp *= cooling
        if temp < 1e-4:
            # Reset temperature occasionally
            temp = 50.0
            current_points = best_points[:]
            current_score = best_score
            
        # Check time again to avoid overshooting during long score calcs
        if time.time() > deadline - 5:
            break

    return best_points
=======
def construct():
    # Set deadline to 80s to be safe against the 90s kill
    deadline = time.time() + 80
    
    # Start with a reasonable initial configuration
    # Use a grid-like structure that might produce repeated areas
    def _initial_points():
        # 6x5 grid
        pts = [(x, y) for x in range(6) for y in range(5)]
        return pts
    
    points = _initial_points()
    best_score = _score(points)
    best_points = points[:]
    
    # Local search: randomly perturb one point, keep if improves
    rng = random.Random(42)
    
    def _random_point():
        return (rng.randint(-30, 30), rng.randint(-30, 30))
    
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
    
    # Simulated annealing style local search
    current_score = best_score
    current_points = best_points[:]
    temp = 10.0
    cooling = 0.995
    
    # Track best score for early termination
    no_improve_streak = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = rng.randint(0, 29)
        old_pt = current_points[idx]
        new_pt = _random_point()
        
        # Try the move
        current_points[idx] = new_pt
        
        if not _valid(current_points):
            current_points[idx] = old_pt
            continue
        
        new_score = _score(current_points)
        
        if new_score >= current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points[:]
                no_improve_streak = 0
            else:
                no_improve_streak += 1
        else:
            # Accept with some probability (annealing)
            if rng.random() < temp:
                # Keep the move
                pass
            else:
                current_points[idx] = old_pt
            no_improve_streak += 1
        
        temp *= cooling
        if temp < 0.01:
            # Reset temperature occasionally
            temp = 5.0
            current_points = best_points[:]
            current_score = best_score
            no_improve_streak = 0
            
        # Check time again to avoid overshooting during long score calcs
        if time.time() > deadline - 10:
            break
        
        # Early termination if stuck
        if no_improve_streak > max_no_improve:
            # Reset to best and try fresh
            current_points = best_points[:]
            current_score = best_score
            temp = 5.0
            no_improve_streak = 0
            max_no_improve = min(max_no_improve * 2, 5000)

    return best_points
>>>>>>> REPLACE
