"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START
import sys

# Precompute all C(30,3) = 4060 triple indices
_TRIPLES = np.array(list(itertools.combinations(range(30), 3)), dtype=np.int32)
N_TRIPLES = _TRIPLES.shape[0]

def _score(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    # points: (30, 2) int array
    # doubled area for triple (i,j,k) = |det([pj-pi, pk-pi])|
    pi = points[_TRIPLES[:, 0]]
    pj = points[_TRIPLES[:, 1]]
    pk = points[_TRIPLES[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    areas = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    # Exclude zero-area
    nonzero = areas[areas > 0]
    if len(nonzero) == 0:
        return 0
    # Count multiplicities
    vals, counts = np.unique(nonzero, return_counts=True)
    return int(counts.max())

def _valid(points):
    """Check 30 distinct integer points in [-30,30]^2."""
    if len(points) != 30:
        return False
    if points.shape[1] != 2:
        return False
    if np.any(points < -30) or np.any(points > 30):
        return False
    if not np.all(np.equal(points, points.astype(np.int64))):
        return False
    # Check distinctness
    flat = set(map(tuple, points.tolist()))
    return len(flat) == 30

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Start with the seed configuration
    pts = np.array([(x, y) for x in range(6) for y in range(5)], dtype=np.int64)
    best_score = _score(pts)
    best_pts = pts.copy()
    
    rng = np.random.RandomState(42)
    
    # Local search: move one point at a time
    # Candidate positions: all integer points in [-30,30]^2
    all_positions = np.array([(x, y) for x in range(-30, 31) for y in range(-30, 31)], dtype=np.int64)
    
    # For speed, we'll use a restricted neighborhood first, then expand
    # Neighborhood: points within Chebyshev distance d of current point
    neighborhood_radius = 3
    
    while time.time() < deadline:
        improved = False
        
        # Try moving each point
        for idx in range(30):
            if time.time() > deadline:
                break
            orig = pts[idx].copy()
            
            # Generate candidate positions near current point
            cx, cy = int(orig[0]), int(orig[1])
            candidates = []
            for dx in range(-neighborhood_radius, neighborhood_radius + 1):
                for dy in range(-neighborhood_radius, neighborhood_radius + 1):
                    nx, ny = cx + dx, cy + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        candidates.append((nx, ny))
            
            # Also add some random far positions
            for _ in range(5):
                nx = rng.randint(-30, 30)
                ny = rng.randint(-30, 30)
                candidates.append((nx, ny))
            
            best_candidate_score = best_score
            best_candidate_pos = None
            
            for (nx, ny) in candidates:
                # Skip if same position or occupied by another point
                if nx == cx and ny == cy:
                    continue
                # Check if position is already taken by another point
                occupied = False
                for j in range(30):
                    if j != idx and pts[j, 0] == nx and pts[j, 1] == ny:
                        occupied = True
                        break
                if occupied:
                    continue
                
                pts[idx] = [nx, ny]
                s = _score(pts)
                if s > best_candidate_score:
                    best_candidate_score = s
                    best_candidate_pos = (nx, ny)
            
            # Revert
            pts[idx] = orig
            
            if best_candidate_pos is not None:
                pts[idx] = best_candidate_pos
                best_score = best_candidate_pos and _score(pts)
                best_pts = pts.copy()
                improved = True
        
        if not improved:
            # Try random restarts with small perturbations
            if time.time() > deadline - 5:
                break
            for _ in range(20):
                if time.time() > deadline:
                    break
                trial = best_pts.copy()
                # Move 1-3 random points to random positions
                n_moves = rng.randint(1, 4)
                moved = set()
                for _ in range(n_moves):
                    idx = rng.randint(0, 29)
                    if idx in moved:
                        continue
                    moved.add(idx)
                    nx = rng.randint(-30, 30)
                    ny = rng.randint(-30, 30)
                    # Check not occupied
                    ok = True
                    for j in range(30):
                        if j != idx and trial[j, 0] == nx and trial[j, 1] == ny:
                            ok = False
                            break
                    if ok:
                        trial[idx] = [nx, ny]
                if _valid(trial):
                    s = _score(trial)
                    if s > best_score:
                        best_score = s
                        best_pts = trial.copy()
    
    return [tuple(p) for p in best_pts.tolist()]
# EVOLVE-BLOCK-END
