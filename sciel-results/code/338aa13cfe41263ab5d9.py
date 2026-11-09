import numpy as np
import time

def compute_max_multiplicity(pts):
    n = len(pts)
    # Generate indices for combinations of 3
    # Using itertools is slow in Python loop, but np.meshgrid is fast
    # For N=30, 4060 triples.
    # We can precompute indices once.
    pass

# Precompute indices for N=30
_N = 30
_i, _j, _k = np.meshgrid(np.arange(_N), np.arange(_N), np.arange(_N), indexing='ij')
_mask = (_i < _j) & (_j < _k)
_I, _J, _K = _i[_mask], _j[_mask], _k[_mask]

def get_score(pts):
    v1 = pts[_J] - pts[_I]
    v2 = pts[_K] - pts[_I]
    dets = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    abs_dets = np.abs(dets)
    non_zero = abs_dets[abs_dets > 0]
    if len(non_zero) == 0:
        return 0
    vals, counts = np.unique(non_zero, return_counts=True)
    return int(np.max(counts))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    # Initialize
    # Try a few random starts and pick the best
    best_pts = None
    best_score = -1
    
    for _ in range(10):
        # Random points in [-30, 30]
        pts = np.random.randint(-30, 31, size=(30, 2))
        # Ensure distinct
        # If duplicates, perturb
        # For now, assume random is distinct enough or check
        unique_pts = np.unique(pts, axis=0)
        if len(unique_pts) < 30:
             # Fill with random
             while len(unique_pts) < 30:
                 p = np.random.randint(-30, 31, size=2)
                 if not any(np.array_equal(p, u) for u in unique_pts):
                     unique_pts = np.vstack([unique_pts, p])
        pts = unique_pts[:30]
        
        score = get_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
            
    # Local Search
    current_pts = best_pts.copy()
    current_score = best_score
    T = 5.0 # Temperature
    cooling = 0.999
    
    while time.time() < deadline:
        # Propose move: Swap two points
        i, j = np.random.choice(30, 2, replace=False)
        new_pts = current_pts.copy()
        new_pts[i], new_pts[j] = new_pts[j], new_pts[i]
        
        # Check distinctness? Swapping preserves distinctness.
        
        new_score = get_score(new_pts)
        
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            # Accept with probability
            if T > 0.1:
                delta = new_score - current_score
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    current_pts = new_pts
                    current_score = new_score
        
        T *= cooling
        if T < 0.1:
            # Restart with random perturb of best
            current_pts = best_pts.copy()
            current_score = best_score
            T = 2.0
            
    # Convert back to list of tuples
    return [tuple(p) for p in best_pts]
