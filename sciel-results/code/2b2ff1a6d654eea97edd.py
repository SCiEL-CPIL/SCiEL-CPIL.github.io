"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # Time limit check helper
    start_time = time.time()
    TIME_LIMIT = 110.0  # Leave 10s buffer

    # 1. Initialization: Concentric Rings
    # Based on typical packings for N=50:
    # Central circle, then rings.
    # Counts: 1, 6, 12, 18, 13 sums to 50.
    # Alternative: 1, 6, 12, 12, 19? 
    # Let's try a few initializations and pick the best one after a short optimization.
    
    def initialize_ring(counts, radii):
        pts = []
        for r, c in zip(radii, counts):
            if c == 1:
                pts.append([0.0, 0.0])
            else:
                angles = np.linspace(0, 2*np.pi, c, endpoint=False)
                # Offset angles slightly for different rings to avoid perfect alignment issues?
                # Usually symmetric is a good start.
                for a in angles:
                    pts.append([r * np.cos(a), r * np.sin(a)])
        return np.array(pts)

    # Candidate Initializations
    # Init 1: 1, 6, 12, 18, 13
    init1 = initialize_ring([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.70, 0.85])
    
    # Init 2: 1, 6, 12, 12, 19
    init2 = initialize_ring([1, 6, 12, 12, 19], [0.0, 0.25, 0.50, 0.70, 0.85])
    
    # Init 3: 1, 6, 12, 15, 16
    init3 = initialize_ring([1, 6, 12, 15, 16], [0.0, 0.25, 0.50, 0.70, 0.85])

    # Init 4: Hexagonal Lattice (similar to seed but optimized init)
    # We'll just use the ring inits for variety.

    inits = [init1, init2, init3]

    def get_score(pts):
        # Vectorized score calculation
        # 1. Boundary constraint
        norms = np.linalg.norm(pts, axis=1)
        min_bound = 1.0 - np.min(norms)
        
        # 2. Pairwise distance constraint
        # N=50, so 50x50 matrix is small
        # diff[i, j, :] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        # Radius is half min distance, limited by boundary
        r = min(min_dist / 2.0, min_bound)
        return r

    best_pts = None
    best_score = -np.inf

    # Run a short optimization for each init to pick the best starting point
    # Then run a longer optimization on the best one.
    
    def optimize(pts, max_iters=5000, time_limit=None):
        current_score = get_score(pts)
        T = 0.01
        T_min = 0.0001
        cooling = 0.999
        step_size = 0.05
        
        for i in range(max_iters):
            if time_limit and (time.time() - start_time > time_limit):
                break
                
            if i % 1000 == 0:
                T *= cooling
                if T < T_min:
                    break
            
            # Pick random point
            idx = np.random.randint(N)
            
            # Random move
            # Scale step size with T to allow larger moves early?
            # Or keep constant.
            delta = np.random.normal(0, step_size, 2)
            new_pts = pts.copy()
            new_pts[idx] += delta
            
            # Optional: Clamp to circle to prevent huge penalties?
            # The score function handles it, but large moves outside might be unproductive.
            # Let's keep it simple.
            
            new_score = get_score(new_pts)
            
            if new_score > current_score:
                pts = new_pts
                current_score = new_score
            else:
                delta_score = new_score - current_score
                if delta_score < 0:
                    prob = np.exp(delta_score / T)
                    if np.random.rand() < prob:
                        pts = new_pts
                        current_score = new_score
        return pts, current_score

    # Phase 1: Quick screening of inits
    for init in inits:
        if time.time() - start_time > TIME_LIMIT * 0.2:
            break
        pts, score = optimize(init, max_iters=2000)
        if score > best_score:
            best_score = score
            best_pts = pts

    # Phase 2: Deep optimization on the best candidate
    if best_pts is not None:
        # Restart SA with fresh temperature for the best candidate
        best_pts, best_score = optimize(best_pts, max_iters=50000, time_limit=TIME_LIMIT)

    # If time remains, try a few random restarts from the best solution
    # Or just return the best found
    return best_pts

# EVOLVE-BLOCK-END
