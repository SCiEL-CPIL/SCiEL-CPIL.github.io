"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
import time

def construct():
    """
    Construct centers of 100 equal spheres in a unit cube [-0.5, 0.5]^3
    to maximize the minimum distance between any two centers and the
    distance to the cube boundary.
    
    Uses a Simulated Annealing approach with:
    1. FCC Lattice Initialization for high-quality starting point.
    2. Spatial indexing (KD-Tree) for efficient nearest neighbor updates.
    3. Geometric moves (random displacement) to explore configuration space.
    """
    np.random.seed(42)
    
    # 1. Initialization: FCC Lattice
    # Generate an FCC lattice that fills the cube
    # FCC unit cell has 4 atoms. We need ~100 atoms.
    # Let's generate a 5x5x5 supercell of FCC or similar and select best 100.
    
    def generate_fcc_lattice(n_points=100, margin=0.0):
        """Generate points on an FCC lattice inside [-0.5, 0.5]^3."""
        points = []
        # FCC basis vectors
        # a is the lattice constant. 
        # We want to fit points in [-0.5, 0.5].
        # Let's try a grid of 4x4x4 unit cells (each has 4 points) = 256 points.
        # Then we can select the best 100 or just use a denser grid and optimize.
        
        # Actually, a simpler robust init:
        # Use a cubic lattice with spacing s, then optimize.
        # But FCC is denser. 
        # Let's create a set of candidate points from an FCC grid with slightly smaller spacing
        # and then pick 100.
        
        # Let's define a grid of side L=1.0.
        # FCC points in a unit cell of side a:
        # (0,0,0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)
        
        # We want to cover [-0.5, 0.5].
        # Let's use a = 0.25. 
        # Grid indices i, j, k from -2 to 2 (5x5x5 cells = 125 cells * 4 pts = 500 pts).
        a = 0.25
        pts = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    base = np.array([i*a, j*a, k*a])
                    pts.append(base)
                    pts.append(base + np.array([0, a/2, a/2]))
                    pts.append(base + np.array([a/2, 0, a/2]))
                    pts.append(base + np.array([a/2, a/2, 0]))
        
        pts = np.array(pts)
        
        # Filter to inside cube
        mask = np.all(np.abs(pts) <= 0.5, axis=1)
        pts = pts[mask]
        
        if len(pts) < n_points:
             # Fallback to random if lattice too sparse (should not happen)
             return np.random.uniform(-0.45, 0.45, size=(n_points, 3))
        
        # We have more than n_points. 
        # We want to select n_points that are well distributed.
        # A simple heuristic: pick points that maximize min distance greedily?
        # Or just take a subset. 
        # Better: Use a random subset of the lattice points to break symmetry,
        # then let SA fix the rest.
        
        # To ensure we don't start with a rigid lattice that traps us, 
        # let's take a random subset of 100 from the available lattice points.
        # This preserves the local density benefits of FCC but breaks global symmetry.
        indices = np.random.choice(len(pts), n_points, replace=False)
        return pts[indices]

    centers = generate_fcc_lattice(N)
    
    # 2. Simulation Setup
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # SA Parameters
    T = 0.05 # Initial temperature
    T_min = 1e-8
    cooling_rate = 0.999
    max_steps = 20000
    
    # Helper to compute score
    def compute_score(c):
        # Boundary check
        if np.any(np.abs(c) > 0.5):
            return -1e9
        # Pairwise distances
        # Using cKDTree for efficiency
        tree = cKDTree(c)
        # Query 2 nearest neighbors for each point to find min dist
        # Note: query(k=2) returns self and nearest.
        dists, _ = tree.query(c, k=2)
        min_pair_dist = np.min(dists[:, 1])
        
        # Boundary distance
        boundary_dists = 0.5 - np.abs(c)
        min_boundary = np.min(boundary_dists)
        
        return min(min_pair_dist / 2.0, min_boundary)
    
    current_score = compute_score(centers)
    best_score = current_score
    best_centers = centers.copy()
    
    # Pre-compute some constants
    step_size = 0.05
    
    for i in range(max_steps):
        if time.time() - start_time > time_limit:
            break
            
        # Generate a move: pick a random point and displace it
        idx = np.random.randint(0, N)
        old_point = centers[idx].copy()
        
        # Random displacement
        displacement = np.random.normal(0, step_size, 3)
        new_point = old_point + displacement
        
        # Check bounds
        if np.any(np.abs(new_point) > 0.5):
            # Reflect or reject? Rejecting is simpler for SA logic
            # But reflecting might help exploration. Let's reject for now.
            continue
            
        centers[idx] = new_point
        
        # Compute new score
        # Optimization: We only need to update distances involving idx.
        # However, recomputing the full score via KDTree is O(N log N).
        # For N=100, this is fast enough.
        new_score = compute_score(centers)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0:
            # Accept
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            # Metropolis criterion
            if T > T_min:
                # Probability of accepting worse move
                # Use exp(delta / T)
                # delta is negative
                p = np.exp(delta / T)
                if np.random.rand() < p:
                    current_score = new_score
            else:
                # Reject, revert
                centers[idx] = old_point
        
        # Cool down
        T *= cooling_rate
        
        # Adaptive step size: if we accept too many, increase step; if too few, decrease
        # Simple heuristic: keep step size constant for stability, 
        # or slightly decrease it over time to converge.
        step_size *= 0.9999
        
    # Final polish: Local optimization on the best found
    # Use a few iterations of Nelder-Mead or simple gradient-free refinement?
    # Since we are using a new approach, let's stick to SA but do a final "anneal" 
    # with very low T and smaller steps to polish.
    
    T = 1e-5
    step_size = 0.005
    for i in range(2000):
        if time.time() - start_time > time_limit:
            break
        idx = np.random.randint(0, N)
        old_point = centers[idx].copy()
        displacement = np.random.normal(0, step_size, 3)
        new_point = old_point + displacement
        
        if np.any(np.abs(new_point) > 0.5):
            continue
            
        centers[idx] = new_point
        new_score = compute_score(centers)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            centers[idx] = old_point
            T *= 0.99

    # Ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

# EVOLVE-BLOCK-END
