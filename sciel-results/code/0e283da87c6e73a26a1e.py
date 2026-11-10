"""
Physics-based circle packing using Delaunay Triangulation and exact geometric pushes.
Strategy: 
1. Initialize with hexagonal packing.
2. Use Delaunay triangulation to identify critical pairs (nearest neighbors).
3. Iteratively push points apart to resolve distance violations relative to a target radius.
4. Gradually increase the target radius to maximize the minimum distance.
"""
import time
import numpy as np
from scipy.spatial import Delaunay, cKDTree
import warnings

# Suppress warnings from Delaunay if points are degenerate
warnings.filterwarnings("ignore")

N = 200
SEED = 42
TIME_LIMIT = 115.0  # Safety margin before 150s kill

def _compute_radius(centers):
    """Compute the packing radius: min of half pairwise distances and distance to walls."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0

    # Pairwise distances using KDTree for efficiency, or cdist
    # For N=200, cdist is fast enough and simple
    d = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0

    # Distance to walls (square is [-0.5, 0.5]^2)
    dist_to_walls = np.min([
        0.5 - centers[:, 0],
        centers[:, 0] + 0.5,
        0.5 - centers[:, 1],
        centers[:, 1] + 0.5
    ], axis=0)
    min_wall = np.min(dist_to_walls)

    return min(min_pair, min_wall)

def _hexagonal_init():
    """Create a hexagonal packing initialization for N points."""
    best_centers = None
    best_count = 0

    # Try various configurations
    for n_rows in range(13, 22):
        for start_offset in [0, 1]:
            for n_cols in range(12, 20):
                # Calculate how many points fit
                total_points = 0
                for row in range(n_rows):
                    if (row + start_offset) % 2 == 0:
                        total_points += n_cols
                    else:
                        total_points += n_cols - 1

                if total_points < N:
                    continue

                # Calculate spacing to fit in unit square
                # Horizontal spacing s
                # Vertical spacing s * sqrt(3) / 2
                # Total width: (n_cols - 1) * s <= 1.0 - 2*r_margin? 
                # We want to maximize s. 
                # Width constraint: s <= 1.0 / n_cols (roughly, assuming centered)
                # Actually, x_extent = (n_cols - 1) * s. We want x_extent <= 1.0.
                # But we also have wall margins. Let's just fit them tightly first.
                s_horiz = 1.0 / n_cols
                if n_rows > 1:
                    # y_extent = (n_rows - 1) * s * sqrt(3) / 2
                    # We want y_extent <= 1.0
                    s_vert = 2.0 / ((n_rows - 1) * np.sqrt(3))
                else:
                    s_vert = 1.0

                s = min(s_horiz, s_vert)

                # Create the grid
                centers = []
                y_extent = (n_rows - 1) * s * np.sqrt(3) / 2
                # Center vertically
                y_start = -0.5 + (1.0 - y_extent) / 2 + s * np.sqrt(3) / 4

                for row in range(n_rows):
                    y = y_start + row * s * np.sqrt(3) / 2
                    if (row + start_offset) % 2 == 0:
                        x_extent = (n_cols - 1) * s
                        x_start = -0.5 + (1.0 - x_extent) / 2
                        for col in range(n_cols):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centers.append((x, y))
                    else:
                        x_extent = (n_cols - 2) * s
                        x_start = -0.5 + (1.0 - x_extent) / 2 + s / 2
                        for col in range(n_cols - 1):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centers.append((x, y))

                centers = np.array(centers)
                if len(centers) >= N and len(centers) > best_count:
                    best_count = len(centers)
                    best_centers = centers[:N].copy()

    if best_centers is not None:
        return best_centers
    
    # Fallback to grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _delaunay_push_step(centers, r_target, alpha):
    """
    Perform one step of the Delaunay-based repulsion.
    Returns updated centers and the minimum distance found.
    """
    centers = centers.copy()
    
    # Compute Delaunay triangulation
    try:
        tri = Delaunay(centers)
        # Get unique edges from triangles
        # tri.simplices is (M, 3)
        # Edges are (0,1), (1,2), (2,0) for each triangle
        edges = np.vstack([
            tri.simplices[:, [0, 1]],
            tri.simplices[:, [1, 2]],
            tri.simplices[:, [2, 0]]
        ])
        # Remove duplicate edges
        # Sort each edge so min index is first, then unique
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)
    except Exception:
        # If Delaunay fails (e.g. coincident points), use all pairs
        # This is expensive but safe
        edges = np.array([(i, j) for i in range(N) for j in range(i+1, N)])

    forces = np.zeros_like(centers)
    
    # Process each edge
    for i, j in edges:
        d_vec = centers[i] - centers[j]
        dist = np.linalg.norm(d_vec)
        
        if dist < 2.0 * r_target:
            # Violation
            if dist < 1e-12:
                # Random perturbation if coincident
                d_vec = np.random.randn(2)
                dist = np.linalg.norm(d_vec)
            
            # Unit vector from j to i
            u = d_vec / dist
            
            # Repulsive force on i (push i away from j)
            # Magnitude proportional to violation
            violation = 2.0 * r_target - dist
            f = u * violation
            
            forces[i] += f
            forces[j] -= f

    # Handle wall constraints
    # Push points away from walls if they are within r_target of the wall
    for axis in range(2):
        # Left wall: x < -0.5
        # Distance to left wall: centers[:, axis] + 0.5
        # If dist < r_target, push right (positive axis direction)
        dist_left = centers[:, axis] + 0.5
        viol_left = np.maximum(0, r_target - dist_left)
        forces[:, axis] += viol_left
        
        # Right wall: x > 0.5
        # Distance to right wall: 0.5 - centers[:, axis]
        # If dist < r_target, push left (negative axis direction)
        dist_right = 0.5 - centers[:, axis]
        viol_right = np.maximum(0, r_target - dist_right)
        forces[:, axis] -= viol_right
        
        # Bottom wall: y < -0.5
        dist_bottom = centers[:, axis] + 0.5
        viol_bottom = np.maximum(0, r_target - dist_bottom)
        forces[:, axis] += viol_bottom
        
        # Top wall: y > 0.5
        dist_top = 0.5 - centers[:, axis]
        viol_top = np.maximum(0, r_target - dist_top)
        forces[:, axis] -= viol_top

    # Update positions
    centers += alpha * forces
    
    # Clamp to square boundaries
    centers = np.clip(centers, -0.5, 0.5)
    
    # Compute current min distance
    # Re-calc distances for all pairs to get true min (Delaunay edges might miss some in transient states? 
    # Actually, min distance pair is always a Delaunay edge, so we could just check those, 
    # but cdist is fast enough for N=200)
    d = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    min_dist = np.min(d) / 2.0
    
    # Check walls
    dist_to_walls = np.min([
        0.5 - centers[:, 0],
        centers[:, 0] + 0.5,
        0.5 - centers[:, 1],
        centers[:, 1] + 0.5
    ], axis=0)
    min_wall = np.min(dist_to_walls)
    
    return centers, min(min_dist, min_wall)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    # Initialize
    centers = _hexagonal_init()
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Optimization Loop
    # We will try to increase the target radius gradually.
    # Start with the current best radius as a baseline.
    
    # Initial target slightly above current best
    r_target = best_radius * 1.001
    if r_target < 0.035:
        r_target = 0.035
        
    # Parameters for the push solver
    alpha = 0.5  # Step size
    max_iter = 2000
    no_improve_count = 0
    improve_threshold = 1e-9
    
    # We will run multiple "phases" with increasing r_target
    # Phase 1: Stabilize at current r
    # Phase 2: Increase r slightly
    # Phase 3: Increase r more
    
    current_r = best_radius
    
    # Main optimization loop
    while time.time() < deadline - 5.0:
        
        # Check if we can increase the target radius
        # If the current configuration satisfies r_target, we can try to go higher
        # We check if the actual min distance is significantly larger than r_target
        actual_r = _compute_radius(centers)
        
        if actual_r > r_target + 1e-6:
            # We have headroom. Increase target.
            # How much? Start small, increase if stable.
            # Heuristic: increase by 1% of current radius
            r_target = min(r_target * 1.005, actual_r)
            no_improve_count = 0
            # Reset alpha to be more aggressive when expanding
            alpha = 0.8
        else:
            # We are at the limit. Try to refine.
            # If we haven't improved the global best in a while, we might be stuck.
            # However, the push method is deterministic given the state.
            # To escape local minima, we could add small noise, but let's stick to pure push first.
            # If actual_r is very close to r_target, we are in equilibrium.
            
            if actual_r > best_radius:
                best_radius = actual_r
                best_centers = centers.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # If stuck for too long, try a small random perturbation to escape local minimum
            if no_improve_count > 50:
                # Perturb
                noise = np.random.randn(N, 2) * 0.001
                centers += noise
                centers = np.clip(centers, -0.5, 0.5)
                no_improve_count = 0
                # Reduce alpha to be careful
                alpha = 0.2
            else:
                # Decay alpha
                alpha = max(0.1, alpha * 0.99)

        # Perform push steps
        for step in range(50): # Do a batch of 50 steps
            if time.time() > deadline - 5.0:
                break
            
            centers, min_r = _delaunay_push_step(centers, r_target, alpha)
            
            # Check for improvement
            if min_r > best_radius:
                best_radius = min_r
                best_centers = centers.copy()
        
        # Update current_r for next loop iteration logic
        current_r = _compute_radius(centers)
        
        # If we have reached a very high radius, reduce step size for precision
        if best_radius > 0.036:
            alpha = min(alpha, 0.1)
            
    # Final high-precision polish
    if time.time() < deadline - 2.0:
        # Use a very tight target and small steps
        r_target = best_radius * 1.0001
        alpha = 0.05
        for _ in range(1000):
            if time.time() > deadline - 1.0:
                break
            centers, min_r = _delaunay_push_step(centers, r_target, alpha)
            if min_r > best_radius:
                best_radius = min_r
                best_centers = centers.copy()

    return best_centers

if __name__ == "__main__":
    centers = construct()
    r = _compute_radius(centers)
    print(f"Packing radius: {r:.12f}")
    print(f"Best known:     0.036612798904")
    print(f"Ratio:          {r / 0.036612798904:.6f}")
