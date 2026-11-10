import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 50
CUBE_HALF = 0.5

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Function to calculate the score (radius)
    def calc_radius(centers):
        # Min distance between centers
        if len(centers) < 2:
            min_dist = 1.0
        else:
            # Use cdist for efficiency
            from scipy.spatial.distance import cdist
            dists = cdist(centers, centers)
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
        
        # Min distance to boundary
        # Distance to each face is 0.5 - |coord|
        dists_to_boundary = 0.5 - np.abs(centers)
        min_boundary = np.min(dists_to_boundary)
        
        return min(min_dist, 2 * min_boundary) / 2.0 # Wait, score is r.
        # Score definition: r = min( min_{i<j} |ci-cj|/2, min_i min_k (0.5 - |cik|) )
        # So r = 0.5 * min( min_pair_dist, min_boundary_dist * 2 )? No.
        # r = min( min_pair_dist / 2, min_boundary_dist )
        
    def get_radius(centers):
        if len(centers) < 2:
            min_pair_dist = 1.0
        else:
            from scipy.spatial.distance import cdist
            dists = cdist(centers, centers)
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
        
        dists_to_boundary = 0.5 - np.abs(centers)
        min_boundary_dist = np.min(dists_to_boundary)
        
        return min(min_pair_dist / 2.0, min_boundary_dist)

    # Initialize with a jittered grid
    def init_centers(seed):
        np.random.seed(seed)
        m = int(np.ceil(N ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m
        # Create grid
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        centers = grid[:N].copy()
        # Shift to [-0.5, 0.5]
        centers -= 0.5
        # Jitter
        centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
        # Clip to ensure valid start
        centers = np.clip(centers, -0.49, 0.49)
        return centers

    best_centers = None
    best_score = -1.0
    
    # Try a few restarts if time permits, or just one long run
    # Given 110s, we can do maybe 2-3 restarts with different seeds
    # Or one restart with many iterations
    
    # Let's do a single robust run with a good seed, and if time remains, try another
    seeds = [42, 123, 999]
    
    for seed in seeds:
        if time.time() > deadline - 10.0:
            break
            
        centers = init_centers(seed)
        current_score = get_radius(centers)
        
        # Lloyd's Algorithm Iteration
        # We will iterate until convergence or time limit
        # To avoid getting stuck in very bad local minima, we might need to
        # handle the case where a Voronoi cell becomes empty or degenerate.
        
        # Pre-compute cube constraints for LP
        # Variables: x, y, z, r
        # Maximize r -> Minimize -r
        # Constraints:
        # 1. Voronoi faces: (ci - cj) . (x - (ci+cj)/2) >= 0
        #    (ci - cj) . x - (ci - cj) . (ci+cj)/2 >= 0
        #    (ci - cj) . x - |ci-cj|^2 / 2 >= 0
        #    -(ci - cj) . x + r <= -|ci-cj|^2 / 2  ? No.
        #    We want point x to be closer to ci than cj.
        #    |x - ci|^2 <= |x - cj|^2
        #    |x|^2 - 2x.ci + |ci|^2 <= |x|^2 - 2x.cj + |cj|^2
        #    2x.(cj - ci) <= |cj|^2 - |ci|^2
        #    x.(cj - ci) <= (|cj|^2 - |ci|^2) / 2
        #    This is a linear constraint on x.
        #    But we also need the sphere to be inside the cell.
        #    The distance from x to the face defined by this constraint must be >= r.
        #    The face is defined by n . x = d, where n = cj - ci (normalized? no, raw).
        #    Distance = (d - n . x) / ||n||.
        #    We require (d - n . x) / ||n|| >= r
        #    d - n . x - r * ||n|| >= 0
        #    n . x + r * ||n|| <= d
        
        # 2. Cube boundaries:
        #    x <= 0.5, -x <= 0.5, etc.
        #    Distance to boundary x=0.5 is 0.5 - x.
        #    0.5 - x >= r  => x + r <= 0.5
        #    Distance to boundary x=-0.5 is x + 0.5.
        #    x + 0.5 >= r  => -x + r <= 0.5
        
        # Let's build the LP for each center
        
        max_iterations = 500
        converged = False
        
        for it in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
                
            # Compute Voronoi
            vor = scipy.spatial.Voronoi(centers)
            
            new_centers = np.zeros_like(centers)
            
            for i in range(N):
                # Get neighbors for site i
                # vor.regions gives indices of vertices for each region.
                # We need the constraints (faces).
                # It is easier to iterate over all other sites j and add the constraint.
                # This is O(N^2) per iteration, which is 2500 constraints total for all LPs?
                # No, for each i, we have N-1 constraints.
                # 50 LPs, each with 49 Voronoi constraints + 6 Cube constraints = 55 constraints.
                # Variables: 4 (x,y,z,r).
                # This is very fast.
                
                A = []
                b = []
                
                # 1. Voronoi constraints
                for j in range(N):
                    if i == j:
                        continue
                    ci = centers[i]
                    cj = centers[j]
                    # Constraint: |x - ci|^2 <= |x - cj|^2
                    # 2 x . (cj - ci) <= |cj|^2 - |ci|^2
                    # n = cj - ci
                    # d = (|cj|^2 - |ci|^2) / 2
                    n = cj - ci
                    norm_n = np.linalg.norm(n)
                    if norm_n < 1e-10:
                        continue
                    d = (np.dot(cj, cj) - np.dot(ci, ci)) / 2.0
                    # Distance from x to plane n . x = d is (d - n . x) / norm_n
                    # We require distance >= r
                    # (d - n . x) / norm_n >= r
                    # d - n . x - r * norm_n >= 0
                    # n . x + r * norm_n <= d
                    A.append(np.append(n, norm_n))
                    b.append(d)
                    
                # 2. Cube constraints
                # x + r <= 0.5
                A.append(np.array([1.0, 0.0, 0.0, 1.0]))
                b.append(0.5)
                # -x + r <= 0.5
                A.append(np.array([-1.0, 0.0, 0.0, 1.0]))
                b.append(0.5)
                # y + r <= 0.5
                A.append(np.array([0.0, 1.0, 0.0, 1.0]))
                b.append(0.5)
                # -y + r <= 0.5
                A.append(np.array([0.0, -1.0, 0.0, 1.0]))
                b.append(0.5)
                # z + r <= 0.5
                A.append(np.array([0.0, 0.0, 1.0, 1.0]))
                b.append(0.5)
                # -z + r <= 0.5
                A.append(np.array([0.0, 0.0, -1.0, 1.0]))
                b.append(0.5)
                
                A = np.array(A)
                b = np.array(b)
                
                # Objective: Minimize -r
                # c = [0, 0, 0, -1]
                c = np.array([0.0, 0.0, 0.0, -1.0])
                
                # Bounds:
                # x, y, z in [-0.5, 0.5]
                # r >= 0
                bounds = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (0, None)]
                
                try:
                    res = scipy.optimize.linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
                    if res.success:
                        new_centers[i] = res.x[:3]
                    else:
                        # Fallback to old center if LP fails
                        new_centers[i] = centers[i]
                except Exception:
                    new_centers[i] = centers[i]
            
            # Update centers
            # Check if moved significantly
            if np.linalg.norm(new_centers - centers) < 1e-6:
                converged = True
                break
            centers = new_centers
            
            # Track best
            score = get_radius(centers)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                
        # End of Lloyd iterations for this seed
        if time.time() > deadline - 5.0:
            break

    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]

    return best_centers
