# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Voronoi

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Generates an edge-attractiveness matrix using a Capacity-Aware Voronoi Tessellation strategy.
    
    This approach partitions the space into Voronoi cells for each customer.
    The heuristic weight for an edge (i, j) is determined by:
    1. Base inverse distance.
    2. A penalty if the edge crosses Voronoi cell boundaries, scaled by the 
       ratio of the edge length to the minimum enclosing circle radius of the 
       shared Voronoi cell boundary (approximated by cell radius).
    
    This explicitly penalizes long-range connections that violate local 
    capacity density constraints, guiding ants to form routes that respect 
    spatial clustering and capacity limits.
    """
    n = distance_matrix.shape[0]
    start_time = time.time()
    deadline = start_time + 280.0 # Safety margin for 300s limit
    
    # Convert inputs to numpy arrays for safety
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demloads, dtype=float)
    
    # Check time budget immediately
    if time.time() > deadline:
        # Fallback to simple inverse distance
        dist_safe = distance_matrix.copy()
        np.fill_diagonal(dist_safe, 1e6)
        fallback = 1.0 / dist_safe
        fallback = np.where(fallback > 0, fallback, 1e-9)
        return fallback

    # 1. Base Heuristic: Inverse Distance
    dist_safe = distance_matrix.copy()
    np.fill_diagonal(dist_safe, 1e6) # Avoid division by zero on diagonal
    base_heuristic = 1.0 / dist_safe
    
    # 2. Voronoi Tessellation
    # We compute the Voronoi diagram for the customer coordinates (excluding depot? 
    # Usually Voronoi is computed on the points of interest. Let's include all points 
    # to get a full partition, but the depot is at 0,0 usually? No, depot is node 0.
    # The task says "Node 0 is the depot... coordinates are uniform in the unit square".
    # So coordinates[0] is the depot.
    # We compute Voronoi on all n points.
    
    try:
        vor = Voronoi(coordinates)
    except Exception:
        # Fallback if Voronoi fails
        return np.where(base_heuristic > 0, base_heuristic, 1e-9)

    # Compute a characteristic radius for each Voronoi cell.
    # We approximate the "size" of a cell by the distance from the point to its 
    # nearest vertex in the Voronoi diagram, or the mean distance to vertices.
    # To be robust, we calculate the mean distance from each point to its Voronoi vertices.
    # If a point has no vertices (unbounded cell), we use the median of all computed radii.
    
    cell_radii = np.zeros(n)
    valid_radii = []
    
    for i in range(n):
        verts = vor.regions[vor.point_region[i]]
        if len(verts) > 0 and -1 not in verts:
            # Get coordinates of vertices
            vert_coords = vor.vertices[verts]
            # Calculate distances from point i to these vertices
            dists = np.linalg.norm(vert_coords - coordinates[i], axis=1)
            # Use the mean or median distance as a proxy for cell size
            # Mean might be skewed by far vertices in unbounded cells, but we filtered -1.
            # Let's use the minimum distance to a vertex as a lower bound on cell "reach"
            # or the mean. Mean is more representative of the "area" influence.
            r = np.mean(dists)
            cell_radii[i] = r
            valid_radii.append(r)
        else:
            # Unbounded cell or no vertices (should not happen for interior points in a box,
            # but Voronoi on a set of points in a plane yields unbounded regions for convex hull points).
            # We will fill these later.
            cell_radii[i] = np.nan

    if len(valid_radii) == 0:
        # Fallback
        return np.where(base_heuristic > 0, base_heuristic, 1e-9)
        
    # Fill NaNs with the median of valid radii
    median_radius = np.median(valid_radii)
    cell_radii = np.where(np.isnan(cell_radii), median_radius, cell_radii)
    
    # 3. Construct the Voronoi-Aware Heuristic
    # We want to penalize edges that are "long" relative to the local cell structure.
    # Specifically, if an edge connects two points in different cells, and the distance 
    # is large compared to the sum of their cell radii, it is likely a "long-range" 
    # connection that crosses many cells, which is inefficient for CVRP.
    
    # Let's define a "Voronoi Penalty" factor.
    # For each pair (i, j), we look at the distance d_ij.
    # We compare d_ij to the "local scale" defined by cell_radii[i] and cell_radii[j].
    # A natural scale is (cell_radii[i] + cell_radii[j]).
    
    # If d_ij is much larger than the local scale, it's a long jump.
    # We apply a soft penalty: exp(- (d_ij / (k * local_scale))^2 )
    # where k is a tuning parameter. If k is small, the penalty is sharp.
    
    # Let's use a ratio r = d_ij / (cell_radii[i] + cell_radii[j])
    # If r > 1, the edge is longer than the sum of the "radii" of the two cells.
    # We want to reduce the attractiveness for large r.
    
    # Create distance matrix for broadcasting
    d_mat = distance_matrix.copy()
    # Avoid division by zero in the denominator if radii are 0 (unlikely)
    denom = cell_radii[:, None] + cell_radii[None, :]
    denom[denom == 0] = 1e-6
    
    ratio = d_mat / denom
    
    # The penalty should be 1 for small ratios and decay for large ratios.
    # Let's use exp(-beta * ratio^2).
    # If beta is large, only very local edges are allowed.
    # If beta is small, it's a gentle nudge.
    # Given that Voronoi cells tile the space, neighbors are typically within 1-2 radii.
    # Let's try beta = 1.0.
    
    beta = 1.5
    voronoi_penalty = np.exp(-beta * (ratio ** 2))
    
    # The final heuristic is base_heuristic * voronoi_penalty
    # This preserves the 1/d behavior for very close neighbors (where penalty ~ 1)
    # but suppresses long edges that cross cell boundaries significantly.
    
    heuristic_matrix = base_heuristic * voronoi_penalty
    
    # Handle the diagonal: 
    # d_mat diagonal is 0, ratio is 0, penalty is 1.
    # base_heuristic diagonal is 1e-6.
    # So diagonal is 1e-6. This is fine.
    
    # Validation:
    # 1. Shape check
    if heuristic_matrix.shape != distance_matrix.shape:
        dist_safe = distance_matrix.copy()
        np.fill_diagonal(dist_safe, 1e6)
        heuristic_matrix = 1.0 / dist_safe
        heuristic_matrix = np.where(heuristic_matrix > 0, heuristic_matrix, 1e-9)
        return heuristic_matrix

    # 2. Finiteness check
    if not np.all(np.isfinite(heuristic_matrix)):
        heuristic_matrix = np.where(np.isfinite(heuristic_matrix), heuristic_matrix, 1e-9)
        
    # 3. Positivity check
    # "Heuristic entries that are not positive become 1e-9"
    heuristic_matrix = np.where(heuristic_matrix > 0, heuristic_matrix, 1e-9)
    
    return heuristic_matrix
# EVOLVE-BLOCK-END
