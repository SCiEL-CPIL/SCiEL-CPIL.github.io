"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_geometric_heuristic(distance_matrix, coordinates, demands, capacity):
    """
    Compute a heuristic matrix based on smoothed angular sectoring.
    
    1. For each node, find its nearest neighbors.
    2. Calculate the angle of each neighbor relative to the node.
    3. Bin these angles into sectors.
    4. Apply a Savitzky-Golay filter to the sector counts to find the dominant local flow direction.
    5. Construct the heuristic: high attractiveness for edges aligned with the smoothed local flow,
       scaled by inverse distance.
    """
    n = distance_matrix.shape[0]
    # Ensure coordinates are 2D array
    coords = np.asarray(coordinates)
    
    # 1. Inverse distance as base
    # Add small epsilon to avoid division by zero for depot or self-loops
    base_heuristic = 1.0 / (distance_matrix + 1e-6)
    
    # 2. Compute angular sectoring for each node
    # We want to capture the "local direction" of the cluster.
    # For each node i, look at all other nodes j.
    # angle[i, j] = atan2(y_j - y_i, x_j - x_i)
    
    # Vectorized angle calculation
    # coords shape (n, 2)
    # diff[i, j] = coords[j] - coords[i]
    # This is a bit memory heavy for large n, but n=50 is tiny.
    diffs = coords[None, :, :] - coords[:, None, :] 
    # diffs[i, j] is vector from i to j
    angles = np.arctan2(diffs[:, :, 1], diffs[:, :, 0])
    
    # 3. Bin angles into sectors
    num_sectors = 16
    sector_width = 2 * np.pi / num_sectors
    
    # Normalize angles to [0, 2*pi)
    angles_norm = angles % (2 * np.pi)
    
    # Create histogram for each node
    # hist[i, s] = count of neighbors j in sector s relative to i
    hist = np.zeros((n, num_sectors))
    
    # Efficiently bin using digitize or simple loops (n=50 is small)
    for i in range(n):
        if i == 0:
            continue # Depot doesn't need local flow heuristic in the same way, or we can skip
        a = angles_norm[i, :]
        a = a[a > 0] # Ignore self? atan2(0,0) is 0, but dist is 0. 
        # Actually, for i != 0, diffs[i,i] is 0, angle is 0.
        # We should ignore self in the histogram to define local flow from others.
        a = angles_norm[i, np.arange(n) != i]
        
        # Bin
        bins = np.digitize(a, np.linspace(0, 2*np.pi, num_sectors+1)[:-1]) - 1
        # Ensure bins are in range
        bins = np.clip(bins, 0, num_sectors-1)
        hist[i, :] = np.bincount(bins, minlength=num_sectors)
    
    # 4. Smooth the histogram to find dominant direction
    # Use a simple moving average or Savitzky-Golay if available, otherwise manual kernel
    # Kernel width: 3 sectors (75 degrees)
    kernel = np.array([0.25, 0.5, 0.25])
    
    smoothed_hist = np.zeros_like(hist)
    for i in range(n):
        if i == 0:
            continue
        h = hist[i, :]
        # Circular convolution
        smoothed_hist[i, :] = np.convolve(np.concatenate([h[-1:], h, h[:1]]), kernel, mode='same')[1:-1]
        
        # Normalize
        total = smoothed_hist[i, :].sum()
        if total > 0:
            smoothed_hist[i, :] /= total
    
    # 5. Construct final heuristic
    # Attractiveness should be high if the edge (i, j) aligns with the smoothed sector of i.
    # We map the angle of (i, j) to the sector index and use the smoothed probability.
    
    heuristic = np.zeros_like(distance_matrix)
    
    for i in range(n):
        if i == 0:
            # For depot, just use inverse distance, maybe slightly boosted
            heuristic[i, :] = base_heuristic[i, :]
            continue
            
        # Get the smoothed distribution for node i
        dist_i = smoothed_hist[i, :]
        
        for j in range(n):
            if i == j:
                heuristic[i, j] = 0.0
                continue
            
            # Get angle of j relative to i
            angle_ij = angles_norm[i, j]
            # Determine which sector it falls into
            sector_idx = int(angle_ij / sector_width)
            if sector_idx >= num_sectors:
                sector_idx = 0
                
            # The attractiveness is proportional to the probability mass in that sector
            # Multiply by base inverse distance to respect geometry
            # If the sector is "strong", we boost the edge.
            # We use a power to sharpen the preference for aligned edges.
            alignment_score = dist_i[sector_idx]
            
            # Combine: base * (1 + alignment_score * boost_factor)
            # Or simply base * alignment_score^power
            # Let's use: base * (0.5 + alignment_score * 1.5)
            # This ensures all edges have some baseline, but aligned ones are boosted.
            
            heuristic[i, j] = base_heuristic[i, j] * (0.5 + alignment_score * 2.0)
            
    # Ensure symmetry? 
    # The problem defines heuristic[i, j]. In ACO, usually symmetric or directed.
    # The current program uses directed. We will keep directed as computed above.
    
    # Ensure positive and finite
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic

def heuristics(distance_matrix, coordinates, demands, capacity):
    deadline = time.time() + 290  # Safety margin
    
    # 1. Compute the geometric heuristic
    heuristic = _compute_geometric_heuristic(distance_matrix, coordinates, demands, capacity)
    
    # 2. Validation
    if heuristic.shape != distance_matrix.shape:
        # Fallback to standard inverse distance
        heuristic = 1.0 / (distance_matrix + 1e-9)
        
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # 3. Optional: Quick sanity check to ensure no division by zero or NaNs
    if not np.all(np.isfinite(heuristic)):
        heuristic = 1.0 / (distance_matrix + 1e-9)
        
    return heuristic
# EVOLVE-BLOCK-END
