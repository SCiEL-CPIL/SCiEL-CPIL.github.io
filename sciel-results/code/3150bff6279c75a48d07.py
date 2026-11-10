import numpy as np
import time

N = 300

def compute_score(positions):
    """
    Computes the score: min(min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|))
    """
    if len(positions) == 0:
        return 0.0
    
    # Boundary constraint
    norms = np.linalg.norm(positions, axis=1)
    min_boundary = 1.0 - np.min(norms)
    
    # Pairwise distance constraint
    # Use broadcasting for distances
    # dist[i, j] = |c_i - c_j|
    # To save memory, we can compute in chunks or just use the full matrix for N=300
    # N=300 -> 90000 distances. Very small.
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dist, np.inf)
    
    min_pair = np.min(dist) / 2.0
    
    return min(min_boundary, min_pair)

def optimize_positions(initial_positions, max_iters=10000, initial_eta=0.005):
    """
    Subgradient ascent to maximize the minimum distance.
    """
    C = initial_positions.copy()
    best_score = compute_score(C)
    best_C = C.copy()
    
    eta = initial_eta
    # Active set tolerance
    tol = 1e-5
    
    for i in range(max_iters):
        # Compute current score and active constraints
        norms = np.linalg.norm(C, axis=1)
        boundary_dists = 1.0 - norms
        
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        min_pair_dist = np.min(dist) / 2.0
        current_score = min(np.min(boundary_dists), min_pair_dist)
        
        if current_score > best_score:
            best_score = current_score
            best_C = C.copy()
            
        # Identify active constraints
        # Active pairs: dist/2 < current_score + tol
        active_pair_mask = (dist / 2.0) < (current_score + tol)
        # Exclude diagonal
        np.fill_diagonal(active_pair_mask, False)
        
        # Active boundaries: 1 - norm < current_score + tol
        active_boundary_mask = boundary_dists < (current_score + tol)
        
        # Compute gradient
        grad = np.zeros_like(C)
        
        # Pair gradients
        # For each active pair (i, j), add (C_i - C_j)/dist to grad[i] and (C_j - C_i)/dist to grad[j]
        # We can vectorize this partially, but for N=300, a loop over active pairs might be fine if not too many.
        # Or use matrix operations.
        # grad[i] = sum_{j active with i} (C_i - C_j) / dist_ij
        
        # Let's use a vectorized approach for pairs
        # We need to sum (C_i - C_j) / dist_ij for all j where active_pair_mask[i,j] is True.
        
        # Create a weight matrix W where W[i,j] = 1/dist_ij if active, else 0
        W = np.zeros_like(dist)
        # Avoid division by zero
        W[active_pair_mask] = 1.0 / dist[active_pair_mask]
        
        # Sum of (C_i - C_j) * W[i,j] over j
        # C_i * sum(W[i,j]) - sum(C_j * W[i,j])
        
        # sum_W[i] = sum_j W[i,j]
        sum_W = np.sum(W, axis=1)
        
        # sum_CW[i] = sum_j C_j * W[i,j]
        # This is W.T @ C
        sum_CW = W.T @ C
        
        pair_grad = C * sum_W[:, np.newaxis] - sum_CW
        
        # Boundary gradients
        # For active boundary i, grad[i] += -C_i / norm_i
        boundary_grad = np.zeros_like(C)
        active_indices = np.where(active_boundary_mask)[0]
        if len(active_indices) > 0:
            boundary_grad[active_indices] = -C[active_indices] / norms[active_indices, np.newaxis]
        
        total_grad = pair_grad + boundary_grad
        
        # Update
        C = C + eta * total_grad
        
        # Project back to unit disk if any point goes outside
        new_norms = np.linalg.norm(C, axis=1)
        out_mask = new_norms > 1.0
        if np.any(out_mask):
            C[out_mask] /= new_norms[out_mask, np.newaxis]
        
        # Decay eta
        eta *= 0.999
        if eta < 1e-7:
            break
            
    return best_C, best_score

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    best_positions = None
    best_score = 0.0
    
    # Try a few spacing values to get close to N points
    # Theoretical optimal r ~ 0.053. Spacing s = 2r ~ 0.106.
    # Let's scan spacing from 0.09 to 0.12
    
    spacings = np.linspace(0.09, 0.12, 10)
    
    for s in spacings:
        if time.time() > deadline:
            break
            
        # Generate hex lattice
        # y = j * s * sqrt(3)/2
        # x = (i + 0.5*(j%2)) * s
        k = int(1.0 / s) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    rows.append((x, y))
        
        points = np.array(rows)
        
        if len(points) < N:
            continue
            
        # If we have more than N points, select the best N
        # Heuristic: Select points that maximize the minimum distance.
        # For simplicity, if len(points) is close to N, we can just take the first N or the ones closest to center?
        # Actually, for a lattice, the "extra" points are on the boundary.
        # We want to keep the interior points.
        # Sort by radius and take the innermost N?
        # This might create a hole in the middle if not careful, but for a lattice cut, the center is dense.
        # Let's sort by radius and take the first N.
        radii = np.linalg.norm(points, axis=1)
        sorted_indices = np.argsort(radii)
        points = points[sorted_indices[:N]]
        
        # Optimize
        opt_positions, score = optimize_positions(points, max_iters=2000, initial_eta=0.005)
        
        if score > best_score:
            best_score = score
            best_positions = opt_positions
            
    if best_positions is None:
        # Fallback: Random
        best_positions = np.random.rand(N, 2) * 2 - 1
        # Ensure inside
        norms = np.linalg.norm(best_positions, axis=1)
        best_positions /= norms[:, np.newaxis]
        best_positions *= 0.9 # Pull in
        
    return best_positions
