"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _hilbert_3d_index_to_coords(n, dim=3):
    """
    Map a 1D index in [0, n^dim) to 3D coordinates in [0, 1)^dim using a
    simplified space-filling curve approach. We use a recursive subdivision
    (Hilbert-like) to generate well-distributed points.
    """
    # For simplicity and speed, we'll use a recursive Hilbert curve for 3D
    # This is a standard algorithm adapted from Wikipedia/online resources
    
    def hilbert3d(n, i):
        """
        Convert integer i in [0, n^3) to 3D coordinates in [0, n)^3
        using a Hilbert-like space filling curve.
        """
        # We use a recursive approach based on the Hilbert curve properties
        # For 3D, we subdivide the cube into 8 sub-cubes
        
        if n == 1:
            return np.array([0.0, 0.0, 0.0])
        
        sub_n = n // 2
        # There are 8 sub-cubes in 3D
        # The order of visiting sub-cubes follows a Hilbert-like pattern
        # For 3D Hilbert curve, the 8 sub-cubes are visited in a specific order
        
        # Define the 8 sub-cube origins in [0, 1)^3 (scaled by sub_n)
        # The Hilbert curve in 3D visits sub-cubes in this order:
        # (0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1)
        # This is one possible ordering; we'll use a standard one
        
        # Actually, let's use a simpler approach: generate points via recursive
        # subdivision with a fixed visiting order that ensures good space-filling
        
        # For robustness, let's use a different strategy:
        # Use a 3D grid with Hilbert-like ordering via bit manipulation
        
        # Convert i to 3D index in [0, n)^3 using a space-filling mapping
        # We'll use a recursive Hilbert curve implementation
        
        x, y, z = 0, 0, 0
        t = i
        for p in range(int(np.log2(n)) if n > 1 else 0, -1, -1):
            rx = (t >> 2) & 1
            ry = (t >> 1) & 1
            rz = t & 1
            
            # Rotate/reflection step for 3D Hilbert
            # This is simplified; a proper 3D Hilbert has more complex rotations
            # For our purposes, a reasonable space-filling order is sufficient
            
            # Simple mapping: use the bits to determine position in sub-cube
            # We'll use a lexicographic-like order with some scrambling
            sub_idx = (rx << 1) | ry  # This gives 0-3 for x,y
            # Actually, let's just use a simple grid-based approach with
            # Hilbert ordering on the grid indices
            
            break  # This approach is getting complex; let's use a simpler method
        
        # Simpler approach: generate a 3D grid and order it using a space-filling curve
        return None  # Placeholder
    
    # Let's use a much simpler and more reliable approach:
    # Generate points on a 3D grid and use a Hilbert curve ordering on the grid
    # For N=50, we can use a 4x4x4 grid (64 points) and take the first 50
    
    # Actually, the simplest robust approach: use a 3D Hilbert curve via
    # the algorithm from "A General Hilbert Curve" or similar
    # But for reliability, let's use a well-tested simple space-filling approach
    
    # We'll generate points using a recursive subdivision with a fixed order
    # that approximates a Hilbert curve
    
    points = []
    _hilbert_recursive(points, 0, 0, 0, 0, n)
    
    # Normalize to [0, 1)
    points = np.array(points) / n
    return points


def _hilbert_recursive(points, x, y, z, depth, n):
    """Recursively generate Hilbert-like ordered points in a 3D grid."""
    if depth == int(np.log2(n)) if n > 1 else 0:
        points.append((x, y, z))
        return
    
    half = n // (2 ** (int(np.log2(n)) - depth)) if depth < int(np.log2(n)) else 1
    # This is getting complicated. Let's use a completely different, simpler approach.


def construct():
    """
    Construct 50 sphere centers in the unit cube [-1/2, 1/2]^3 using a
    space-filling curve initialization followed by gradient-based optimization.
    """
    # Step 1: Initialize using a space-filling curve approach
    # For N=50, we'll use a 4x4x4 grid (64 points) and select 50
    # Order them using a Hilbert-like curve for better initial distribution
    
    m = 4  # 4^3 = 64 >= 50
    ticks = (np.arange(m) + 0.5) / m - 0.5  # In [-1/2, 1/2]
    
    # Generate all grid points
    all_points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Order them using a simple space-filling curve (Hilbert-like)
    # For a 4x4x4 grid, we can use a recursive ordering
    # Let's use a simple Z-order (Morton code) which is easier to implement
    # and still provides good space-filling properties
    
    def morton_code_3d(x, y, z, bits=2):
        """Compute 3D Morton code for coordinates in [0, 2^bits)."""
        code = 0
        for i in range(bits):
            code |= ((x >> i) & 1) << (3 * i)
            code |= ((y >> i) & 1) << (3 * i + 1)
            code |= ((z >> i) & 1) << (3 * i + 2)
        return code
    
    # Map grid indices to Morton codes
    grid_indices = np.arange(m ** 3).reshape(m, m, m)
    morton_codes = np.zeros((m, m, m), dtype=int)
    for i in range(m):
        for j in range(m):
            for k in range(m):
                morton_codes[i, j, k] = morton_code_3d(i, j, k, bits=int(np.log2(m)))
    
    # Sort by Morton code
    flat_indices = np.argsort(morton_codes.flatten())
    
    # Get the first N points in Morton order
    selected_indices = flat_indices[:N]
    centers = all_points[selected_indices].copy()
    
    # Step 2: Optimize using gradient-based method
    # Objective: maximize the minimum distance between any two centers
    # and from each center to the cube boundary
    # We'll use a smooth approximation: minimize the sum of inverse squared distances
    
    def objective(x):
        """
        x: flat array of shape (N*3,)
        Returns a scalar objective to minimize.
        We want to maximize the minimum pairwise distance and boundary distance.
        Use a smooth approximation: -log(sum of exp(-d_ij^2 / sigma)) for pairs,
        and similar for boundaries.
        """
        x = x.reshape(N, 3)
        
        # Pairwise distances
        diffs = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Boundary distances: distance to each face
        # For center c, distance to face at -0.5 is c_i + 0.5, to face at 0.5 is 0.5 - c_i
        # Minimum boundary distance for point i is min over all 6 faces
        # dist_to_face = min(c_i + 0.5, 0.5 - c_i) for each coordinate
        boundary_dists = np.minimum(x + 0.5, 0.5 - x)  # shape (N, 3)
        min_boundary = np.min(boundary_dists, axis=1)  # shape (N,)
        
        # Smooth minimum of pairwise distances
        # Use soft-min: -sigma * log(sum(exp(-d^2 / sigma)))
        sigma = 0.01
        # Only consider off-diagonal elements
        mask = ~np.eye(N, dtype=bool)
        dist_sq_off = dist_sq[mask]
        
        # Soft minimum of distances (not squared)
        # d = sqrt(dist_sq), soft_min(d) ≈ -sigma * log(sum(exp(-d^2 / sigma)))
        # But this is expensive. Let's use a simpler approach.
        
        # Alternative: minimize -log(min_pair_dist) approximately
        # Use: -log(sum(exp(-(d_ij - r0)^2 / sigma)) ) for pairs
        # and -log(sum(exp(-(d_i_boundary - r0)^2 / sigma))) for boundaries
        
        # Simpler: just minimize the negative of a soft minimum
        # soft_min(d) = -sigma * log(sum(exp(-d^2 / sigma)))
        
        # For pairs
        d_sq_pairs = dist_sq_off
        # Avoid log(0)
        d_sq_pairs = np.maximum(d_sq_pairs, 1e-10)
        soft_min_pairs = -sigma * np.log(np.sum(np.exp(-d_sq_pairs / sigma)))
        
        # For boundaries
        d_sq_boundary = min_boundary ** 2
        d_sq_boundary = np.maximum(d_sq_boundary, 1e-10)
        soft_min_boundary = -sigma * np.log(np.sum(np.exp(-d_sq_boundary / sigma)))
        
        # We want to maximize both, so minimize the negative
        # The radius is min(soft_min_pairs/2, soft_min_boundary)
        # Approximate by minimizing a combined objective
        obj = -(soft_min_pairs / 2 + soft_min_boundary)
        
        return obj
    
    # Use BFGS or L-BFGS-B for optimization
    # Start from the initial centers
    x0 = centers.flatten()
    
    # Add constraints to keep points inside the cube
    # We'll use bounds instead of explicit constraints
    bounds = [(-0.5 + 1e-6, 0.5 - 1e-6)] * (N * 3)
    
    # Run optimization
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    optimized_centers = result.x.reshape(N, 3)
    
    # Ensure points are strictly inside the cube
    optimized_centers = np.clip(optimized_centers, -0.5 + 1e-7, 0.5 - 1e-7)
    
    return optimized_centers

# EVOLVE-BLOCK-END
