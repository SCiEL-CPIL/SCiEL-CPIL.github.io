"""Improved program using FCC lattice packing for 100 equal spheres in a unit cube."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree


def construct():
    """
    Generate a face-centered cubic (FCC) lattice of points, filter those inside the cube,
    and select the 100 points that maximize the minimum distance to the boundary.
    
    The FCC lattice is denser than simple cubic, allowing larger minimum distances between
    points. By selecting points with the largest minimum distance to the boundary, we
    ensure spheres can be as large as possible while staying inside the cube.
    """
    # FCC lattice: conventional cubic cell with points at corners and face centers
    # Generate a sufficiently large grid of FCC lattice points
    # FCC lattice vectors: a1 = a/2(0,1,1), a2 = a/2(1,0,1), a3 = a/2(1,1,0)
    # For simplicity, we'll generate points in a cubic supercell and keep those in [-0.5, 0.5]^3
    
    # The FCC lattice can be described as all points (i, j, k) where i+j+k is even,
    # scaled by a/2, plus the origin. More precisely, the FCC lattice consists of:
    # - All points (i, j, k) with i, j, k integers and i+j+k even, scaled by a/2
    # Let's use a lattice constant a such that the minimum distance between points is a/sqrt(2)
    # (nearest neighbors in FCC are at distance a/sqrt(2))
    
    # We want to generate enough points. Let's try different lattice spacings.
    # The cube is [-0.5, 0.5]^3, so side length is 1.
    # For FCC, the nearest neighbor distance is a/sqrt(2).
    # We want to maximize the minimum distance, so we want a as large as possible
    # while still having at least 100 points in the cube.
    
    # Let's generate FCC points with a range of lattice constants and pick the best.
    
    def generate_fcc_points(a, n_range):
        """Generate FCC lattice points with lattice constant a within [-n_range*a/2, n_range*a/2]^3."""
        points = []
        # FCC lattice: points at (i*a/2, j*a/2, k*a/2) where i+j+k is even
        for i in range(-n_range, n_range + 1):
            for j in range(-n_range, n_range + 1):
                for k in range(-n_range, n_range + 1):
                    if (i + j + k) % 2 == 0:
                        points.append((i * a / 2.0, j * a / 2.0, k * a / 2.0))
        return np.array(points)
    
    best_points = None
    best_score = 0
    
    # Try different lattice constants
    # The minimum distance between FCC points is a/sqrt(2)
    # We need at least 100 points in the cube [-0.5, 0.5]^3
    # Volume of cube is 1, so we need density >= 100
    # FCC has 4 atoms per cubic cell of side a, so density = 4/a^3
    # 4/a^3 >= 100 => a^3 <= 0.04 => a <= 0.342
    # But we also need points to be inside the cube, so let's be more careful.
    
    for a in np.linspace(0.1, 0.4, 41):
        # Estimate how many points we'll get
        # Points in [-0.5, 0.5]^3 with spacing a/2 in each direction (but only even sum)
        # Rough estimate: (1/(a/2))^3 / 2 = 8/a^3 / 2 = 4/a^3
        # We need at least 100, so 4/a^3 >= 100 => a <= (0.04)^(1/3) ≈ 0.342
        
        n_range = int(np.ceil(0.5 / (a / 2.0))) + 2
        points = generate_fcc_points(a, n_range)
        
        # Filter points inside the cube [-0.5, 0.5]^3
        mask = np.all(np.abs(points) <= 0.5, axis=1)
        points = points[mask]
        
        if len(points) < N:
            continue
        
        # Compute minimum distance to boundary for each point
        # Distance to boundary = min(0.5 - |x|, 0.5 - |y|, 0.5 - |z|)
        dist_to_boundary = np.min(0.5 - np.abs(points), axis=1)
        
        # Sort by distance to boundary, descending
        order = np.argsort(dist_to_boundary)[::-1]
        selected = points[order[:N]]
        
        # Compute the score: min(min pairwise distance / 2, min distance to boundary)
        # Compute minimum pairwise distance
        tree = cKDTree(selected)
        pairs = tree.query_pairs(1e10)  # All pairs within large distance
        if len(pairs) == 0:
            min_pair_dist = np.inf
        else:
            min_pair_dist = min(np.linalg.norm(selected[i] - selected[j]) for i, j in pairs)
        
        min_boundary_dist = np.min(dist_to_boundary[order[:N]])
        
        radius = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if radius > best_score:
            best_score = radius
            best_points = selected
    
    # If we didn't find any valid configuration, fall back to simple cubic
    if best_points is None:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_points
# EVOLVE-BLOCK-END
