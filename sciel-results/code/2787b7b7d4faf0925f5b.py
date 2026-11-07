"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _generate_icosahedral(n, spacing=1.0):
    """
    Generate positions for an icosahedral cluster.
    Uses nested shells. 
    Shell 0: 1 atom
    Shell 1: 12 atoms
    Shell 2: 42 atoms
    Shell 3: 92 atoms
    ...
    We generate enough shells to cover n and take the first n.
    """
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # We need to generate shells.
    # A simple way is to use a Fibonacci sphere for the outer shell if we don't have exact shell counts,
    # but for icosahedral symmetry, we can just place points on a sphere with icosahedral distribution.
    # However, for N=13, 55, 147, exact shells are better.
    # Let's generate a large set of points on an icosahedral grid and pick the closest N to origin.
    # This is complex. Alternative: Use known coordinates for small N, or approximate with sphere.
    
    # Let's use a robust "dense icosahedral" approximation:
    # Generate points on a sphere using the golden spiral (Fibonacci) which approximates uniform distribution,
    # but we want icosahedral symmetry for the first shell.
    
    # Actually, for N=38, 55, 59, 75, 79, 80, random compact or FCC might be better than pure icosahedral if the icosahedral shell doesn't fit.
    # Let's create a helper that generates points on a sphere with radius r.
    
    def fib_sphere(n_pts, radius, seed=0):
        rng = np.random.default_rng(seed)
        # Add small noise to break symmetry slightly? No, keep symmetric for now.
        # Fibonacci sphere
        phi_vals = np.linspace(0, 1, n_pts)
        # Avoid poles if n_pts is small?
        theta = np.arccos(1 - 2 * phi_vals)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        # Add a slight random offset to theta to avoid perfect alignment issues if needed, 
        # but standard fib sphere is good.
        theta += rng.uniform(-0.1, 0.1, n_pts)
        phi = golden_angle * np.arange(n_pts)
        
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        return np.column_stack((x, y, z))

    # Strategy: 
    # 1. Center atom
    # 2. If n > 1, add 12 icosahedral vertices at r1
    # 3. If n > 13, add more points. 
    # For simplicity and robustness against N not being a magic number, 
    # let's just generate a "ball" of points.
    
    # Better approach for arbitrary N:
    # 1. Start with 1 point.
    # 2. Add points one by one on a sphere of increasing radius to maintain packing.
    # This is slow.
    
    # Let's stick to specific generators for known good structures:
    # 1. FCC fragment
    # 2. HCP fragment
    # 3. Icosahedral (if N is close to magic numbers, else truncate)
    
    # For now, let's just return the center + fib sphere for the rest, scaled to fit.
    # This is a "spherical" cluster, which is a good baseline.
    if n == 1:
        return np.array([0.0, 0.0, 0.0]).reshape(1, 3)
    
    # Radius estimation: N ~ (4/3) pi r^3 / v_atom
    # v_atom ~ spacing^3
    r = spacing * (3.0 * n / (4.0 * np.pi)) ** (1.0 / 3.0)
    
    # Generate n-1 points on sphere of radius r
    # Plus the center
    outer = fib_sphere(n - 1, r, seed=42)
    return np.vstack((np.array([0.0, 0.0, 0.0]), outer))

def _generate_fcc(n, spacing=1.0):
    """Generate an FCC fragment."""
    points = []
    # Generate a cubic grid of FCC points
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    basis = np.array([
        [0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]
    ])
    
    # Estimate side length of cube
    # Number of points ~ 4 * (L/spacing)^3
    L = spacing * (n / 4.0) ** (1.0 / 3.0)
    # Grid size
    size = int(np.ceil(L / spacing)) + 2
    
    grid = np.arange(-size, size + 1)
    for x in grid:
        for y in grid:
            for z in grid:
                for b in basis:
                    p = np.array([x, y, z]) + b
                    # Scale to spacing
                    p = p * spacing
                    # Center it roughly
                    points.append(p)
    
    points = np.array(points)
    # Center the cluster
    points -= np.mean(points, axis=0)
    
    # Sort by distance to center and take the first n
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    
    return points

def _generate_hcp(n, spacing=1.0):
    """Generate an HCP fragment."""
    points = []
    # HCP basis in a hexagonal prism
    # a = spacing, c/a = sqrt(8/3) ~ 1.633
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    
    # Generate layers
    # Layer 0: (0,0,0), (1,0,0), (0,1,0), (1,1,0) ... Hexagonal grid
    # Basis for hex layer:
    # v1 = (1, 0, 0), v2 = (1/2, sqrt(3)/2, 0)
    # Points: i*v1 + j*v2
    # Layer 1: (1/3, 2/(3*sqrt(3)), c/2) ... 
    # This is getting complicated. Let's use a simpler "random compact" or just FCC.
    # HCP is often similar to FCC for small clusters.
    # For now, return FCC-like structure with slight distortion or just use FCC.
    return _generate_fcc(n, spacing)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    # Time budget management
    start_time = time.time()
    total_time_limit = 300.0
    safety_margin = 10.0
    max_time = total_time_limit - safety_margin
    
    # Spacing parameter
    # For Sutton-Chen, equilibrium distance is around 0.72-0.75
    spacing = 0.72
    
    candidates = []
    
    # 1. FCC Fragment
    try:
        pts_fcc = _generate_fcc(size, spacing)
        candidates.append(pts_fcc)
    except Exception:
        pass
        
    # 2. Icosahedral / Spherical
    try:
        pts_ico = _generate_icosahedral(size, spacing)
        candidates.append(pts_ico)
    except Exception:
        pass
        
    # 3. Random Compact (a few variations)
    rng = np.random.default_rng(size)
    radius = spacing * size ** (1.0 / 3.0) * 0.8
    for i in range(3):
        points = []
        # Use rejection sampling with a tighter bound to ensure compactness
        # But for large N, this might fail. 
        # Better: Generate random points in a box, then relax?
        # The seed did this. Let's do a better random:
        # Generate N points uniformly in a sphere of radius R.
        # R is estimated from density.
        R = spacing * (3.0 * size / (4.0 * np.pi)) ** (1.0 / 3.0) * 1.2
        pts = rng.uniform(-R, R, (size, 3))
        # Filter points inside sphere
        norms = np.linalg.norm(pts, axis=1)
        mask = norms <= R
        if np.sum(mask) >= size:
             pts = pts[mask][:size]
        else:
             # Fallback: just take the first size points
             pts = pts[:size]
        candidates.append(pts)
        
    best_energy = float('inf')
    best_positions = None
    
    # Relax each candidate
    # We have limited time. 
    # Sort candidates by heuristic "quality"? 
    # FCC is usually good. Icosahedral is good for specific N.
    
    # Allocate time per candidate
    n_candidates = len(candidates)
    time_per_candidate = max_time / n_candidates
    
    for i, pts in enumerate(candidates):
        # Check time
        if time.time() - start_time > max_time:
            break
            
        # Ensure validity: no two atoms closer than 0.35
        # Quick check
        # For large N, O(N^2) check is expensive but N<=80 is small (6400 pairs)
        # We can skip the explicit check if we trust the generators, 
        # but relax might move them. 
        # The evaluator checks the final result. 
        # If relax produces invalid coords, it might return inf or crash?
        # The task says "Non-finite coordinates or two atoms closer than 0.35 make the whole result invalid."
        # So we must ensure the returned positions are valid.
        
        try:
            # Run relaxation
            # We don't know how long relax takes. 
            # If it takes too long, we might miss the deadline.
            # We can't interrupt relax easily.
            # So we must be careful with the number of relaxations.
            # For N=80, one L-BFGS-B run might take 1-5 seconds.
            # 5 candidates * 5s = 25s. Safe.
            
            energy, positions = relax(pts, potential, kind)
            
            if not np.isfinite(energy):
                continue
                
            # Validate positions
            if not np.all(np.isfinite(positions)):
                continue
                
            # Check min distance
            # Vectorized distance check
            # dists = np.sqrt(np.sum((positions[:, None, :] - positions[None, :, :])**2, axis=-1))
            # This is O(N^2) memory. N=80 -> 6400 floats. Fine.
            diff = positions[:, None, :] - positions[None, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            
            if min_dist < 0.35:
                # Invalid structure
                continue
                
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
                
        except Exception:
            continue
            
    # If no valid position found, return a dummy valid one?
    # We must return an (n, 3) array.
    if best_positions is None:
        # Fallback: generate a valid random sphere
        R = spacing * (3.0 * size / (4.0 * np.pi)) ** (1.0 / 3.0) * 1.5
        rng = np.random.default_rng(123)
        pts = rng.uniform(-R, R, (size, 3))
        # Force valid distances by scaling up if needed?
        # Just return this, hoping it's valid. 
        # Better: construct a grid of points with large spacing.
        # Grid side L such that L^3 * density >= N
        # Spacing 0.72. 
        # Let's just return the FCC one, it's usually valid.
        try:
            best_positions = _generate_fcc(size, spacing)
        except:
            best_positions = np.zeros((size, 3))
            # Add small noise to avoid exact overlaps
            best_positions += rng.uniform(-0.1, 0.1, (size, 3))
            
    return best_positions

# EVOLVE-BLOCK-END
