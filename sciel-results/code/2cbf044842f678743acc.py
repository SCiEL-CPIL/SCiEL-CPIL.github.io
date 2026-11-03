"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 12
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    
    # Grid-based discrete search
    # The inner hexagons are placed on a triangular lattice with a single global rotation.
    # We search over rotation angles and lattice translations to minimize the bounding hexagon.
    
    # Triangular lattice basis vectors (for unit hexagons, distance between centers is 2*sqrt(3) for non-overlapping)
    # Actually, for unit hexagons (side length 1), the minimum center distance to avoid overlap is 2*sqrt(3).
    # Let's use a triangular lattice with spacing d = 2*sqrt(3) + epsilon.
    
    d = 2.0 * math.sqrt(3.0) + 1e-6  # minimum distance between hexagon centers
    a1 = np.array([d, 0.0])
    a2 = np.array([d/2.0, d*math.sqrt(3.0)/2.0])
    
    # We'll try a small grid of lattice points around the origin
    # and search over rotation angles
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try a range of rotation angles
    # The hexagon symmetry suggests checking angles in [0, 60) degrees
    angle_steps = np.linspace(0, math.pi/3, 61)  # 61 steps in [0, 60)
    
    for angle in angle_steps:
        if time.time() > DEADLINE:
            break
            
        # Rotate the lattice basis vectors
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        r1 = R @ a1
        r2 = R @ a2
        
        # Try a small combinatorial search over lattice translations
        # We'll place hexagons on lattice points (i, j) -> i*r1 + j*r2
        # and center them to minimize the bounding hexagon
        
        # For N=12, we need 12 lattice points
        # Let's try a compact arrangement: a hexagonal cluster of 12 points
        # A hexagonal number arrangement: 1 + 6 + ... 
        # For 12 points, we can use a 4x3 rectangular grid on the triangular lattice
        
        # Let's define candidate sets of 12 lattice points
        # Option 1: 4x3 grid
        # Option 2: Hexagonal cluster (1 center + 6 ring + 5 more)
        
        # Let's try a few patterns
        patterns = []
        
        # Pattern 1: 4x3 rectangular grid
        for i in range(-2, 2):
            for j in range(-1, 2):
                patterns.append((i, j))
        
        # Pattern 2: Hexagonal cluster (center + 6 neighbors + 5 more)
        hex_pattern = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
                       (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2)]
        patterns = hex_pattern
        
        # For each pattern, compute the positions and find the minimal bounding hexagon
        for pattern in [patterns]:  # Just use one pattern for now
            # Compute positions
            positions = []
            for (i, j) in pattern:
                pos = i * r1 + j * r2
                positions.append(pos)
            positions = np.array(positions)
            
            # Center the pattern
            center = np.mean(positions, axis=0)
            positions_centered = positions - center
            
            # Find the minimal bounding hexagon
            # A regular hexagon with side length s has vertices at distance s*sqrt(3)/2 from center
            # in directions 0, 60, 120, 180, 240, 300 degrees (or rotated)
            
            # The minimal bounding hexagon for a set of points with a given orientation
            # is determined by the maximum projection onto the 6 normal directions
            
            # For a hexagon with vertices at angles theta + k*pi/3, the supporting lines
            # are perpendicular to the edge directions, which are at theta + pi/6 + k*pi/3
            
            # Let's search over the hexagon orientation
            hex_angles = np.linspace(0, math.pi/3, 31)  # 31 steps in [0, 60)
            
            for hex_angle in hex_angles:
                if time.time() > DEADLINE:
                    break
                    
                # The 6 normal directions for the hexagon
                normals = []
                for k in range(6):
                    ang = hex_angle + k * math.pi / 3 + math.pi / 6
                    normals.append(np.array([math.cos(ang), math.sin(ang)]))
                
                # Project all points onto each normal
                max_projections = []
                for normal in normals:
                    projections = positions_centered @ normal
                    max_proj = np.max(projections)
                    min_proj = np.min(projections)
                    # The width in this direction is max_proj - min_proj
                    # For a hexagon, the side length s satisfies: width = s * sqrt(3)
                    # Actually, for a regular hexagon with side s, the distance between
                    # parallel sides is 2*s*sqrt(3)/2 = s*sqrt(3)
                    # So s = width / sqrt(3)
                    width = max_proj - min_proj
                    s = width / math.sqrt(3.0)
                    max_projections.append(s)
                
                # The required side length is the maximum of these
                required_s = max(max_projections)
                
                if required_s < best_side:
                    best_side = required_s
                    # Compute the inner positions with the optimal center
                    inner = []
                    for pos in positions:
                        # The position relative to the outer hexagon center
                        # We need to determine the outer hexagon center
                        # The center of the bounding box in each normal direction
                        # should be at the center of the outer hexagon
                        pass
                    
                    # For now, just update the best side length
                    # We'll compute the actual positions later
    
    # If we found a good configuration, use it
    # Otherwise, fall back to a simple arrangement
    
    if best_side < float('inf') and best_side < 40.0:
        # Use the best configuration found
        # For simplicity, let's use the hexagonal pattern centered at origin
        # with the optimal rotation angle
        
        # Recompute with the best angle
        # ... (simplified for now)
        pass
    
    # Fallback: simple linear arrangement
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END
