"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time

# Import the relaxation function if available
try:
    from science_chemistry import relax
except ImportError:
    # Fallback if import fails, though task says it's available
    def relax(positions):
        return positions

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s deadline to be safe
    
    # 1. Construct 13-atom icosahedron
    # Vertices of a regular icosahedron
    phi = (1 + np.sqrt(5)) / 2
    # Vertices: (0, +/-1, +/-phi), (+/-1, +/-phi, 0), (+/-phi, 0, +/-1)
    v = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    v = np.array(v, dtype=float)
    # Normalize to unit radius
    norms = np.linalg.norm(v, axis=1)
    v = v / norms[:, np.newaxis]
    
    # Scale to approximate LJ minimum distance for inner shell
    # Edge length of unit icosahedron is 2/sqrt(1+phi^2/2) ?
    # Distance between (0,1,phi) and (1,phi,0) is sqrt(1 + (phi-1)^2 + phi^2)
    # phi-1 = 1/phi
    # d^2 = 1 + 1/phi^2 + phi^2 = 1 + 1/phi^2 + (phi+1) = 1 + 1/phi^2 + phi + 1
    # phi^2 = phi + 1
    # d^2 = 2 + phi + 1/phi^2
    # 1/phi^2 = 2 - phi
    # d^2 = 2 + phi + 2 - phi = 4
    # d = 2
    # So for unit radius, edge length is 2.
    # We want edge length ~ 1.12 (LJ min)
    # Scale factor = 1.12 / 2 = 0.56
    scale_inner = 0.56
    inner_points = v * scale_inner
    
    # Add center atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    # 2. Add 42 outer atoms
    # Place them on a sphere of radius R_outer
    # Estimate R_outer. In Mackay, outer shell is further.
    # Let's try R_outer = 2.0 * scale_inner? No, scale_inner is radius.
    # Inner radius is 0.56.
    # Let's try R_outer = 1.5
    R_outer = 1.5
    
    # Generate 42 points on a sphere using Fibonacci lattice
    N_outer = 42
    i = np.arange(N_outer)
    # Fibonacci sphere
    phi_golden = np.pi * (3 - np.sqrt(5))
    y = 1 - (i / (N_outer - 1)) * 2  # y goes from 1 to -1
    radius = np.sqrt(1 - y**2)  # radius at y
    theta = phi_golden * i  # golden angle increment
    
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    
    outer_points = np.column_stack((x, y, z))
    outer_points = outer_points * R_outer
    
    # Combine
    positions = np.vstack([center, inner_points, outer_points])
    
    # Ensure minimum distance constraint is roughly met
    # If not, relax will handle it, but initial overlaps can be bad.
    # 0.56 radius inner, 1.5 radius outer. Distance between inner and outer is ~0.94.
    # Inner-inner distance is 1.12.
    # Outer-outer distance: 42 points on sphere R=1.5.
    # Surface area = 4 * pi * 1.5^2 = 28.27.
    # Area per point = 28.27 / 42 = 0.67.
    # Effective radius per point = sqrt(0.67/pi) = 0.46.
    # Chord distance approx 2 * 0.46 = 0.92.
    # This is < 0.5? No, 0.92 > 0.5. So it should be valid.
    
    # 3. Relax
    try:
        relaxed_positions = relax(positions)
    except Exception:
        relaxed_positions = positions
        
    # Check time
    if time.time() > deadline:
        # If we are running out of time, return what we have
        pass
        
    return relaxed_positions
# EVOLVE-BLOCK-END
