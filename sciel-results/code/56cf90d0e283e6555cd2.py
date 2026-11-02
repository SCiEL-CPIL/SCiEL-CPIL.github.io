import numpy as np
from scipy.optimize import minimize

N = 11

def _hex_vertices(angle_deg):
    """Return 6 unit vertices of a regular hexagon rotated by angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _outer_normals(phi_deg):
    """Return 6 outward unit normals for outer hexagon with rotation phi_deg."""
    phi = np.radians(phi_deg)
    angles = phi + np.arange(6) * np.pi / 3
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _compute_s(positions, angles_deg, cx, cy, phi_deg):
    """Compute minimum outer side length s needed."""
    normals = _outer_normals(phi_deg)
    max_val = 0.0
    for i in range(len(positions)):
        verts = _hex_vertices(angles_deg[i])
        for v in verts:
            p = positions[i] + v - np.array([cx, cy])
            dots = p @ normals.T
            max_val = max(max_val, np.max(dots))
    return max_val / (np.sqrt(3) / 2)

def _objective(params):
    """Unpack params and compute s."""
    n = N
    positions = params[0:2*n].reshape(n, 2)
    angles_deg = params[2*n:3*n]
    cx, cy = params[3*n], params[3*n+1]
    phi_deg = params[3*n+2]
    return _compute_s(positions, angles_deg, cx, cy, phi_deg)

def construct():
    best_s = float('inf')
    best_params = None
    
    # Multi-start optimization
    np.random.seed(42)
    
    # Initial guess: honeycomb-like arrangement
    # Center at 0, neighbors at distance 2 (hex side 1, touching means center distance 2*apothem = sqrt(3)?)
    # Actually for two unit hexagons touching, center distance = 2 * inradius = 2 * sqrt(3)/2 = sqrt(3)
    # Wait: inradius of unit hex = sqrt(3)/2. For edge-to-edge, center distance = 2 * sqrt(3)/2 = sqrt(3)
    
    # Let's start with a compact cluster
    # 11 hexagons: try a 3-4-4 row pattern or a hexagonal number pattern
    # Hexagonal numbers: 1, 7, 19... so 11 is between 7 and 19
    # Try: center + 6 around + 4 more
    
    starts = []
    
    # Start 1: hexagonal cluster (7) + 4 extra
    base = np.array([[0, 0]])
    for k in range(6):
        a = k * np.pi / 3
        base = np.vstack([base, np.sqrt(3) * np.array([np.cos(a), np.sin(a)])])
    # Add 4 more at larger radius
    for k in range(4):
        a = (k + 0.5) * np.pi / 3
        base = np.vstack([base, 2 * np.sqrt(3) * np.array([np.cos(a), np.sin(a)])])
    starts.append(np.concatenate([base.flatten(), np.zeros(N), 0, 0, 0]))
    
    # Start 2: linear-ish but compact
    for trial in range(10):
        pos = np.random.randn(N, 2) * 3
        ang = np.random.uniform(0, 60, N)
        cx, cy = 0, 0
        phi = 0
        params = np.concatenate([pos.flatten(), ang, [cx, cy, phi]])
        starts.append(params)
    
    for params0 in starts:
        res = minimize(_objective, params0, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.success and res.fun < best_s:
            best_s = res.fun
            best_params = res.x
    
    if best_params is None:
        # Fallback
        return [[0.0, 0.0, 0.0]] * N, (0.0, 0.0), 10.0, 0.0
    
    positions = best_params[0:2*N].reshape(N, 2)
    angles_deg = best_params[2*N:3*N]
    cx, cy = best_params[3*N], best_params[3*N+1]
    phi_deg = best_params[3*N+2]
    
    inner = [[float(positions[i][0]), float(positions[i][1]), float(angles_deg[i])] for i in range(N)]
    return inner, (float(cx), float(cy)), float(best_s), float(phi_deg)
