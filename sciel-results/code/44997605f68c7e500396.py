import numpy as np
import time
from scipy.optimize import minimize

N = 12

def hex_support(theta, u):
    """
    Calculate the support function of a unit regular hexagon 
    with center at origin, rotation theta, in direction u (unit vector).
    u is (ux, uy).
    """
    # Vertices of unit hexagon rotated by theta
    # Angles: theta + k * pi/3 for k=0..5
    k = np.arange(6)
    angles = theta + k * np.pi / 3.0
    vx = np.cos(angles)
    vy = np.sin(angles)
    
    # Dot product with u
    dots = vx * u[0] + vy * u[1]
    return np.max(dots)

def get_inner_vertices(theta):
    """Get vertices of a unit hexagon rotated by theta."""
    k = np.arange(6)
    angles = theta + k * np.pi / 3.0
    return np.column_stack((np.cos(angles), np.sin(angles)))

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin for 150s limit
    
    # Outer hexagon parameters
    # Fixed center (0,0), fixed rotation 0
    # Side length S is a variable
    
    # Variables: [S, x0, y0, th0, x1, y1, th1, ..., x11, y11, th11]
    # Size: 1 + 12*3 = 37
    
    def objective(x):
        return x[0] # Minimize S
    
    def constraints(x):
        S = x[0]
        if S <= 0:
            return [0.0] * 100 # Invalid, return small values to push away? 
                               # Better to handle in bounds or penalty.
                               # SLSQP expects c(x) >= 0.
        
        # Extract positions and rotations
        centers = []
        thetas = []
        for i in range(N):
            base_idx = 1 + i * 3
            cx = x[base_idx]
            cy = x[base_idx + 1]
            th = x[base_idx + 2]
            centers.append((cx, cy))
            thetas.append(th)
            
        cons = []
        
        # 1. Non-overlap constraints
        # For each pair i, j, distance >= h_i(u) + h_j(u)
        for i in range(N):
            for j in range(i + 1, N):
                cx_i, cy_i = centers[i]
                cx_j, cy_j = centers[j]
                dx = cx_j - cx_i
                dy = cy_j - cy_i
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < 1e-8:
                    # If centers are same, overlap is huge
                    cons.append(-10.0) 
                    continue
                
                ux = dx / dist
                uy = dy / dist
                u = (ux, uy)
                
                h_i = hex_support(thetas[i], u)
                h_j = hex_support(thetas[j], u)
                
                # Constraint: dist - (h_i + h_j) >= 0
                val = dist - (h_i + h_j)
                cons.append(val)
                
        # 2. Containment constraints
        # Outer hexagon inradius R = S * sqrt(3) / 2
        R_out = S * np.sqrt(3) / 2.0
        
        # Normals for outer hexagon (rotation 0)
        # Edge normals are at 30, 90, 150, 210, 270, 330 degrees
        k_out = np.arange(6)
        angles_out = np.pi/6.0 + k_out * np.pi/3.0
        nx = np.cos(angles_out)
        ny = np.sin(angles_out)
        
        for i in range(N):
            cx, cy = centers[i]
            th = thetas[i]
            
            for k in range(6):
                u = (nx[k], ny[k])
                # Projection of center onto normal
                proj_c = cx * nx[k] + cy * ny[k]
                # Support function of inner hex in this direction
                h_in = hex_support(th, u)
                
                # Constraint: proj_c + h_in <= R_out
                # => R_out - proj_c - h_in >= 0
                val = R_out - proj_c - h_in
                cons.append(val)
                
        return cons

    def bounds():
        b = [(0.1, 20.0)] # S bounds
        for i in range(N):
            b.append((-10.0, 10.0)) # x
            b.append((-10.0, 10.0)) # y
            b.append((-np.pi, np.pi)) # theta
        return b

    best_S = 100.0
    best_x = None
    
    # Initial guess: Hexagonal lattice
    # Center at 0,0
    # Ring 1: 6 hexagons at distance sqrt(3)
    # Ring 2: 6 hexagons at distance 2*sqrt(3)
    
    r1 = np.sqrt(3)
    r2 = 2 * np.sqrt(3)
    
    init_centers = [(0.0, 0.0)]
    for k in range(6):
        angle = k * np.pi / 3.0
        init_centers.append((r1 * np.cos(angle), r1 * np.sin(angle)))
    for k in range(6):
        # Pick every other one in the second ring to make 12 total?
        # Second ring has 12 spots. We need 6 more.
        # Let's place them at angles 0, 60, 120, 180, 240, 300 for the outer ring too?
        # That would overlap with ring 1 directions.
        # In a hex lattice, the 12 neighbors of the center are at:
        # 6 at dist r1, angles 0, 60...
        # 6 at dist r1, angles 30, 90... (these are the "diagonal" neighbors? No)
        # Actually, the 12 closest neighbors in a hex lattice are:
        # 6 at distance r1 (sharing edges)
        # 6 at distance r1 * sqrt(3)? No.
        # Let's just use a generic cluster.
        angle = k * np.pi / 3.0 + np.pi/6.0
        init_centers.append((r2 * np.cos(angle), r2 * np.sin(angle)))
        
    # Wait, 1 + 6 + 6 = 13. We need 12. Remove one.
    init_centers = init_centers[:12]
    
    init_thetas = [0.0] * 12
    
    x0 = [10.0] # Initial S
    for i in range(12):
        x0.append(init_centers[i][0])
        x0.append(init_centers[i][1])
        x0.append(init_thetas[i])
    
    # Run optimization from a few starts
    starts = [x0]
    
    # Add some random perturbations
    np.random.seed(42)
    for _ in range(10):
        x_rand = x0.copy()
        x_rand[0] = 8.0 + np.random.rand() * 4.0
        for i in range(12):
            base = 1 + i * 3
            x_rand[base] += np.random.randn() * 0.5
            x_rand[base+1] += np.random.randn() * 0.5
            x_rand[base+2] += np.random.randn() * 0.2
        starts.append(x_rand)
        
    for x_start in starts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, x_start, method='SLSQP', bounds=bounds(), 
                           constraints=[{'type': 'ineq', 'fun': constraints}],
                           options={'maxiter': 500, 'ftol': 1e-6})
            
            if res.success and res.x[0] < best_S:
                # Verify solution
                # Check if all constraints are satisfied
                vals = constraints(res.x)
                if np.all(np.array(vals) >= -1e-5):
                    best_S = res.x[0]
                    best_x = res.x
        except Exception as e:
            pass
            
    if best_x is None:
        # Fallback to initial guess if optimization failed
        best_x = x0
        best_S = x0[0]
        
    # Extract final solution
    S = best_x[0]
    inner = []
    for i in range(12):
        base = 1 + i * 3
        x = best_x[base]
        y = best_x[base+1]
        ang = best_x[base+2]
        # Convert radians to degrees for the output interface
        ang_deg = np.degrees(ang)
        inner.append([x, y, ang_deg])
        
    return inner, (0.0, 0.0), S, 0.0

# EVOLVE-BLOCK-END
