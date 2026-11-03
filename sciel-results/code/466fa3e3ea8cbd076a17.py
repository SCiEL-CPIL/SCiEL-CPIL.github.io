import numpy as np
from scipy.optimize import minimize

N = 12

def get_hex_vertices(cx, cy, angle_deg, side=1.0):
    angle_rad = np.deg2rad(angle_deg)
    # Vertices of a unit hexagon centered at 0,0
    # Pointy top?
    # Angles: 0, 60, 120, 180, 240, 300
    # If angle=0, vertex at (1,0)?
    # Standard math: cos(0)=1, sin(0)=0.
    # Let's define vertices at angles 0, 60, ...
    angles = np.linspace(0, 2*np.pi, 7)[:-1] + angle_rad
    return np.column_stack((cx + side * np.cos(angles), cy + side * np.sin(angles)))

def check_intersection(v1, v2):
    # SAT check
    # v1, v2 are (6,2) arrays
    axes = []
    for i in range(6):
        # Normal to edge i of v1
        p1 = v1[i]
        p2 = v1[(i+1)%6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        normal = normal / np.linalg.norm(normal)
        axes.append(normal)
        
    # We only need to check axes from one polygon if they are regular hexagons?
    # No, if rotated, we need axes from both.
    for i in range(6):
        p1 = v2[i]
        p2 = v2[(i+1)%6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        normal = normal / np.linalg.norm(normal)
        axes.append(normal)
        
    for normal in axes:
        proj1 = np.dot(v1, normal)
        proj2 = np.dot(v2, normal)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 or max2 < min1:
            return False, 0.0
    # If no separating axis, they intersect.
    # Estimate penetration depth for penalty
    # Simple heuristic: distance between centers
    c1 = np.mean(v1, axis=0)
    c2 = np.mean(v2, axis=0)
    dist = np.linalg.norm(c1 - c2)
    # Max possible distance for non-intersection is ~2.0 (diagonal)
    # Min distance for non-intersection is ~1.732 (side)
    # Penalty should be large if dist < 1.7
    return True, max(0, 1.8 - dist)

def calc_min_L(centers, angles, outer_angle=0.0):
    # centers: (N, 2)
    # angles: (N,)
    # Outer hexagon centered at (0,0) for calculation?
    # We should center the system.
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    max_L = 0.0
    outer_angle_rad = np.deg2rad(outer_angle)
    
    for i in range(len(centers)):
        cx, cy = centers[i]
        ang = angles[i]
        verts = get_hex_vertices(cx, cy, ang)
        for vx, vy in verts:
            # Distance from origin
            r = np.sqrt(vx*vx + vy*vy)
            if r == 0: continue
            theta = np.arctan2(vy, vx)
            
            # Distance from center to boundary of outer hexagon at angle theta
            # Outer hexagon side L.
            # Apothem a = sqrt(3)/2 * L
            # Angle relative to nearest normal
            # Normals are at 0, 60, 120... + outer_angle
            # Let's find the angle to the nearest normal
            delta = (theta - outer_angle_rad) % (np.pi/3)
            # delta is in [0, 60). We want distance to nearest, so min(delta, 60-delta)
            # Actually, the normal is at 0. The side is at 30 deg from normal?
            # No.
            # Hexagon vertices at 0, 60...
            # Sides are perpendicular to angles 30, 90, 150...
            # So normals are at 30, 90...
            # Let's adjust.
            # If vertices are at 0,60, then normals are at 30, 90...
            # So we shift theta by 30 deg.
            theta_shifted = (theta - outer_angle_rad - np.pi/6) % (np.pi/3)
            # theta_shifted is in [0, 60).
            # Distance to boundary is a / cos(theta_shifted)
            # a = sqrt(3)/2 * L
            # r <= a / cos(theta_shifted)
            # L >= 2/sqrt(3) * r * cos(theta_shifted)
            
            # Handle case where cos is negative? No, theta_shifted in [0,60], cos is positive.
            L_required = (2.0 / np.sqrt(3.0)) * r * np.cos(theta_shifted)
            if L_required > max_L:
                max_L = L_required
    return max_L, centroid

def objective(params):
    # params: [x1, y1, a1, ..., x12, y12, a12]
    centers = np.array(params[0:2*N]).reshape(N, 2)
    angles = np.array(params[2*N:3*N])
    
    # 1. Calculate L for containment
    L, centroid = calc_min_L(centers, angles)
    
    # 2. Check intersections
    penalty = 0.0
    verts_list = []
    for i in range(N):
        verts_list.append(get_hex_vertices(centers[i,0], centers[i,1], angles[i]))
        
    for i in range(N):
        for j in range(i+1, N):
            inter, pen = check_intersection(verts_list[i], verts_list[j])
            if inter:
                penalty += pen * 10.0 # Weight penalty heavily
    
    return L + penalty

# EVOLVE-BLOCK-START
def construct():
    # Initial guess: 1-2-3-3-2-1 cluster
    u = np.array([np.sqrt(3), 0.0])
    v = np.array([np.sqrt(3)/2, 1.5])
    
    points = []
    # Row 0: 1
    points.append(0*u + 0*v)
    # Row 1: 2
    points.append(0*u + 1*v)
    points.append(1*u + 1*v)
    # Row 2: 3
    points.append(0*u + 2*v)
    points.append(1*u + 2*v)
    points.append(2*u + 2*v)
    # Row 3: 3
    points.append(0*u + 3*v)
    points.append(1*u + 3*v)
    points.append(2*u + 3*v)
    # Row 4: 2
    points.append(0*u + 4*v)
    points.append(1*u + 4*v)
    # Row 5: 1
    points.append(0*u + 5*v)
    
    points = np.array(points)
    # Center the points
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    initial_params = []
    for i in range(N):
        initial_params.append(points[i,0])
        initial_params.append(points[i,1])
        initial_params.append(0.0) # angle
    
    # Optimize
    # Use Nelder-Mead for robustness with non-differentiable penalties
    result = minimize(objective, initial_params, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    best_params = result.x
    centers = best_params[0:2*N].reshape(N, 2)
    angles = best_params[2*N:3*N]
    
    # Calculate final L and center
    L, centroid = calc_min_L(centers, angles)
    
    # Construct return values
    inner = []
    for i in range(N):
        inner.append([centers[i,0], centers[i,1], np.rad2deg(angles[i])])
    
    return inner, (centroid[0], centroid[1]), L, 0.0
# EVOLVE-BLOCK-END
