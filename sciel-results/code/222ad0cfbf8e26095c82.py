import math
import numpy as np
import time
from scipy.optimize import linprog, minimize
from scipy.spatial import ConvexHull

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

# Precompute D_min for relative angles
# D_min(delta_phi) is the minimum distance between centers of two non-overlapping unit hexagons
def compute_D_min_table():
    # delta_phi in [0, pi/6]
    # We can calculate this by finding the minimum norm of the boundary of H + (-H_rotated)
    # Or simply: for a given delta_phi, the Minkowski sum is a polygon.
    # The minimum distance is the minimum support function of the Minkowski sum?
    # Actually, if the origin is in the interior of H + (-H), they overlap.
    # So we need the distance from origin to the boundary of H + (-H).
    
    # Vertices of unit hexagon (side 1, flat top? or pointy top?)
    # Let's use flat top (vertex at 90 deg? No, flat top means horizontal edge at top).
    # Standard hexagon vertices:
    # (1, 0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2), (-1, 0), (-1/2, -sqrt(3)/2), (1/2, -sqrt(3)/2)
    # This has side length 1? Distance between (1,0) and (1/2, sqrt(3)/2) is sqrt(1/4 + 3/4) = 1. Yes.
    
    def hex_vertices(angle=0.0):
        verts = []
        for i in range(6):
            a = angle + i * math.pi / 3.0
            verts.append((math.cos(a), math.sin(a)))
        return np.array(verts)
    
    v0 = hex_vertices(0.0)
    
    D_table = {}
    for deg in range(0, 31, 1):
        dp = math.radians(deg)
        v1 = hex_vertices(dp)
        # Minkowski sum H + (-H_rot)
        # Vertices of sum are sums of vertices? No, we need the convex hull of all sums.
        # But for convex polygons, the Minkowski sum vertices are subsets of the sums of vertices.
        # We can compute the convex hull of all v0_i + (-v1_j).
        
        sums = []
        for v0_i in v0:
            for v1_j in v1:
                sums.append(v0_i - v1_j)
        
        sums = np.array(sums)
        
        # Find the minimum distance from origin to the convex hull of sums
        # This is the minimum norm of a point in the hull?
        # No, the origin is inside the hull if they overlap.
        # We want the minimum distance such that the origin is OUTSIDE.
        # This is the distance from origin to the boundary of the hull.
        
        try:
            hull = ConvexHull(sums)
            # Check if origin is inside
            # If inside, distance is 0 (overlap).
            # If outside, distance is min distance to edges.
            
            # Simple check: project origin onto each edge of the hull
            min_dist = float('inf')
            for i in range(len(hull.vertices)):
                p1 = sums[hull.vertices[i]]
                p2 = sums[hull.vertices[(i + 1) % len(hull.vertices)]]
                
                # Vector p1 to p2
                v = p2 - p1
                # Vector p1 to origin
                w = -p1
                
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                
                if c1 <= 0:
                    dist = np.linalg.norm(w)
                elif c1 >= c2:
                    dist = np.linalg.norm(w + v)
                else:
                    t = c1 / c2
                    proj = p1 + t * v
                    dist = np.linalg.norm(-proj)
                
                if dist < min_dist:
                    min_dist = dist
            
            D_table[deg] = min_dist
        except Exception:
            D_table[deg] = 2.0 # Fallback
            
    return D_table

D_MIN_TABLE = compute_D_min_table()

def get_D_min(delta_phi):
    # delta_phi in radians, normalize to [0, pi/6]
    dp = delta_phi % (math.pi / 3.0) # Hexagon symmetry is 60 deg
    if dp > math.pi / 6.0:
        dp = math.pi / 3.0 - dp
    deg = int(round(math.degrees(dp)))
    # Interpolate? For now, use nearest
    if deg not in D_MIN_TABLE:
        deg = min(D_MIN_TABLE.keys(), key=lambda x: abs(x - deg))
    return D_MIN_TABLE[deg]

def get_min_containing_hex(pts, angles):
    # pts: (N, 2)
    # angles: (N,)
    # Returns min_S, theta, center
    
    best_S = float('inf')
    best_theta = 0.0
    best_center = (0.0, 0.0)
    
    # Optimize theta in [0, pi/6]
    for i in range(19): # 0 to 30 deg, step 1.5 deg
        theta = i * (math.pi / 6.0) / 18.0
        
        # Normals of outer hexagon
        normals = []
        for k in range(6):
            ang = theta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        A_ub = []
        b_ub = []
        
        # Constraints for each inner hexagon
        for i_h in range(len(pts)):
            cx_h, cy_h = pts[i_h]
            ang_h = angles[i_h]
            
            # Vertices of inner hexagon
            verts = []
            for k in range(6):
                va = ang_h + k * math.pi / 3.0
                vx = cx_h + math.cos(va)
                vy = cy_h + math.sin(va)
                verts.append((vx, vy))
            
            for vx, vy in verts:
                for nx, ny in normals:
                    # Constraint: (vx - cx) * nx + (vy - cy) * ny <= S * APOTHEM
                    # => -nx * cx - ny * cy - APOTHEM * S <= - (vx * nx + vy * ny)
                    A_ub.append([-nx, -ny, -APOTHEM])
                    b_ub.append(-(vx * nx + vy * ny))
                    
                    # Constraint: -(vx - cx) * nx - (vy - cy) * ny <= S * APOTHEM
                    # => nx * cx + ny * cy - APOTHEM * S <= vx * nx + vy * ny
                    A_ub.append([nx, ny, -APOTHEM])
                    b_ub.append(vx * nx + vy * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                S = res.x[2]
                if S < best_S:
                    best_S = S
                    best_theta = theta
                    best_center = (res.x[0], res.x[1])
        except Exception:
            pass
            
    return best_S, best_theta, best_center

def calc_overlap_penalty(pts, angles):
    penalty = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[j])
            dp = angles[i] - angles[j]
            d_min = get_D_min(dp)
            if d < d_min:
                penalty += (d_min - d) ** 2
    return penalty

def objective(params):
    # params: 33 values (11 x, 11 y, 11 angles)
    pts = np.array(params[:22]).reshape(11, 2)
    angles = params[22:]
    
    # Check overlap
    pen = calc_overlap_penalty(pts, angles)
    if pen > 1e-6:
        return 1000.0 + pen
    
    # Get min containing hex
    S, theta, center = get_min_containing_hex(pts, angles)
    
    if S == float('inf'):
        return 10000.0
        
    return S

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Initial guesses
    # 1. Flower pattern
    pts_init = []
    angles_init = []
    pts_init.append([0.0, 0.0])
    angles_init.append(0.0)
    for k in range(6):
        a = k * math.pi / 3.0
        pts_init.append([math.cos(a) * 1.75, math.sin(a) * 1.75])
        angles_init.append(0.0)
    # 4 more
    for k in range(4):
        a = k * math.pi / 2.0 + math.pi / 4.0
        pts_init.append([math.cos(a) * 3.0, math.sin(a) * 3.0])
        angles_init.append(0.0)
        
    params_init = np.array(pts_init).flatten().tolist() + angles_init
    
    best_S = float('inf')
    best_params = None
    best_theta = 0.0
    best_center = (0,0)
    
    starts = [params_init]
    # Add random starts
    for _ in range(5):
        rand_pts = np.random.uniform(-3, 3, (11, 2))
        rand_angles = np.random.uniform(0, math.pi/3, 11)
        starts.append(rand_pts.flatten().tolist() + rand_angles.tolist())
        
    for start in starts:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < best_S and res.fun < 1000:
                best_S = res.fun
                best_params = res.x
                # Recover theta and center
                pts = np.array(best_params[:22]).reshape(11, 2)
                angles = best_params[22:]
                _, best_theta, best_center = get_min_containing_hex(pts, angles)
        except Exception:
            pass
            
    if best_params is None:
        # Use the fallback from the seed program logic if optimization failed
        # Simple grid
        best_S = 4.0
        best_theta = 0.0
        best_center = (0,0)
        pts = np.array([[0,0], [1.8,0], [-1.8,0], [0,1.8], [0,-1.8], [1.5,1.5], [-1.5,1.5], [1.5,-1.5], [-1.5,-1.5], [3,0], [0,3]])
        angles = np.zeros(11)
    else:
        pts = np.array(best_params[:22]).reshape(11, 2)
        angles = best_params[22:]
        
    # Final validation and scaling
    # We might need to scale down slightly to ensure no floating point errors cause overlap
    # Or just return as is. The evaluator is strict.
    
    # Re-calculate exact S for the final config
    S_final, theta_final, center_final = get_min_containing_hex(pts, angles)
    
    inner = []
    for i in range(11):
        inner.append([float(pts[i][0]), float(pts[i][1]), float(math.degrees(angles[i]))])
        
    center = [float(center_final[0]), float(center_final[1])]
    side = float(S_final)
    rotation = float(math.degrees(theta_final))
    
    return inner, center, side, rotation

