import time
import numpy as np

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 140.0

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _min_enclosing_hex_side(points, outer_angle_deg=0.0):
    """
    Compute the minimum side length of a regular hexagon centered at origin
    with fixed rotation angle that encloses all given points.
    
    A regular hexagon centered at origin with side s and rotation angle alpha
    contains a point (x, y) if and only if:
    For each of the 6 sides, the signed distance from the point to the side 
    (measured inward) is >= 0.
    
    Alternatively, using support function:
    The hexagon is the intersection of 6 half-planes.
    The normal vectors to the sides are at angles: alpha + i*pi/3 for i=0..5?
    No, the sides are perpendicular to the normals.
    The vertices are at angles alpha + i*pi/3.
    The sides are between vertices.
    The normal to the side between vertex i and i+1 is at angle alpha + (i+0.5)*pi/3.
    The distance from center to side is s * sqrt(3) / 2.
    
    So for a point p, the condition to be inside is:
    dot(p, n_i) <= s * sqrt(3) / 2 for all i=0..5, where n_i are unit normals.
    
    Wait, the normals point outward.
    If n_i is the outward normal, then dot(p, n_i) <= s * sqrt(3) / 2.
    The outward normal angle for the side opposite to vertex i is...
    Let's define the hexagon by its 6 supporting lines.
    The distance from center to each side is apothem = s * sqrt(3) / 2.
    The outward normals are at angles: beta + i * pi/3, where beta is the angle of the first outward normal.
    For a hexagon with a vertex at angle alpha, the first side is between vertex alpha and alpha+pi/3.
    The midpoint of this side is at angle alpha + pi/6.
    So the outward normal is at angle alpha + pi/6.
    Thus, outward normals are at angles: alpha + pi/6 + i * pi/3 for i=0..5.
    
    Condition: dot(p, n_i) <= apothem for all i.
    So apothem >= max_i dot(p, n_i).
    Side s = 2 * apothem / sqrt(3).
    """
    a = np.radians(outer_angle_deg)
    # Outward normal angles
    normal_angles = a + np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    
    # points: (N, 2)
    # dot products: (N, 6)
    dots = points[:, 0] * nx[None, :] + points[:, 1] * ny[None, :]
    
    # We need apothem >= max dot for all points.
    # max_dot = max over points and normals
    max_dot = np.max(dots)
    
    apothem = max_dot
    side = 2 * apothem / np.sqrt(3)
    return side

def construct():
    global START_TIME, DEADLINE
    
    # Define the fixed relative positions of the 11 hexagons.
    # This is the standard dense packing:
    # 1 central hexagon at (0,0)
    # 6 surrounding hexagons at distance 2*sqrt(3) from center, angles 0, 60, ..., 300
    # 4 outer hexagons at distance 6 from center, angles 30, 150, 210, 330
    
    sqrt3 = np.sqrt(3)
    
    # Relative positions (local coordinates, cluster center at origin)
    # Hex 0: center
    # Hex 1-6: ring 1
    # Hex 7-10: ring 2
    
    rel_pos = np.zeros((N, 2))
    rel_pos[0] = [0.0, 0.0]
    
    for i in range(6):
        angle = i * np.pi / 3
        rel_pos[i+1] = [2 * sqrt3 * np.cos(angle), 2 * sqrt3 * np.sin(angle)]
    
    # The 4 outer hexagons are in the "gaps" of the ring, further out.
    # Their angles are 30, 150, 210, 330 degrees (pi/6, 5pi/6, 7pi/6, 11pi/6)
    # Their distance from center is 6.
    # Let's verify:
    # Hex 7 (angle 30) touches Hex 1 (angle 0) and Hex 2 (angle 60).
    # Dist(7,1) should be 2*sqrt(3).
    # pos7 = (6cos30, 6sin30) = (3sqrt3, 3)
    # pos1 = (2sqrt3, 0)
    # diff = (sqrt3, 3). dist = sqrt(3+9) = sqrt(12) = 2sqrt3. Correct.
    
    outer_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for i in range(4):
        angle = outer_angles[i]
        rel_pos[i+7] = [6.0 * np.cos(angle), 6.0 * np.sin(angle)]
    
    # All inner hexagons have the same orientation relative to the cluster.
    # In the optimal packing, they are all aligned with the lattice.
    # Let's assume all inner hexagons have angle 0 in the local frame.
    # We can also optimize the internal rotation if needed, but usually 0 is optimal for this symmetric cluster.
    
    best_side = float('inf')
    best_rotation = 0.0
    
    # We need to find the optimal global rotation of the cluster relative to the outer hexagon.
    # The outer hexagon can be fixed at angle 0.
    # We rotate the cluster by angle theta.
    
    # The function side(theta) is periodic with period pi/3 (60 degrees) due to symmetry.
    # We can scan theta from 0 to pi/3.
    
    # Since the function is likely smooth and unimodal in this range, we can use a fine scan or golden section.
    # Let's do a coarse scan first to find the minimum, then refine.
    
    theta_coarse = np.linspace(0, np.pi/3, 100)
    
    min_val_coarse = float('inf')
    min_theta_coarse = 0.0
    
    for theta in theta_coarse:
        # Rotate relative positions
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array([[c, -s], [s, c]])
        pos = rel_pos @ rot_mat.T
        
        side = _min_enclosing_hex_side(pos, 0.0)
        
        if side < min_val_coarse:
            min_val_coarse = side
            min_theta_coarse = theta
            
    # Refine around the best coarse angle
    # Use Nelder-Mead or simple ternary search on the interval around min_theta_coarse
    
    # Let's use scipy minimize for precision
    from scipy.optimize import minimize_scalar
    
    def obj(theta):
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array([[c, -s], [s, c]])
        pos = rel_pos @ rot_mat.T
        return _min_enclosing_hex_side(pos, 0.0)
    
    # Bounds for the local minimum
    # The period is pi/3. The minimum found is at min_theta_coarse.
    # We can search in [min_theta_coarse - pi/60, min_theta_coarse + pi/60]
    # But we must respect the periodicity.
    # Actually, since it's periodic, we can just search [0, pi/3] with minimize_scalar if we handle boundaries.
    # Or just refine the specific interval.
    
    # Let's assume the minimum is unique in [0, pi/3].
    # We'll search in a small window around the coarse best.
    
    if min_theta_coarse > 0.01 and min_theta_coarse < np.pi/3 - 0.01:
        a_opt = max(0, min_theta_coarse - 0.1)
        b_opt = min(np.pi/3, min_theta_coarse + 0.1)
    else:
        a_opt = 0.0
        b_opt = np.pi/3
        
    res = minimize_scalar(obj, bounds=(a_opt, b_opt), method='bounded', options={'xatol': 1e-10})
    
    if res.success:
        best_rotation = res.x
        best_side = res.fun
    else:
        best_rotation = min_theta_coarse
        best_side = min_val_coarse
        
    # Construct the final positions
    c, s = np.cos(best_rotation), np.sin(best_rotation)
    rot_mat = np.array([[c, -s], [s, c]])
    final_pos = rel_pos @ rot_mat.T
    
    # Inner hexagons angles are 0 (aligned with cluster)
    final_angles = np.zeros(N)
    
    inner_list = [[float(final_pos[i,0]), float(final_pos[i,1]), float(final_angles[i])] for i in range(N)]
    
    # Outer hexagon center is (0,0), side is best_side, angle is 0.
    return inner_list, (0.0, 0.0), float(best_side), 0.0

if __name__ == "__main__":
    construct()
