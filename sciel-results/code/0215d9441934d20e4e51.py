"""Optimized packing of 12 unit hexagons in a minimal outer hexagon."""
import time
import math
import random
import numpy as np

N = 12
DEADLINE = 145.0  # safety margin before the 150s kill

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _hex_vertices(theta):
    """Regular hexagon, side 1, circumradius 1, rotated by theta, centered at origin."""
    return np.array([[math.cos(theta + k * math.pi / 3.0),
                      math.sin(theta + k * math.pi / 3.0)] for k in range(6)])

def _hex_axes(theta):
    """6 edge-normal directions (unit vectors) of a hexagon oriented by theta."""
    return np.array([[math.cos(theta + math.pi / 6.0 + k * math.pi / 3.0),
                      math.sin(theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])

def _project_polygon(verts, axes):
    """Project polygon vertices onto given axes. Returns array of shape (len(axes), 2) = (min, max)."""
    P = verts @ axes.T
    return np.vstack([P.min(axis=0), P.max(axis=0)]).T  # shape (n_axes, 2)

def _sat_overlap(A, B):
    """SAT overlap test between two hexagons. Returns True if overlapping."""
    EPS = 1e-9
    # Get all 12 axes (6 from each hexagon)
    axesA = _hex_axes_from_verts(A)
    axesB = _hex_axes_from_verts(B)
    all_axes = np.vstack([axesA, axesB])
    
    projA = _project_polygon(A, all_axes)
    projB = _project_polygon(B, all_axes)
    
    # Check separation on each axis
    for i in range(len(all_axes)):
        if projA[i, 1] < projB[i, 0] - EPS or projB[i, 1] < projA[i, 0] - EPS:
            return False
    return True

def _hex_axes_from_verts(verts):
    """Compute edge normals from polygon vertices."""
    axes = []
    for k in range(6):
        p1 = verts[k]
        p2 = verts[(k + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        nl = np.linalg.norm(normal)
        if nl > 1e-9:
            axes.append(normal / nl)
    return np.array(axes)

def _outer_hex_side(all_pts, theta):
    """Compute side length of smallest regular hexagon (orientation theta, optimal center) containing points."""
    axes = _hex_axes(theta)
    P = all_pts @ axes.T
    # Optimal center in normal space: midpoint of min and max
    c = (P.min(axis=0) + P.max(axis=0)) / 2.0
    # Side length: 2 * max deviation / sqrt(3)
    max_dev = np.max(P.max(axis=0) - c)
    return 2.0 * max_dev / math.sqrt(3.0)

def _compute_best_outer(all_pts):
    """Find best outer hexagon orientation and center for given set of points."""
    best_side = float('inf')
    best_center = np.array([0.0, 0.0])
    best_angle = 0.0
    
    # Sweep over 60-degree range (hexagon symmetry)
    for ang in np.linspace(0.0, math.pi / 3.0, 361):
        axes = _hex_axes(ang)
        P = all_pts @ axes.T
        c = (P.min(axis=0) + P.max(axis=0)) / 2.0
        side = 2.0 * np.max(P.max(axis=0) - c) / math.sqrt(3.0)
        if side < best_side - 1e-12:
            best_side = side
            best_center = axes.T @ c
            best_angle = float(ang)
    
    return best_side, best_center, best_angle

def _is_valid_fast(centers, angles, outer_side, outer_theta, outer_center):
    """Check if configuration is valid: no overlaps, all contained in outer hexagon."""
    EPS = 1e-9
    verts = [_hex_vertices(angles[k]) + centers[k] for k in range(N)]
    
    # Pairwise overlap check
    for i in range(N):
        for j in range(i + 1, N):
            if _sat_overlap(verts[i], verts[j]):
                return False
    
    # Containment check
    axes = _hex_axes(outer_theta)
    oc = np.array(outer_center) @ axes.T
    for k in range(N):
        P = verts[k] @ axes.T
        for face in range(6):
            if (P[:, face] - oc[face]).max() > outer_side * math.sqrt(3.0) / 2.0 + EPS:
                return False
    
    return True

def _quick_score(centers, angles):
    """Quick score: bounding hexagon side length (ignores validity)."""
    all_pts = np.vstack([_hex_vertices(angles[k]) + centers[k] for k in range(N)])
    return _outer_hex_side(all_pts, 0.0)

# ---------------------------------------------------------------------------
# Objective function with penalty
# ---------------------------------------------------------------------------
def _objective(centers, angles, outer_theta, outer_center):
    """Compute objective: outer side length + overlap penalty + containment penalty."""
    verts = [_hex_vertices(angles[k]) + centers[k] for k in range(N)]
    all_pts = np.vstack(verts)
    
    # Bounding hexagon side
    axes = _hex_axes(outer_theta)
    P = all_pts @ axes.T
    c = np.array(outer_center) @ axes.T
    max_dev = np.max(P.max(axis=0) - c)
    side = 2.0 * max_dev / math.sqrt(3.0)
    
    # Overlap penalty
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            if _sat_overlap(verts[i], verts[j]):
                # Compute minimum separation depth
                axesA = _hex_axes_from_verts(verts[i])
                axesB = _hex_axes_from_verts(verts[j])
                all_axes = np.vstack([axesA, axesB])
                projA = _project_polygon(verts[i], all_axes)
                projB = _project_polygon(verts[j], all_axes)
                min_sep = float('inf')
                for k in range(len(all_axes)):
                    sep = min(projA[k, 1] - projB[k, 0], projB[k, 1] - projA[k, 0])
                    min_sep = min(min_sep, sep)
                if min_sep < 0:
                    overlap_pen += 10.0 * min_sep * min_sep
    
    # Containment penalty
    contain_pen = 0.0
    for k in range(N):
        Pk = verts[k] @ axes.T
        for face in range(6):
            exceed = (Pk[:, face] - c[face]).max() - outer_side * math.sqrt(3.0) / 2.0
            if exceed > 0:
                contain_pen += 5.0 * exceed * exceed
    
    return side + overlap_pen + contain_pen

# ---------------------------------------------------------------------------
# Initial configuration generators
# ---------------------------------------------------------------------------
def _construct_initial_config(seed_type=0):
    """Generate initial configuration based on seed type."""
    if seed_type == 0:
        # Hexagonal cluster: center + 6 neighbors + 5 more
        d = 2.0 * math.sqrt(3.0) + 0.001
        a1 = np.array([d, 0.0])
        a2 = np.array([d / 2.0, d * math.sqrt(3.0) / 2.0])
        coords = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
                  (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2)]
        pts = []
        for (i, j) in coords:
            pts.append(i * a1 + j * a2)
        pts = np.array(pts)
        c = pts.mean(axis=0)
        pts = pts - c
        angles = np.zeros(N)
        return pts, angles
    
    elif seed_type == 1:
        # Triangular lattice
        d = 2.0 * math.sqrt(3.0) + 0.001
        a1 = np.array([d, 0.0])
        a2 = np.array([d / 2.0, d * math.sqrt(3.0) / 2.0])
        pts = []
        for i in range(-1, 3):
            for j in range(-1, 2):
                pts.append(i * a1 + j * a2)
        pts = np.array(pts)
        c = pts.mean(axis=0)
        pts = pts - c
        angles = np.zeros(N)
        return pts, angles
    
    elif seed_type == 2:
        # Random in disk
        pts = []
        for _ in range(N):
            r = random.uniform(0, 4.0)
            theta = random.uniform(0, 2 * math.pi)
            pts.append([r * math.cos(theta), r * math.sin(theta)])
        pts = np.array(pts)
        c = pts.mean(axis=0)
        pts = pts - c
        angles = np.array([random.uniform(-math.pi/6, math.pi/6) for _ in range(N)])
        return pts, angles
    
    else:
        # Ring + center
        pts = [np.array([0.0, 0.0])]
        for k in range(11):
            ang = 2 * math.pi * k / 11
            r = 2.5 * math.sqrt(3.0)
            pts.append([r * math.cos(ang), r * math.sin(ang)])
        pts = np.array(pts)
        c = pts.mean(axis=0)
        pts = pts - c
        angles = np.zeros(N)
        return pts, angles

# ---------------------------------------------------------------------------
# Nelder-Mead style local optimization
# ---------------------------------------------------------------------------
def _nelder_mead(centers, angles, outer_theta, outer_center, max_iters=500, step_size=0.5):
    """Nelder-Mead optimization on positions and angles."""
    centers = centers.copy()
    angles = angles.copy()
    
    # Initial simplex: perturb each hexagon's position
    best_obj = _objective(centers, angles, outer_theta, outer_center)
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        improved = False
        # Try moving each hexagon in small steps
        for i in range(N):
            for dx in [-step_size, step_size]:
                for dy in [-step_size, step_size]:
                    new_centers = centers.copy()
                    new_centers[i] += np.array([dx, dy])
                    new_obj = _objective(new_centers, angles, outer_theta, outer_center)
                    if new_obj < best_obj - 1e-10:
                        centers = new_centers
                        best_obj = new_obj
                        improved = True
            
            # Try rotating
            for dangle in [-step_size * 0.3, step_size * 0.3]:
                new_angles = angles.copy()
                new_angles[i] += dangle
                new_obj = _objective(centers, new_angles, outer_theta, outer_center)
                if new_obj < best_obj - 1e-10:
                    angles = new_angles
                    best_obj = new_obj
                    improved = True
        
        if not improved:
            # Reduce step size
            step_size *= 0.8
            if step_size < 0.001:
                break
    
    return centers, angles, best_obj

# ---------------------------------------------------------------------------
# Simulated annealing
# ---------------------------------------------------------------------------
def _simulated_annealing(centers, angles, outer_theta, outer_center, max_iters=5000):
    """Simulated annealing to find better packings."""
    best_centers = centers.copy()
    best_angles = angles.copy()
    best_obj = _objective(centers, angles, outer_theta, outer_center)
    
    current_obj = best_obj
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        # Propose move
        i = random.randint(0, N - 1)
        move_type = random.random()
        
        if move_type < 0.7:
            # Move center
            delta = np.array([random.gauss(0, T), random.gauss(0, T)])
            new_centers = centers.copy()
            new_centers[i] += delta
            new_angles = angles.copy()
        else:
            # Rotate
            new_centers = centers.copy()
            new_angles = angles.copy()
            new_angles[i] += random.gauss(0, T * 0.5)
        
        new_obj = _objective(new_centers, new_angles, outer_theta, outer_center)
        delta_E = new_obj - current_obj
        
        # Accept/reject
        if delta_E < 0 or random.random() < math.exp(-delta_E / max(T, 1e-10)):
            centers = new_centers
            angles = new_angles
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_centers = centers.copy()
                best_angles = angles.copy()
        
        T = max(T * cooling, T_min)
    
    return best_centers, best_angles, best_obj

# ---------------------------------------------------------------------------
# Main construct function
# ---------------------------------------------------------------------------
def construct():
    start = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_side = float('inf')
    best_centers = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    def _consider(centers, angles, side, center, angle):
        nonlocal best_side, best_centers, best_angles, best_center, best_angle
        if side < best_side:
            best_side = side
            best_centers = centers.copy()
            best_angles = angles.copy()
            best_center = center
            best_angle = angle
    
    # Try multiple seed types and outer angles
    seed_types = [0, 1, 2, 3]
    outer_angles = [0.0, math.pi / 6.0, math.pi / 3.0 * 0.5]
    
    for seed_type in seed_types:
        for oa in outer_angles:
            if time.time() > DEADLINE - 10:
                break
            
            centers, angles = _construct_initial_config(seed_type)
            
            # Nelder-Mead local optimization
            bc, ba, bobj = _nelder_mead(centers, angles, oa, (0.0, 0.0), max_iters=200, step_size=0.5)
            
            # Simulated annealing
            bc2, ba2, bobj2 = _simulated_annealing(bc, ba, oa, (0.0, 0.0), max_iters=2000)
            
            # Compute best outer hexagon
            all_pts = np.vstack([_hex_vertices(ba2[k]) + bc2[k] for k in range(N)])
            side, center, angle = _compute_best_outer(all_pts)
            
            # Final Nelder-Mead polish with the computed outer
            bc3, ba3, bobj3 = _nelder_mead(bc2, ba2, angle, center, max_iters=100, step_size=0.2)
            all_pts = np.vstack([_hex_vertices(ba3[k]) + bc3[k] for k in range(N)])
            side2, center2, angle2 = _compute_best_outer(all_pts)
            
            _consider(bc3, ba3, side2, center2, angle2)
    
    # Final refinement on best configuration
    if best_centers is not None:
        # Fine-tune outer angle
        all_pts = np.vstack([_hex_vertices(best_angles[k]) + best_centers[k] for k in range(N)])
        side, center, angle = _compute_best_outer(all_pts)
        
        # Final polish
        bc_final, ba_final, _ = _nelder_mead(best_centers, best_angles, angle, center, max_iters=50, step_size=0.1)
        all_pts = np.vstack([_hex_vertices(ba_final[k]) + bc_final[k] for k in range(N)])
        side_final, center_final, angle_final = _compute_best_outer(all_pts)
        
        _consider(bc_final, ba_final, side_final, center_final, angle_final)
    
    # Fallback
    if best_centers is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Format output
    inner = []
    for k in range(N):
        inner.append([float(best_centers[k][0]), float(best_centers[k][1]), float(math.degrees(best_angles[k]))])
    
    return inner, best_center, best_side, math.degrees(best_angle)
