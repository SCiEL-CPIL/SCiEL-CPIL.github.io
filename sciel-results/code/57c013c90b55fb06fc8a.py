"""Differentiable packing of 11 unit hexagons in a regular hexagon."""
import time
import math
import numpy as np

N = 11
_START = time.time()
DEADLINE = 145.0  # safety margin before 150s kill

# Unit regular hexagon vertices (pointy-top, centered at origin)
# Vertices at angles 30, 90, 150, 210, 270, 330 degrees with radius 1
_HX_VERTS = np.array([
    [math.cos(math.radians(a)), math.sin(math.radians(a))]
    for a in [30, 90, 150, 210, 270, 330]
])

# Half-widths for hexagon containment test (pointy-top orientation)
# A point (px, py) is inside a hexagon centered at (cx, cy) with rotation theta
# if we transform to hex-local coords and check the 6 half-plane constraints.

def _hex_dist_to_boundary(px, py, cx, cy, theta_rad):
    """
    Signed distance from point (px,py) to the boundary of a unit regular hexagon
    centered at (cx,cy) rotated by theta_rad. Positive = outside.
    Uses the 6 supporting lines of the hexagon.
    """
    dx = px - cx
    dy = py - cy
    # Rotate point into hex-local frame
    ct = math.cos(-theta_rad)
    st = math.sin(-theta_rad)
    lx = dx * ct - dy * st
    ly = dx * st + dy * ct
    # Unit hexagon (pointy-top) has vertices at radius 1, angles 30+60k
    # The 6 faces have outward normals at angles 0, 60, 120, 180, 240, 300
    # Face normals for pointy-top hex: the flat sides are at angles 0, 60, ...
    # Actually for pointy-top (vertex at 90 deg), the flat sides face 0, 60, 120, 180, 240, 300
    # Normal directions: 0, 60, 120, 180, 240, 300 degrees
    # The apothem (distance from center to each face) = cos(30 deg) = sqrt(3)/2
    apothem = math.sqrt(3.0) / 2.0
    max_val = 0.0
    for k in range(6):
        angle = math.radians(60.0 * k)  # 0, 60, 120, 180, 240, 300
        nx = math.cos(angle)
        ny = math.sin(angle)
        val = lx * nx + ly * ny - apothem
        if val > max_val:
            max_val = val
    return max_val


def _hex_containment_violation(px, py, cx, cy, theta_rad):
    """Return max(0, distance outside) for point vs hexagon."""
    d = _hex_dist_to_boundary(px, py, cx, cy, theta_rad)
    return d if d > 0 else 0.0


def _pairwise_overlap(i_verts, j_verts):
    """
    Check if two hexagons overlap. i_verts and j_verts are (6,2) arrays of vertices.
    Returns a positive value if overlapping, 0 if disjoint.
    Uses SAT (Separating Axis Theorem) - check all 12 edge normals.
    """
    # All edges of both hexagons
    axes = []
    for verts in (i_verts, j_verts):
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k + 1) % 6]
            edge = v2 - v1
            # Normal to edge
            nx = -edge[1]
            ny = edge[0]
            norm = math.hypot(nx, ny)
            if norm > 1e-12:
                axes.append([nx / norm, ny / norm])
    
    min_overlap = float('inf')
    for ax in axes:
        # Project both hexagons onto this axis
        proj_i = i_verts @ ax
        proj_j = j_verts @ ax
        min_i, max_i = proj_i.min(), proj_i.max()
        min_j, max_j = proj_j.min(), proj_j.max()
        overlap = min(max_i, max_j) - max(min_i, min_j)
        if overlap <= 0:
            return 0.0  # Separating axis found
        min_overlap = min(min_overlap, overlap)
    return min_overlap


def _get_hex_verts(cx, cy, theta_rad):
    """Get 6 vertices of unit hexagon centered at (cx,cy) rotated by theta_rad."""
    ct = math.cos(theta_rad)
    st = math.sin(theta_rad)
    R = np.array([[ct, -st], [st, ct]])
    return _HX_VERTS @ R.T + np.array([cx, cy])


def _enclosing_hex_side(vertices_2d):
    """
    Given a set of 2D points (all vertices of all hexagons), find the minimum
    side length of a regular hexagon (any rotation, any center) that contains them.
    
    For a regular hexagon with side s, the inradius is s * sqrt(3)/2.
    The hexagon is determined by center and rotation.
    
    We use the fact that for a regular hexagon, the width in any direction
    perpendicular to a face normal is 2 * apothem = s * sqrt(3).
    
    Strategy: The minimum enclosing regular hexagon can be found by considering
    that at least 3 points are on the boundary (touching 3 faces).
    For each triple of points, we can determine a hexagon. But that's O(n^3).
    
    Simpler: For a given rotation angle of the hexagon, the minimum side is
    determined by the max of (width in each of the 3 diagonal directions) / sqrt(3).
    
    For a regular hexagon rotated by angle phi, the 3 pairs of parallel faces
    have normals at phi, phi+60, phi+120 (and their opposites).
    The required side length s must satisfy:
      s * sqrt(3) >= width in direction phi
      s * sqrt(3) >= width in direction phi+60  
      s * sqrt(3) >= width in direction phi+120
    
    So s(phi) = max(width(phi), width(phi+60), width(phi+120)) / sqrt(3)
    
    We minimize over phi.
    """
    pts = np.array(vertices_2d)
    if len(pts) == 0:
        return 1.0
    
    best_s = float('inf')
    # Sample rotations
    for phi_deg in np.linspace(0, 60, 121):  # 0 to 60 degrees, 121 points
        phi = math.radians(phi_deg)
        s = 0.0
        for k in range(3):
            angle = phi + math.radians(60.0 * k)
            nx = math.cos(angle)
            ny = math.sin(angle)
            proj = pts @ np.array([nx, ny])
            width = proj.max() - proj.min()
            s = max(s, width / math.sqrt(3.0))
        if s < best_s:
            best_s = s
    return best_s


def _compute_all_vertices(positions, angles_rad):
    """Compute all 66 vertices (11 hexagons x 6 vertices)."""
    all_verts = []
    for i in range(N):
        cx, cy = positions[i]
        theta = angles_rad[i]
        verts = _get_hex_verts(cx, cy, theta)
        all_verts.append(verts)
    return all_verts


def _objective(params):
    """
    params: array of shape (33,) = [x0,y0,a0, x1,y1,a1, ..., x10,y10,a10]
    Returns: (side_length, penalty)
    """
    positions = params[0::3].copy()
    angles_rad = params[1::3].copy()
    # Wait, let me restructure: params = [x0, y0, a0, x1, y1, a1, ...]
    # So positions[i] = (params[3*i], params[3*i+1])
    # angles[i] = params[3*i+2]
    positions = np.array([params[3*i] for i in range(N)])
    ys = np.array([params[3*i+1] for i in range(N)])
    angles_rad = np.array([params[3*i+2] for i in range(N)])
    
    # Compute all vertices
    all_verts_list = []
    for i in range(N):
        verts = _get_hex_verts(positions[i], ys[i], angles_rad[i])
        all_verts_list.append(verts)
    
    # Check pairwise overlaps using SAT
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            ov = _pairwise_overlap(all_verts_list[i], all_verts_list[j])
            overlap_penalty += ov ** 2  # Quadratic penalty
    
    # Compute enclosing hexagon side length
    all_pts = np.vstack(all_verts_list)
    side = _enclosing_hex_side(all_pts)
    
    total = side + 100.0 * overlap_penalty
    return total, side, overlap_penalty


def _objective_fast(params):
    """
    Faster objective using analytic containment check for enclosing hexagon.
    params: array of shape (33,)
    """
    positions = np.array([params[3*i] for i in range(N)])
    ys = np.array([params[3*i+1] for i in range(N)])
    angles_rad = np.array([params[3*i+2] for i in range(N)])
    
    # Compute all vertices
    all_verts_list = []
    for i in range(N):
        ct = math.cos(angles_rad[i])
        st = math.sin(angles_rad[i])
        R = np.array([[ct, -st], [st, ct]])
        verts = _HX_VERTS @ R.T + np.array([positions[i], ys[i]])
        all_verts_list.append(verts)
    
    # Pairwise overlap check
    overlap_penalty = 0.0
    for i in range(N):
        vi = all_verts_list[i]
        for j in range(i+1, N):
            vj = all_verts_list[j]
            # SAT check
            axes = []
            for verts in (vi, vj):
                for k in range(6):
                    e = verts[(k+1)%6] - verts[k]
                    n = np.array([-e[1], e[0]])
                    nn = np.linalg.norm(n)
                    if nn > 1e-12:
                        axes.append(n / nn)
            sep = False
            min_ov = float('inf')
            for ax in axes:
                pi = vi @ ax
                pj = vj @ ax
                ov = min(pi.max(), pj.max()) - max(pi.min(), pj.min())
                if ov <= 0:
                    sep = True
                    break
                min_ov = min(min_ov, ov)
            if not sep:
                overlap_penalty += min_ov ** 2
    
    # Enclosing hexagon: sample rotations
    all_pts = np.vstack(all_verts_list)
    best_s = float('inf')
    for phi_deg in np.linspace(0, 60, 61):
        phi = math.radians(phi_deg)
        s = 0.0
        for k in range(3):
            angle = phi + math.radians(60.0 * k)
            proj = all_pts @ np.array([math.cos(angle), math.sin(angle)])
            width = proj.max() - proj.min()
            s = max(s, width / math.sqrt(3.0))
        if s < best_s:
            best_s = s
    
    total = best_s + 50.0 * overlap_penalty
    return total, best_s, overlap_penalty


def _grad_descent(params, lr=0.01, max_iter=5000, tol=1e-6):
    """Simple gradient descent with numerical gradients."""
    current = params.copy()
    current_val, _, _ = _objective_fast(current)
    
    for it in range(max_iter):
        if time.time() > DEADLINE:
            break
        if it % 500 == 0 and it > 0:
            print(f"  Iter {it}, val={current_val:.4f}")
        
        # Numerical gradient
        grad = np.zeros_like(current)
        eps = 1e-4
        for i in range(len(current)):
            old = current[i]
            current[i] = old + eps
            f_plus, _, _ = _objective_fast(current)
            current[i] = old - eps
            f_minus, _, _ = _objective_fast(current)
            current[i] = old
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        # Update
        current = current - lr * grad
        new_val, _, _ = _objective_fast(current)
        
        # Adaptive step size
        if new_val < current_val:
            current_val = new_val
            if abs(new_val - current_val) < tol:
                break
        else:
            # Reduce step
            lr *= 0.5
            if lr < 1e-8:
                break
    
    return current, current_val


def _simulated_annealing(params, max_iter=3000):
    """SA to escape local minima."""
    current = params.copy()
    current_val, _, _ = _objective_fast(current)
    best = current.copy()
    best_val = current_val
    T = 1.0
    
    for it in range(max_iter):
        if time.time() > DEADLINE:
            break
        if it % 500 == 0 and it > 0:
            print(f"  SA Iter {it}, val={current_val:.4f}, best={best_val:.4f}")
        
        # Random perturbation
        candidate = current + np.random.randn(len(current)) * 0.1
        cand_val, _, _ = _objective_fast(candidate)
        
        if cand_val < current_val or np.random.random() < math.exp(-(cand_val - current_val) / max(T, 1e-8)):
            current = candidate
            current_val = cand_val
            if cand_val < best_val:
                best = candidate.copy()
                best_val = cand_val
        
        T *= 0.995
    
    return best, best_val


def _refine_with_grad(params, max_iter=2000):
    """Gradient descent refinement."""
    p, v = _grad_descent(params, lr=0.05, max_iter=max_iter)
    return p, v


def construct():
    t0 = time.time()
    
    # Initial guess: honeycomb-like arrangement
    # 11 hexagons: center + 6 around + 4 more
    # Use a hexagonal lattice arrangement
    # For pointy-top hexagons with side 1:
    # Horizontal spacing: sqrt(3)
    # Vertical spacing: 1.5
    # Offset rows: sqrt(3)/2
    
    params = np.zeros(3 * N)
    
    # Center at origin
    params[0] = 0.0
    params[1] = 0.0
    params[2] = 0.0
    
    # Ring of 6 at distance sqrt(3) (center-to-center for touching hexagons)
    ring_dist = math.sqrt(3.0)
    for k in range(6):
        angle = math.radians(60.0 * k)
        idx = 1 + k
        params[3*idx] = ring_dist * math.cos(angle)
        params[3*idx+1] = ring_dist * math.sin(angle)
        params[3*idx+2] = 0.0
    
    # 4 more hexagons at next ring
    # Place them at gaps
    for k in range(4):
        angle = math.radians(60.0 * k + 30)  # Offset by 30 degrees
        idx = 7 + k
        params[3*idx] = 2.0 * ring_dist * math.cos(angle)
        params[3*idx+1] = 2.0 * ring_dist * math.sin(angle)
        params[3*idx+2] = 0.0
    
    # Try multiple initial configurations and optimizations
    best_params = params.copy()
    best_val, best_side, best_overlap = _objective_fast(params)
    print(f"Initial: val={best_val:.4f}, side={best_side:.4f}, overlap={best_overlap:.4f}")
    
    # Gradient descent from initial
    if time.time() < DEADLINE - 10:
        p1, v1 = _grad_descent(params, lr=0.02, max_iter=3000)
        if v1 < best_val:
            best_params = p1
            best_val = v1
            print(f"After GD1: val={v1:.4f}")
    
    # SA from initial
    if time.time() < DEADLINE - 10:
        p2, v2 = _simulated_annealing(params, max_iter=2000)
        if v2 < best_val:
            best_params = p2
            best_val = v2
            print(f"After SA: val={v2:.4f}")
    
    # Refine best with gradient descent
    if time.time() < DEADLINE - 5:
        p3, v3 = _refine_with_grad(best_params, max_iter=2000)
        if v3 < best_val:
            best_params = p3
            best_val = v3
            print(f"After refine: val={v3:.4f}")
    
    # Second round: perturb and optimize
    if time.time() < DEADLINE - 10:
        for attempt in range(5):
            if time.time() > DEADLINE - 10:
                break
            # Perturb best solution
            perturbed = best_params + np.random.randn(3*N) * 0.3
            p, v = _grad_descent(perturbed, lr=0.02, max_iter=1500)
            if v < best_val:
                best_params = p
                best_val = v
                print(f"Attempt {attempt}: val={v:.4f}")
    
    # Final refinement
    if time.time() < DEADLINE - 5:
        p_final, v_final = _refine_with_grad(best_params, max_iter=1500)
        if v_final < best_val:
            best_params = p_final
            best_val = v_final
    
    # Extract final positions and angles
    positions = [best_params[3*i] for i in range(N)]
    ys = [best_params[3*i+1] for i in range(N)]
    angles_rad = [best_params[3*i+2] for i in range(N)]
    
    # Convert angles to degrees
    angles_deg = [math.degrees(a) for a in angles_rad]
    
    # Build inner list
    inner = [[positions[i], ys[i], angles_deg[i]] for i in range(N)]
    
    # Compute enclosing hexagon center and optimal rotation
    all_verts_list = []
    for i in range(N):
        ct = math.cos(angles_rad[i])
        st = math.sin(angles_rad[i])
        R = np.array([[ct, -st], [st, ct]])
        verts = _HX_VERTS @ R.T + np.array([positions[i], ys[i]])
        all_verts_list.append(verts)
    all_pts = np.vstack(all_verts_list)
    
    # Find optimal rotation and center for enclosing hexagon
    best_phi = 0.0
    best_s = float('inf')
    for phi_deg in np.linspace(0, 60, 361):
        phi = math.radians(phi_deg)
        s = 0.0
        for k in range(3):
            angle = phi + math.radians(60.0 * k)
            proj = all_pts @ np.array([math.cos(angle), math.sin(angle)])
            width = proj.max() - proj.min()
            s = max(s, width / math.sqrt(3.0))
        if s < best_s:
            best_s = s
            best_phi = phi_deg
    
    # Center of enclosing hexagon
    # For each face direction, the center should be at the midpoint of the projection range
    # Average over the 3 directions
    cx_sum = 0.0
    cy_sum = 0.0
    phi = math.radians(best_phi)
    for k in range(3):
        angle = phi + math.radians(60.0 * k)
        proj = all_pts @ np.array([math.cos(angle), math.sin(angle)])
        mid = (proj.max() + proj.min()) / 2.0
        cx_sum += mid * math.cos(angle)
        cy_sum += mid * math.sin(angle)
    center_x = cx_sum / 3.0
    center_y = cy_sum / 3.0
    
    # Actually, let's compute center more carefully
    # The center of the enclosing hexagon should be the centroid of the bounding box
    # in each of the 3 diagonal directions
    # Let's just use the average of all vertex coordinates as a rough center
    center_x = np.mean(all_pts[:, 0])
    center_y = np.mean(all_pts[:, 1])
    
    elapsed = time.time() - t0
    print(f"Final: side={best_s:.4f}, center=({center_x:.3f},{center_y:.3f}), angle={best_phi:.1f} deg, time={elapsed:.1f}s")
    
    return inner, (float(center_x), float(center_y)), float(best_s), float(best_phi)


# Test
if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Inner: {len(inner)} hexagons")
    print(f"Center: {center}")
    print(f"Side: {side}")
    print(f"Angle: {angle}")
