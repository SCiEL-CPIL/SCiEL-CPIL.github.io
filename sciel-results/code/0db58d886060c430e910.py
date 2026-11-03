"""Rewritten program for Heilbronn triangles: maximize min triangle area for 11 points."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2.0  # area of the equilateral triangle


def barycentric_to_cartesian(bary):
    """Convert barycentric coordinates (u, v, w) with u+v+w=1 to Cartesian.
    We use 2N variables: for each point, (u, v) where w = 1 - u - v.
    Constraint: u >= 0, v >= 0, u + v <= 1.
    """
    u = bary[0::2]
    v = bary[1::2]
    x = u + 0.5 * v
    y = (np.sqrt(3) / 2.0) * v
    return np.column_stack([x, y])


def cartesian_to_barycentric(pts):
    """Convert Cartesian points to barycentric coordinates (u, v) pairs."""
    x = pts[:, 0]
    y = pts[:, 1]
    v = 2.0 * y / np.sqrt(3)
    u = x - 0.5 * v
    return np.column_stack([u, v]).flatten()


def min_triangle_area_from_bary(bary):
    """Compute the minimum triangle area from barycentric coordinates."""
    pts = barycentric_to_cartesian(bary)
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area of triangle with vertices pts[i], pts[j], pts[k]
                ax = pts[j][0] - pts[i][0]
                ay = pts[j][1] - pts[i][1]
                bx = pts[k][0] - pts[i][0]
                by = pts[k][1] - pts[i][1]
                area = 0.5 * abs(ax * by - ay * bx)
                if area < min_area:
                    min_area = area
    return min_area


def negative_min_area(bary):
    """Objective for minimization: negative of min triangle area."""
    return -min_triangle_area_from_bary(bary)


def min_triangle_area_grad(bary):
    """Compute gradient of min triangle area with respect to barycentric coords.
    Returns (value, gradient) where value is negative min area."""
    pts = barycentric_to_cartesian(bary)
    n = len(pts)
    min_area = float('inf')
    min_idx = None
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax = pts[j][0] - pts[i][0]
                ay = pts[j][1] - pts[i][1]
                bx = pts[k][0] - pts[i][0]
                by = pts[k][1] - pts[i][1]
                area = 0.5 * abs(ax * by - ay * bx)
                if area < min_area:
                    min_area = area
                    min_idx = (i, j, k)
    
    # Compute gradient of the active triangle area
    i, j, k = min_idx
    # area = 0.5 * |cross(pj - pi, pk - pi)|
    # Let's compute gradient with respect to each point's barycentric coords
    
    # For a triangle with vertices pi, pj, pk:
    # area = 0.5 * |(pj - pi) x (pk - pi)|
    # = 0.5 * |(pjx - pix)(pky - piy) - (pjy - piy)(pkx - pix)|
    
    # Gradient with respect to pi:
    # d(area)/d(pi) = 0.5 * (-(pky - piy), (pkx - pix) - (pjx - pix), (pjy - piy))
    # Wait, let me be more careful.
    
    # Let A = (pj - pi) x (pk - pi) = (pjx-pix)(pky-piy) - (pjy-piy)(pkx-pix)
    # dA/dpix = -(pky-piy) - (pjy-piy) = -pky + piy - pjy + piy... 
    
    # Actually, let me just use the signed area formula.
    # signed_area = 0.5 * [(pjx-pix)(pky-piy) - (pjy-piy)(pkx-pix)]
    # d(signed_area)/dpi = 0.5 * [-(pky-piy) - (pjy-piy)] = 0.5 * (-pky + piy - pjy + piy)
    # Hmm, this is getting complicated. Let me use a simpler approach.
    
    # The area of triangle (pi, pj, pk) as a function of pi:
    # area(pi) = 0.5 * |(pj - pi) x (pk - pi)|
    # = 0.5 * |(pj x pk - pj x pi - pi x pk + pi x pi)|
    # = 0.5 * |pj x pk - pj x pi - pi x pk|  (since pi x pi = 0)
    # = 0.5 * |pj x pk - (pj - pk) x pi|
    
    # Let C = pj x pk, D = pj - pk
    # area = 0.5 * |C - D x pi|
    
    # d(area)/dpi = 0.5 * sign(C - D x pi) * (-D x d(pi)/dpi)
    # In 2D, for vectors a, b: a x b = ax*by - ay*bx
    # d/dpi of (D x pi) = D x d(pi)/dpi
    # If pi = (px, py), D = (Dx, Dy):
    # D x pi = Dx*py - Dy*px
    # d/dpx = -Dy, d/dpy = Dx
    
    # So d(area)/dpi = 0.5 * sign(C - D x pi) * (-(-Dy), -(Dx)) = 0.5 * sign(...) * (Dy, -Dx)
    
    # Similarly for pj and pk.
    
    # Let me just compute it directly for each point.
    pi, pj, pk = pts[i], pts[j], pts[k]
    
    # Signed area (2x the actual area)
    cross_val = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
    sign_val = np.sign(cross_val)
    
    # Gradient of signed_area (2x area) with respect to each point
    # d(signed_area)/dpi = (-(pk[1]-pi[1]) - (pj[1]-pi[1]), (pk[0]-pi[0]) - (pj[0]-pi[0]))
    # = (-pk[1] + pi[1] - pj[1] + pi[1], pk[0] - pi[0] - pj[0] + pi[0])
    # = (-pk[1] - pj[1] + 2*pi[1], pk[0] - pj[0])
    
    grad_signed = np.zeros((3, 2))
    # For pi:
    grad_signed[0] = np.array([-(pk[1] - pi[1]) - (pj[1] - pi[1]), 
                                (pk[0] - pi[0]) - (pj[0] - pi[0])])
    # For pj:
    grad_signed[1] = np.array([(pk[1] - pi[1]), -(pk[0] - pi[0])])
    # For pk:
    grad_signed[2] = np.array([-(pj[1] - pi[1]), (pj[0] - pi[0])])
    
    # Actual area = 0.5 * |signed_area|, so gradient = 0.5 * sign * grad_signed
    grad_area = 0.5 * sign_val * grad_signed
    
    # Now convert from Cartesian gradient to barycentric gradient
    # x = u + 0.5*v, y = (sqrt(3)/2)*v
    # d(x)/du = 1, d(x)/dv = 0.5
    # d(y)/du = 0, d(y)/dv = sqrt(3)/2
    
    # grad_bary[u] = grad_cart[x] * 1 + grad_cart[y] * 0 = grad_cart[x]
    # grad_bary[v] = grad_cart[x] * 0.5 + grad_cart[y] * (sqrt(3)/2)
    
    bary_grad = np.zeros(2 * N)
    for idx, point_idx in enumerate([i, j, k]):
        gx = grad_area[idx][0]
        gy = grad_area[idx][1]
        bary_grad[2 * point_idx] = gx  # du
        bary_grad[2 * point_idx + 1] = 0.5 * gx + (np.sqrt(3) / 2.0) * gy  # dv
    
    return -min_area, -bary_grad


def is_valid_bary(bary, eps=1e-9):
    """Check if barycentric coordinates are valid (inside or on triangle)."""
    u = bary[0::2]
    v = bary[1::2]
    w = 1.0 - u - v
    return np.all(u >= -eps) and np.all(v >= -eps) and np.all(w >= -eps)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_bary = None
    best_score = 0.0
    
    # Differential evolution for global search
    # Bounds for barycentric coords: u in [0, 1], v in [0, 1], with u+v <= 1
    # We'll use bounds [0, 1] for both and rely on the objective to penalize invalid points
    # Actually, let's use a penalty approach
    
    def objective_with_penalty(bary):
        u = bary[0::2]
        v = bary[1::2]
        w = 1.0 - u - v
        # Penalty if any coordinate is out of [0, 1]
        penalty = 0.0
        for val in np.concatenate([u, v, w]):
            if val < 0 or val > 1:
                penalty += 10.0 * abs(min(val, 1 - val))
        if penalty > 0:
            return 10.0 + penalty  # Large penalty
        return negative_min_area(bary)
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Run differential evolution
    try:
        de_result = differential_evolution(
            objective_with_penalty,
            bounds=bounds,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=False,
            init='sobol'
        )
        
        elapsed = time.time() - start_time
        if elapsed < deadline - 30:
            # Polish with L-BFGS-B
            result = minimize(
                negative_min_area,
                de_result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if is_valid_bary(result.x):
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_bary = result.x
            else:
                # Try with de_result.x directly
                if is_valid_bary(de_result.x):
                    score = -objective_with_penalty(de_result.x)
                    if score > best_score:
                        best_score = score
                        best_bary = de_result.x
    except Exception:
        pass
    
    # Multiple restarts with L-BFGS-B if time permits
    elapsed = time.time() - start_time
    rng = np.random.RandomState(123)
    
    while time.time() < deadline - 10:
        # Generate random initial point in barycentric coords
        # Sample uniformly in the triangle
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        
        # Create a random configuration
        bary_init = np.zeros(2 * N)
        for i in range(N):
            ui = rng.random()
            vi = rng.random()
            if ui + vi > 1:
                ui = 1 - ui
                vi = 1 - vi
            bary_init[2 * i] = ui
            bary_init[2 * i + 1] = vi
        
        try:
            result = minimize(
                negative_min_area,
                bary_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if is_valid_bary(result.x):
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_bary = result.x
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 15:
            break
    
    # If no valid solution found, use a simple fallback
    if best_bary is None:
        # Place points in a grid-like pattern inside the triangle
        pts = []
        # Use a structured placement
        rows = [1, 2, 3, 3, 2]  # 11 points
        for row_idx, count in enumerate(rows):
            for col in range(count):
                # Barycentric coordinates
                u = (col + 1) / (count + 1)
                v = (row_idx + 1) / (sum(rows) + 1)
                if u + v > 1:
                    # Adjust
                    u = 0.5 * u
                    v = 0.5 * v
                pts.append([u, v])
        best_bary = np.array(pts).flatten()
    
    # Convert to Cartesian
    points = barycentric_to_cartesian(best_bary)
    
    # Ensure points are inside the triangle
    for i in range(N):
        x, y = points[i]
        # Check if inside
        # Barycentric coords
        v_b = 2.0 * y / np.sqrt(3)
        u_b = x - 0.5 * v_b
        w_b = 1.0 - u_b - v_b
        if u_b < 0 or v_b < 0 or w_b < 0:
            # Project back into triangle
            # Simple: scale towards centroid
            cx = 0.5
            cy = np.sqrt(3) / 6.0
            points[i] = [cx + 0.9 * (x - cx), cy + 0.9 * (y - cy)]
    
    return points
