"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time


def _min_triangle_area_vec(points):
    """Vectorized: compute min area of any triangle formed by the n points.
    points: (n, 2) array. Returns scalar min area."""
    n = len(points)
    # For each pair (i,j), compute cross product with all k
    # Use broadcasting: points[i], points[j], points[k]
    # Area = 0.5 * |cross(P_j - P_i, P_k - P_i)|
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            d = pj - pi  # (2,)
            # Cross product of d with (points[k] - pi) for all k
            # cross = d[0]*(y_k - y_i) - d[1]*(x_k - x_i)
            diff = points - pi  # (n, 2)
            cross = d[0] * diff[:, 1] - d[1] * diff[:, 0]  # (n,)
            areas = 0.5 * np.abs(cross)
            # Only consider k > j to avoid duplicates, but for min we can check all
            # Actually for min, we just need the global min, so check all k != i, j
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            if np.any(mask):
                local_min = np.min(areas[mask])
                if local_min < min_area:
                    min_area = local_min
    return min_area


def _bary_to_cart(bary):
    """Convert (n,3) barycentric coords to (n,2) cartesian in unit equilateral triangle.
    Vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    Point = a*A + b*B + c*C, where a+b+c=1"""
    a = bary[:, 0]
    b = bary[:, 1]
    c = bary[:, 2]
    x = b * 1.0 + c * 0.5
    y = c * (np.sqrt(3) / 2)
    return np.column_stack([x, y])


def _cart_to_bary(points):
    """Convert (n,2) cartesian to (n,3) barycentric."""
    x = points[:, 0]
    y = points[:, 1]
    c = y / (np.sqrt(3) / 2)
    b = x - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])


def _objective(bary):
    """Return negative min triangle area (for minimization). bary is (n,3)."""
    # Project to valid simplex
    bary = np.maximum(bary, 0)
    bary = bary / bary.sum(axis=1, keepdims=True)
    points = _bary_to_cart(bary)
    min_area = _min_triangle_area_vec(points)
    return -min_area


def _coordinate_descent(bary, max_iter=500, step=0.01):
    """Refine barycentric coordinates via coordinate descent on min area."""
    bary = bary.copy()
    for it in range(max_iter):
        improved = False
        for i in range(N):
            for coord in range(3):
                best_val = bary[i, coord]
                best_obj = _objective(bary)
                # Try small perturbations
                for delta in [step, -step, step*0.5, -step*0.5, step*0.1, -step*0.1]:
                    bary[i, coord] = best_val + delta
                    # Project
                    bary[i] = np.maximum(bary[i], 0)
                    s = bary[i].sum()
                    if s < 1e-10:
                        bary[i] = np.array([1/3, 1/3, 1/3])
                    else:
                        bary[i] /= s
                    obj = _objective(bary)
                    if obj < best_obj:
                        best_obj = obj
                        best_val = bary[i, coord]
                        improved = True
                bary[i, coord] = best_val
                # Re-project
                bary[i] = np.maximum(bary[i], 0)
                s = bary[i].sum()
                if s < 1e-10:
                    bary[i] = np.array([1/3, 1/3, 1/3])
                else:
                    bary[i] /= s
        if not improved:
            break
    return bary


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s

    # Initialize: distribute points in barycentric coords
    # Use a mix of corners, edge midpoints, and interior points
    bary_init = np.array([
        [0.1, 0.1, 0.8],
        [0.1, 0.8, 0.1],
        [0.8, 0.1, 0.1],
        [0.15, 0.15, 0.7],
        [0.15, 0.7, 0.15],
        [0.7, 0.15, 0.15],
        [0.3, 0.35, 0.35],
        [0.35, 0.3, 0.35],
        [0.35, 0.35, 0.3],
        [0.25, 0.5, 0.25],
        [0.5, 0.25, 0.25],
    ])
    # Normalize
    bary_init = np.maximum(bary_init, 0)
    bary_init = bary_init / bary_init.sum(axis=1, keepdims=True)

    best_bary = bary_init.copy()
    best_obj = _objective(best_bary)

    # Simulated Annealing
    rng = np.random.default_rng(42)
    T = 0.01
    T_min = 1e-7
    cooling = 0.995
    n_iter = 50000

    for it in range(n_iter):
        if time.time() > deadline - 30:
            break

        # Perturb a random point
        i = rng.integers(0, N)
        new_bary = best_bary.copy()
        # Add random perturbation to barycentric coords
        perturb = rng.normal(0, 0.05, size=3)
        new_bary[i] += perturb
        # Project to simplex
        new_bary[i] = np.maximum(new_bary[i], 0)
        s = new_bary[i].sum()
        if s < 1e-10:
            new_bary[i] = np.array([1/3, 1/3, 1/3])
        else:
            new_bary[i] /= s

        new_obj = _objective(new_bary)

        # Accept or reject
        if new_obj < best_obj:
            best_obj = new_obj
            best_bary = new_bary
        else:
            delta = new_obj - best_obj
            if delta < 0:
                accept = True
            else:
                accept = rng.random() < np.exp(-delta / T)
            if accept:
                best_bary = new_bary

        T = max(T_min, T * cooling)

    # Coordinate descent refinement
    if time.time() < deadline - 10:
        best_bary = _coordinate_descent(best_bary, max_iter=200, step=0.005)
        best_obj = _objective(best_bary)

    # Final check
    if time.time() < deadline - 5:
        # Try a few more CD rounds with smaller steps
        for step in [0.002, 0.001]:
            if time.time() > deadline - 5:
                break
            best_bary = _coordinate_descent(best_bary, max_iter=100, step=step)
            best_obj = _objective(best_bary)

    # Convert to cartesian
    points = _bary_to_cart(best_bary)

    # Verify all points are inside the triangle
    # In barycentric coords with a,b,c >= 0 and sum=1, they are always inside
    # But let's clamp for numerical safety
    for i in range(N):
        x, y = points[i]
        if y < 0:
            points[i, 1] = 0
        if y > np.sqrt(3)/2:
            points[i, 1] = np.sqrt(3)/2
        if x < 0:
            points[i, 0] = 0
        if x > 1:
            points[i, 0] = 1
        # Check left edge: y <= sqrt(3)*x
        if x < 0.5 and y > np.sqrt(3) * x:
            # Project onto left edge
            x_new = y / np.sqrt(3)
            points[i, 0] = x_new
        # Check right edge: y <= sqrt(3)*(1-x)
        if x >= 0.5 and y > np.sqrt(3) * (1 - x):
            x_new = 1 - y / np.sqrt(3)
            points[i, 0] = x_new

    return points
# EVOLVE-BLOCK-END
