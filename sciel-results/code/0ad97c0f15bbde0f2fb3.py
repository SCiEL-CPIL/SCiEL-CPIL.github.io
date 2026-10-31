"""Force-directed circle packing in the unit square with LP polish."""
import time
import numpy as np
from scipy.optimize import linprog

N = 26

# EVOLVE-BLOCK-START

def _soft_overlap_force(d, r_i, r_j, k_rep=500.0):
    """Repulsive force between two circles when overlapping (soft)."""
    # d: distance between centers; r_i + r_j: sum of radii
    # overlap = max(0, r_i + r_j - d)
    # Use smooth approximation: overlap_soft = 0.5*(s + sqrt(s^2 + eps)) - 0.5*eps where s = r_i+r_j-d
    s = r_i + r_j - d
    eps = 1e-6
    overlap_soft = 0.5 * (s + np.sqrt(s * s + eps)) - 0.5 * eps
    # Force magnitude proportional to overlap, direction along center line
    return k_rep * overlap_soft


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill

    # --- Initialization: random positions, small equal radii ---
    rng = np.random.default_rng(42)
    n = N
    x = rng.uniform(0.05, 0.95, size=(n, 2))
    r = np.full(n, 0.04)  # start with small equal radii

    # --- Force-directed simulation with velocity-Verlet ---
    v = np.zeros((n, 2))
    dt = 0.005
    damping = 0.92
    k_rep = 800.0       # repulsion between circles
    k_wall = 1200.0     # repulsion from walls
    k_grav = 50.0       # gentle gravity toward center to prevent drift
    max_iter = 80000

    best_value = 0.0
    best_x, best_r = x.copy(), r.copy()

    for it in range(max_iter):
        if time.time() > deadline:
            break

        # Compute forces
        F = np.zeros((n, 2))

        # Circle-circle repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[j, 0] - x[i, 0]
                dy = x[j, 1] - x[i, 1]
                d = np.sqrt(dx * dx + dy * dy)
                if d < 1e-12:
                    dx, dy = 1e-8, 1e-8
                    d = np.sqrt(dx * dx + dy * dy)
                f = _soft_overlap_force(d, r[i], r[j], k_rep)
                fx = f * dx / d
                fy = f * dy / d
                F[i, 0] -= fx
                F[i, 1] -= fy
                F[j, 0] += fx
                F[j, 1] += fy

        # Wall repulsion (soft)
        for i in range(n):
            # Left wall: x[i] - r[i] > 0
            s_left = x[i, 0] - r[i]
            if s_left < 0:
                F[i, 0] += k_wall * (-s_left)
            # Right wall: 1 - x[i] - r[i] > 0
            s_right = 1.0 - x[i, 0] - r[i]
            if s_right < 0:
                F[i, 0] -= k_wall * (-s_right)
            # Bottom wall
            s_bot = x[i, 1] - r[i]
            if s_bot < 0:
                F[i, 1] += k_wall * (-s_bot)
            # Top wall
            s_top = 1.0 - x[i, 1] - r[i]
            if s_top < 0:
                F[i, 1] -= k_wall * (-s_top)

        # Gentle centering force to keep circles in view
        for i in range(n):
            F[i, 0] += k_grav * (0.5 - x[i, 0]) * 0.1
            F[i, 1] += k_grav * (0.5 - x[i, 1]) * 0.1

        # Velocity-Verlet integration
        v = v * damping + F * dt
        x = x + v * dt

        # Hard constraint projection: keep circles inside [0,1]^2
        for i in range(n):
            x[i, 0] = np.clip(x[i, 0], r[i], 1.0 - r[i])
            x[i, 1] = np.clip(x[i, 1], r[i], 1.0 - r[i])

        # Adaptive radius growth: slowly increase radii to fill space
        # Growth rate decreases over time
        growth_rate = 0.00002 * (1.0 - 0.5 * it / max_iter)
        r = r + growth_rate
        # Cap radii to avoid overflow
        r = np.minimum(r, 0.45)

        # Track best value
        value = np.sum(r)
        if value > best_value:
            best_value = value
            best_x, best_r = x.copy(), r.copy()

    x, r = best_x, best_r

    # --- LP-based radius polish given final positions ---
    # For each circle i, radius is limited by:
    #   r_i <= x_i - 0,  r_i <= 1 - x_i,  r_i <= y_i - 0,  r_i <= 1 - y_i
    #   r_i + r_j <= ||p_i - p_j|| for all i < j
    # Maximize sum(r_i)
    # Variables: r_0, ..., r_{n-1}
    c = -np.ones(n)  # minimize -sum(r)

    # Inequality constraints: A_ub @ r <= b_ub
    A_rows = []
    b_rows = []

    # Wall constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(n):
        for wall_val in [x[i, 0], 1.0 - x[i, 0], x[i, 1], 1.0 - x[i, 1]]:
            row = np.zeros(n)
            row[i] = 1.0
            A_rows.append(row)
            b_rows.append(max(wall_val, 1e-9))

    # Pairwise: r_i + r_j <= dist_ij
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i, 0] - x[j, 0]) ** 2 + (x[i, 1] - x[j, 1]) ** 2)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_rows.append(row)
            b_rows.append(max(dist, 1e-9))

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    # Bounds: 0 <= r_i <= 0.45
    bounds = [(0.0, 0.45)] * n

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            r_lp = res.x
            # Verify feasibility and value
            val_lp = np.sum(r_lp)
            if val_lp > best_value:
                r = r_lp
                best_value = val_lp
    except Exception:
        pass

    # Final safety: ensure all circles are strictly inside and non-overlapping
    # Slight shrink to guarantee disjoint interiors
    r = r * 0.999
    # Re-clip positions to be safe
    for i in range(n):
        x[i, 0] = np.clip(x[i, 0], r[i] * 1.001, 1.0 - r[i] * 1.001)
        x[i, 1] = np.clip(x[i, 1], r[i] * 1.001, 1.0 - r[i] * 1.001)

    result = np.column_stack([x[:, 0], x[:, 1], r])
    return result

# EVOLVE-BLOCK-END
