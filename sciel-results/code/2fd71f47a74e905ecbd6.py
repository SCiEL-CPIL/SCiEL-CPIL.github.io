"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles.
Strategy: Position-Based Dynamics (PBD) with outward pressure + Gauss-Seidel
constraint projection, followed by SLSQP polish.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
DEADLINE = 145.0  # safety margin below 150s kill


def _valid_check(arr, tol=1e-7):
    """Check all circles are in-square and non-overlapping."""
    for i in range(len(arr)):
        x, y, r = arr[i]
        if r < 0 or x < r + tol or x > 1 - r - tol or y < r + tol or y > 1 - r - tol:
            return False
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            d = np.hypot(arr[i, 0] - arr[j, 0], arr[i, 1] - arr[j, 1])
            if d < arr[i, 2] + arr[j, 2] - tol:
                return False
    return True


def _max_radius(pos, r):
    """Maximal radius for each circle given current positions and radii."""
    n = len(r)
    mr = np.empty(n)
    for i in range(n):
        dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        for j in range(n):
            if i != j:
                dmin = min(dmin, np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]) - r[j])
        mr[i] = max(0.0, dmin)
    return mr


def _delaunay_edges(pos):
    """Return set of (i,j) pairs that are Delaunay neighbors."""
    try:
        tri = Delaunay(pos)
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = simplex[a], simplex[b]
                    edges.add((min(i, j), max(i, j)))
        return edges
    except Exception:
        return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _pbd_step(pos, rad, pressure=0.02, iters=30, rng=None):
    """
    One PBD iteration:
    1. Apply outward pressure (force pushing circles apart and toward walls)
    2. Project constraints (non-overlap, boundary) via Gauss-Seidel
    3. Update radii to max feasible
    """
    pos = pos.copy()
    rad = rad.copy()
    n = len(rad)
    
    # Mass proportional to radius squared (minimum 0.01 to avoid division by zero)
    mass = np.maximum(rad ** 2, 0.01)
    inv_mass = 1.0 / mass
    
    for it in range(iters):
        # --- Apply outward pressure ---
        # Pressure pushes each circle away from others and toward walls
        forces = np.zeros((n, 2))
        for i in range(n):
            # Wall pressure: push toward center if too close to wall
            # This is handled by constraint projection, so we add inter-circle repulsion
            for j in range(n):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.hypot(dx, dy)
                if d < 1e-8:
                    # Random tiny perturbation
                    dx = 0.001 * (2 * np.random.random() - 1)
                    dy = 0.001 * (2 * np.random.random() - 1)
                    d = np.hypot(dx, dy)
                # Repulsion force (inverse square, capped)
                if d < rad[i] + rad[j] + 0.05:
                    f = pressure / max(d, 0.01) ** 2
                    forces[i, 0] += f * dx / d
                    forces[i, 1] += f * dy / d
        
        # Update positions with force (velocity Verlet-like, but simple Euler)
        dt = 0.01
        for i in range(n):
            pos[i, 0] += dt * forces[i, 0] * inv_mass[i]
            pos[i, 1] += dt * forces[i, 1] * inv_mass[i]
        
        # --- Constraint projection: Gauss-Seidel ---
        for _ in range(3):  # Multiple passes for convergence
            # Boundary constraints
            for i in range(n):
                r = rad[i]
                if pos[i, 0] < r:
                    pos[i, 0] = r
                elif pos[i, 0] > 1 - r:
                    pos[i, 0] = 1 - r
                if pos[i, 1] < r:
                    pos[i, 1] = r
                elif pos[i, 1] > 1 - r:
                    pos[i, 1] = 1 - r
            
            # Non-overlap constraints
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    d = np.hypot(dx, dy)
                    min_d = rad[i] + rad[j]
                    if d < min_d and d > 1e-10:
                        # Distribute correction proportional to inverse mass
                        overlap = min_d - d
                        wi = inv_mass[i] / (inv_mass[i] + inv_mass[j])
                        wj = inv_mass[j] / (inv_mass[i] + inv_mass[j])
                        pos[i, 0] += wi * overlap * dx / d
                        pos[i, 1] += wi * overlap * dy / d
                        pos[j, 0] -= wj * overlap * dx / d
                        pos[j, 1] -= wj * overlap * dy / d
                    elif d <= 1e-10:
                        # Coincident: push apart randomly
                        angle = np.random.random() * 2 * np.pi
                        push = min_d / 2
                        pos[i, 0] += push * np.cos(angle)
                        pos[i, 1] += push * np.sin(angle)
                        pos[j, 0] -= push * np.cos(angle)
                        pos[j, 1] -= push * np.sin(angle)
    
    # Update radii to max feasible
    rad_new = _max_radius(pos, rad)
    # Keep radii non-decreasing where possible (growth bias)
    rad_new = np.maximum(rad_new, rad * 0.99)
    
    return pos, rad_new


def _slsqp_polish(pos, rad, maxiter=200):
    """Polish a near-valid configuration using SLSQP with hard constraints."""
    pos0 = pos.copy()
    rad0 = np.maximum(rad, 0.0)
    x0 = np.concatenate([pos0.reshape(-1), rad0])
    
    # Quick validity check and repair
    arr = np.column_stack([pos0, rad0])
    if not _valid_check(arr):
        rad0 = np.maximum(rad0 * 0.99, 0.0)
        x0 = np.concatenate([pos0.reshape(-1), rad0])
        arr = np.column_stack([pos0, rad0])
        if not _valid_check(arr):
            rad0 = np.maximum(rad0 * 0.95, 0.0)
            x0 = np.concatenate([pos0.reshape(-1), rad0])
            arr = np.column_stack([pos0, rad0])
            if not _valid_check(arr):
                return pos0, rad0

    edges = _delaunay_edges(pos0)
    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*N + i] - x[2*i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*N + i] - x[2*i+1]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N + i]})
    for (i, j) in edges:
        cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j:
                     np.hypot(x[2*i] - x[2*j], x[2*i+1] - x[2*j+1]) - x[2*N+i] - x[2*N+j]})
    return cons, edges


def _slsqp_polish_full(pos, rad, maxiter=200):
    """Full SLSQP polish."""
    pos0 = pos.copy()
    rad0 = np.maximum(rad, 0.0)
    x0 = np.concatenate([pos0.reshape(-1), rad0])
    
    arr = np.column_stack([pos0, rad0])
    if not _valid_check(arr):
        rad0 = np.maximum(rad0 * 0.99, 0.0)
        x0 = np.concatenate([pos0.reshape(-1), rad0])
        arr = np.column_stack([pos0, rad0])
        if not _valid_check(arr):
            rad0 = np.maximum(rad0 * 0.95, 0.0)
            x0 = np.concatenate([pos0.reshape(-1), rad0])
            arr = np.column_stack([pos0, rad0])
            if not _valid_check(arr):
                return pos0, rad0

    edges = _delaunay_edges(pos0)
    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*N + i] - x[2*i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*N + i] - x[2*i+1]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N + i]})
    for (i, j) in edges:
        cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j:
                     np.hypot(x[2*i] - x[2*j], x[2*i+1] - x[2*j+1]) - x[2*N+i] - x[2*N+j]})

    def obj(x):
        return -np.sum(x[2*N:])

    def jac(x):
        g = np.zeros_like(x)
        g[2*N:] = -1.0
        return g

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 0.5))

    try:
        res = minimize(obj, x0, jac=jac, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': maxiter, 'ftol': 1e-10})
        if res.success or np.sum(res.x[2*N:]) > np.sum(rad0):
            pos_new = res.x[:2*N].reshape(N, 2)
            rad_new = np.maximum(res.x[2*N:], 0.0)
            arr_new = np.column_stack([pos_new, rad_new])
            if _valid_check(arr_new):
                return pos_new, rad_new
    except Exception:
        pass
    return pos0, rad0


def construct():
    t_start = time.time()
    rng = np.random.default_rng(42)
    best = None
    best_score = -1.0

    def consider(pos, rad):
        nonlocal best, best_score
        arr = np.column_stack([pos, rad])
        if not _valid_check(arr):
            rad_fixed = np.maximum(rad * 0.999, 0)
            arr_fixed = np.column_stack([pos, rad_fixed])
            if not _valid_check(arr_fixed):
                rad_fixed = np.maximum(rad * 0.99, 0)
                arr_fixed = np.column_stack([pos, rad_fixed])
                if not _valid_check(arr_fixed):
                    return
            rad = rad_fixed
            arr = arr_fixed
        s = float(arr[:, 2].sum())
        if s > best_score:
            best_score = s
            best = arr

    # ---- Phase A: Generate diverse initializations ----
    starts = []

    # Grid initializations
    for side in range(4, 8):
        if time.time() > DEADLINE - 50:
            break
        cells = []
        for i in range(side):
            for j in range(side):
                cells.append(((i + 0.5) / side, (j + 0.5) / side))
        cells = cells[:N]
        pos = np.array(cells, dtype=float)
        rad = np.full(N, 0.5 / side - 1e-4)
        rad = _max_radius(pos, rad)
        starts.append((pos.copy(), rad.copy()))

        # Perturbed grids
        for k in range(2):
            if time.time() > DEADLINE - 50:
                break
            p0 = pos + 0.05 * (rng.random((N, 2)) * 2 - 1)
            p0 = np.clip(p0, 0.05, 0.95)
            r0 = np.full(N, 0.5 / side)
            r0 = _max_radius(p0, r0)
            starts.append((p0, r0))

    # Hexagonal packing initialization
    if time.time() <= DEADLINE - 50:
        hex_pos = []
        r_hex = 0.12
        dx = 2 * r_hex
        dy = r_hex * np.sqrt(3)
        for row in range(8):
            for col in range(8):
                x = col * dx + (row % 2) * dx / 2 + 0.15
                y = row * dy + 0.15
                if 0.05 < x < 0.95 and 0.05 < y < 0.95:
                    hex_pos.append((x, y))
        hex_pos = hex_pos[:N]
        if len(hex_pos) >= N:
            hp = np.array(hex_pos, dtype=float)
            hr = _max_radius(hp, np.full(N, 0.1))
            starts.append((hp, hr))

    # Random initializations
    for k in range(6):
        if time.time() > DEADLINE - 50:
            break
        p0 = rng.random((N, 2)) * 0.8 + 0.1
        r0 = np.full(N, 0.05)
        r0 = _max_radius(p0, r0)
        starts.append((p0, r0))

    # ---- Phase B: PBD relaxation from each start ----
    for idx, (pos0, rad0) in enumerate(starts):
        if time.time() > DEADLINE - 30:
            break
        budget = min(15.0, max(5.0, (DEADLINE - 30 - time.time()) / max(1, len(starts) - idx)))
        pos_pbd = pos0.copy()
        rad_pbd = rad0.copy()
        
        # Run PBD for the allocated budget
        t0 = time.time()
        while time.time() - t0 < budget:
            pos_pbd, rad_pbd = _pbd_step(pos_pbd, rad_pbd, pressure=0.02, iters=20, rng=rng)
        
        consider(pos_pbd, rad_pbd)

    # ---- Phase C: Polish the best solution found with SLSQP ----
    if best is not None:
        pos_b = best[:, :2].copy()
        rad_b = best[:, 2].copy()
        pos_p, rad_p = _slsqp_polish_full(pos_b, rad_b)
        consider(pos_p, rad_p)

    # ---- Phase D: Local search perturbation on best ----
    if best is not None and time.time() < DEADLINE - 15:
        for k in range(5):
            if time.time() > DEADLINE - 5:
                break
            pos_p = best[:, :2].copy()
            rad_p = best[:, 2].copy()
            n_perturb = rng.integers(3, 8)
            idx_perturb = rng.choice(N, size=n_perturb, replace=False)
            pos_p[idx_perturb] += 0.02 * (rng.random((n_perturb, 2)) * 2 - 1)
            pos_p = np.clip(pos_p, 0.05, 0.95)
            rad_p = _max_radius(pos_p, rad_p)
            
            # Short PBD relaxation
            t0 = time.time()
            while time.time() - t0 < 3.0:
                pos_p, rad_p = _pbd_step(pos_p, rad_p, pressure=0.02, iters=10, rng=rng)
            
            pos_f, rad_f = _slsqp_polish_full(pos_p, rad_p)
            consider(pos_f, rad_f)

    if best is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    return best


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
