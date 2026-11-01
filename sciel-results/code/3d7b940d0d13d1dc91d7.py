"""Packing 200 equal circles in a unit square using model-and-optimize approach.
Combined strategy: LogSumExp energy with analytic gradient + SLSQP active-set polish.
"""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import scipy.spatial


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    if n < 2:
        return min_boundary
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    return min(min_pair_dist / 2.0, min_boundary)


def _lse_energy(params, beta=100.0):
    """
    LogSumExp-based energy with analytic gradient.
    Minimizes -soft_min(pairwise/2, boundary) using log-sum-exp.
    """
    c = params.reshape(N, 2)
    x = c[:, 0]
    y = c[:, 1]

    # Pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2 + 1e-30)
    np.fill_diagonal(d, 1.0)

    iu = np.triu_indices(N, k=1)
    d_pairs = d[iu]

    # Soft min of pairwise distances
    d_min_p = np.min(d_pairs)
    w_p = np.exp(-beta * (d_pairs - d_min_p))
    w_p /= w_p.sum()
    soft_min_p = d_min_p - np.log(w_p.sum()) / beta

    # Boundary distances
    d_left = 0.5 - x
    d_right = 0.5 + x
    d_down = 0.5 - y
    d_up = 0.5 + y
    d_bound = np.stack([d_left, d_right, d_down, d_up], axis=1)  # (N, 4)
    d_min_b = np.min(d_bound, axis=1)
    w_b = np.exp(-beta * (d_bound - d_min_b[:, None]))
    w_b /= w_b.sum(axis=1, keepdims=True)
    soft_min_b = d_min_b - np.log(w_b.sum(axis=1)).mean() / beta

    # Combined: soft min of soft_min_p/2 and soft_min_b
    r_p = soft_min_p / 2.0
    r_b = soft_min_b
    r_min = min(r_p, r_b)
    arr = np.array([r_p, r_b])
    soft_min_comb = r_min - np.log(np.sum(np.exp(-beta * (arr - r_min)))) / beta

    energy = -soft_min_comb

    # Gradient
    grad = np.zeros(2 * N)

    # Pairwise gradient
    i_idx, j_idx = iu
    vec_x = x[i_idx] - x[j_idx]
    vec_y = y[i_idx] - y[j_idx]
    inv_d = 1.0 / np.maximum(d_pairs, 1e-15)

    # d(soft_min_p)/d(d_k) = -w_p[k]
    # d(d_k)/d(c_i) = (c_i - c_j)/d_k
    # So d(soft_min_p)/d(c_i_x) = -sum_k w_p[k] * (x_i - x_j)/d_k
    # energy = -soft_min_comb, so d(energy)/d(c_i_x) = +sum_k w_p[k]/2 * (x_i - x_j)/d_k
    # (factor 1/2 from r_p = soft_min_p/2)

    grad_x_p = np.zeros(N)
    grad_y_p = np.zeros(N)
    contrib_x = w_p * inv_d * 0.5  # factor 1/2 for /2 in r_p
    contrib_y = w_p * inv_d * 0.5

    grad_x_p[i_idx] += contrib_x * vec_x
    grad_x_p[j_idx] -= contrib_x * vec_x
    grad_y_p[i_idx] += contrib_y * vec_y
    grad_y_p[j_idx] -= contrib_y * vec_y

    # Boundary gradient
    # d(soft_min_b)/d(c_ik) for the active boundary
    # soft_min_b = d_min_b - (1/beta)*log(sum exp(-beta*(d_bound - d_min_b)))
    # d(soft_min_b)/d(d_bound[i,k]) = -w_b[i,k]
    # d(d_left)/d(x_i) = -1, d(d_right)/d(x_i) = 1, etc.
    # energy = -soft_min_comb, so sign flips

    # The combined soft min has derivative:
    # d(soft_min_comb)/d(r_p) = exp(-beta*(r_p - r_min)) / sum(exp(-beta*(arr-r_min)))
    # d(soft_min_comb)/d(r_b) = exp(-beta*(r_b - r_min)) / sum(exp(-beta*(arr-r_min)))

    denom = np.sum(np.exp(-beta * (arr - r_min)))
    w_rp = np.exp(-beta * (r_p - r_min)) / denom
    w_rb = np.exp(-beta * (r_b - r_min)) / denom

    # d(energy)/d(c) = -d(soft_min_comb)/d(c)
    # = -(w_rp * d(r_p)/d(c) + w_rb * d(r_b)/d(c))

    # d(r_p)/d(c) = 0.5 * d(soft_min_p)/d(c) = -0.5 * sum_k w_p[k] * unit_k
    # So d(energy)/d(c) from pairs = w_rp * 0.5 * sum_k w_p[k] * unit_k
    # This is already computed above as grad_x_p, grad_y_p (which are d(energy)/d(c) from pairs with the 0.5)
    # Actually let me recompute: grad_x_p above = sum_k w_p[k]*0.5*inv_d*(x_i-x_j)
    # This equals 0.5 * d(soft_min_p)/d(x_i) with sign... let me be careful.
    # soft_min_p = d_min - (1/beta)log(sum exp(-beta(d-d_min)))
    # d(soft_min_p)/d(d_k) = -w_p[k]  (since w_p is normalized)
    # d(d_k)/d(x_i) = (x_i - x_j)/d_k for i, (x_j-x_i)/d_k = -(x_i-x_j)/d_k for j
    # So d(soft_min_p)/d(x_i) = -sum_{k involving i} w_p[k] * (x_i - x_j_k)/d_k
    # r_p = soft_min_p/2, so d(r_p)/d(x_i) = -0.5 * sum w_p[k] * unit
    # soft_min_comb = r_min - (1/beta)log(sum exp(-beta(arr-r_min)))
    # d(soft_min_comb)/d(r_p) = w_rp
    # d(soft_min_comb)/d(x_i) = w_rp * d(r_p)/d(x_i) + w_rb * d(r_b)/d(x_i)
    # energy = -soft_min_comb
    # d(energy)/d(x_i) = -w_rp * d(r_p)/d(x_i) - w_rb * d(r_b)/d(x_i)
    # = w_rp * 0.5 * sum w_p[k]*unit + w_rb * d(r_b)/d(x_i)

    # The grad_x_p I computed = sum_k w_p[k]*0.5*vec_x*inv_d for i_idx
    # = 0.5 * sum_k w_p[k] * (x_i - x_j)/d_k
    # = -d(r_p)/d(x_i)  (since d(r_p)/d(x_i) = -0.5*sum w_p*unit)
    # So d(energy)/d(x_i) from pairs = w_rp * grad_x_p[i]  ... wait
    # d(energy)/d(x_i) = -w_rp * d(r_p)/d(x_i) = -w_rp * (-0.5*sum w_p*unit) = w_rp * 0.5*sum w_p*unit
    # And grad_x_p[i] = 0.5*sum_{k:i involved} w_p[k]*(x_i-x_j)/d_k = 0.5*sum w_p*unit
    # So d(energy)/d(x_i) from pairs = w_rp * grad_x_p[i] ✓

    # For boundary:
    # d(r_b)/d(x_i) = d(soft_min_b)/d(x_i)
    # soft_min_b uses d_min_b which is per-point, but the mean makes it coupled.
    # Simplification: for each point i, d(soft_min_b_i)/d(x_i) = -sum_k w_b[i,k] * d(d_bound[i,k])/d(x_i)
    # d(r_b)/d(x_i) ≈ (1/N) * d(soft_min_b_i)/d(x_i)  (from the mean)
    # Actually the mean couples them, but for gradient purposes:
    # d(mean(soft_min_b_i))/d(x_i) = (1/N) * d(soft_min_b_i)/d(x_i)
    # d(soft_min_b_i)/d(x_i) = -w_b[i,0]*(-1) - w_b[i,1]*(1) - w_b[i,2]*(0) - w_b[i,3]*(0) for x
    # = w_b[i,0] - w_b[i,1]
    # So d(r_b)/d(x_i) = (1/N)*(w_b[i,0] - w_b[i,1])
    # d(energy)/d(x_i) from bounds = -w_rb * d(r_b)/d(x_i) = -w_rb*(1/N)*(w_b[i,0]-w_b[i,1])

    grad_x_b = np.zeros(N)
    grad_y_b = np.zeros(N)
    grad_x_b = -(w_rb / N) * (w_b[:, 0] - w_b[:, 1])
    grad_y_b = -(w_rb / N) * (w_b[:, 2] - w_b[:, 3])

    grad[:N] = w_rp * grad_x_p + grad_x_b
    grad[N:] = w_rp * grad_y_p + grad_y_b

    return energy, grad


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0.0

    for d in np.linspace(0.052, 0.078, 40):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2

        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2 + 1e-9:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2 + 1e-9:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1

        if len(points) >= N:
            c = np.array(points[:N])
            c = np.clip(c, -0.4999, 0.4999)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()

    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N].copy()

    return best_c


def _local_optimize(c0, max_time=30.0, beta_schedule=None):
    """Local optimization using L-BFGS-B with LSE energy and analytic gradient."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)

    if beta_schedule is None:
        beta_schedule = [150.0, 100.0, 60.0, 40.0, 25.0, 15.0]

    current_c = c0.copy()
    bounds = [(-0.49999, 0.49999)] * (2 * N)

    for beta in beta_schedule:
        if time.time() > start_time + max_time - 3:
            break

        def fun_and_grad(params, _beta=beta):
            return _lse_energy(params, beta=_beta)

        try:
            result = minimize(
                fun_and_grad,
                current_c.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxfun': 10000, 'maxls': 100}
            )

            c_new = np.clip(result.x.reshape(N, 2), -0.49999, 0.49999)
            r_new = _compute_radius(c_new)

            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
            current_c = c_new
        except Exception:
            continue

    return best_c, best_r


def _slsqp_active_polish(c0, max_time=30.0):
    """SLSQP polish focusing on active constraints with radius as variable."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)

    for attempt in range(4):
        if time.time() > start_time + max_time - 3:
            break

        current_c = best_c.copy()
        current_r = best_r

        # Identify active constraints
        tol = 1e-7
        dists = pdist(current_c, metric='euclidean')
        dist_mat = squareform(dists)

        active_pairs = np.argwhere(dist_mat < 2 * current_r + tol)
        active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]

        bdists = np.minimum(
            np.minimum(0.5 - current_c[:, 0], 0.5 + current_c[:, 0]),
            np.minimum(0.5 - current_c[:, 1], 0.5 + current_c[:, 1])
        )
        active_bounds = np.where(bdists < current_r + tol)[0]

        if len(active_pairs) == 0 and len(active_bounds) == 0:
            break

        # Build constraints
        constraints = []
        for i, j in active_pairs:
            def _pair_con(x, _i=i, _j=j):
                ci = x[1 + 2*_i:1 + 2*_i+2]
                cj = x[1 + 2*_j:1 + 2*_j+2]
                return np.dot(ci - cj, ci - cj) - (2.0 * x[0])**2
            constraints.append({'type': 'ineq', 'fun': _pair_con})

        for i in active_bounds:
            d_left = 0.5 - current_c[i, 0]
            d_right = current_c[i, 0] + 0.5
            d_down = 0.5 - current_c[i, 1]
            d_up = current_c[i, 1] + 0.5
            min_d = min(d_left, d_right, d_down, d_up)

            if d_left == min_d:
                def _cl(x, _i=i):
                    return 0.5 - x[1 + 2*_i] - x[0]
                constraints.append({'type': 'ineq', 'fun': _cl})
            elif d_right == min_d:
                def _cr(x, _i=i):
                    return x[1 + 2*_i] + 0.5 - x[0]
                constraints.append({'type': 'ineq', 'fun': _cr})
            elif d_down == min_d:
                def _cd(x, _i=i):
                    return 0.5 - x[1 + 2*_i + 1] - x[0]
                constraints.append({'type': 'ineq', 'fun': _cd})
            else:
                def _cu(x, _i=i):
                    return x[1 + 2*_i + 1] + 0.5 - x[0]
                constraints.append({'type': 'ineq', 'fun': _cu})

        def obj(x):
            return -x[0]

        def obj_grad(x):
            g = np.zeros(1 + 2 * N)
            g[0] = -1.0
            return g

        x0 = np.concatenate([[current_r], current_c.flatten()])

        try:
            result = minimize(
                obj,
                x0,
                method='SLSQP',
                jac=obj_grad,
                bounds=[(current_r * 0.99, 0.05)] + [(-0.49999, 0.49999)] * (2 * N),
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-14, 'disp': False}
            )

            t_new = result.x[0]
            c_new = np.clip(result.x[1:].reshape(N, 2), -0.49999, 0.49999)
            r_new = _compute_radius(c_new)

            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
            else:
                break
        except Exception:
            break

    return best_c, best_r


def _polish(c0, max_time=15.0):
    """High-precision polish with very low temperature LSE."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)

    beta_schedule = [30.0, 20.0, 12.0, 8.0]
    current_c = c0.copy()
    bounds = [(-0.49999, 0.49999)] * (2 * N)

    for beta in beta_schedule:
        if time.time() > start_time + max_time - 1:
            break

        def fun_and_grad(params, _beta=beta):
            return _lse_energy(params, beta=_beta)

        try:
            result = minimize(
                fun_and_grad,
                current_c.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxfun': 20000, 'maxls': 200}
            )

            c_new = np.clip(result.x.reshape(N, 2), -0.49999, 0.49999)
            r_new = _compute_radius(c_new)

            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
            current_c = c_new
        except Exception:
            continue

    return best_c, best_r


def _delaunay_refine(c0, max_time=20.0):
    """
    Delaunay-based refinement: identify the critical Delaunay triangles
    (those with smallest circumradius) and apply targeted gradient ascent
    to increase the minimum circumradius.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)

    current_c = c0.copy()
    for iteration in range(30):
        if time.time() > start_time + max_time - 2:
            break

        # Compute Delaunay triangulation
        try:
            tri = scipy.spatial.Delaunay(current_c)
        except Exception:
            break

        # Compute circumradius for each triangle
        triangles = tri.simplices  # (n_tri, 3)
        n_tri = len(triangles)

        circumradii = np.zeros(n_tri)
        for k in range(n_tri):
            a, b, c = current_c[triangles[k]]
            # Circumradius = abc / (4 * Area)
            ab = np.linalg.norm(b - a)
            bc = np.linalg.norm(c - b)
            ca = np.linalg.norm(a - c)
            area = 0.5 * abs(np.cross(b - a, c - a))
            if area < 1e-15:
                circumradii[k] = np.inf
            else:
                circumradii[k] = (ab * bc * ca) / (4.0 * area)

        min_circum = np.min(circumradii)
        # The minimum pairwise distance is related to Delaunay edges
        # But the critical constraint is the minimum circumradius of Delaunay triangles
        # and the boundary distance

        # Identify critical triangles (circumradius < best_r * 1.5)
        critical = np.where(circumradii < min_circum * 1.2)[0]

        if len(critical) == 0:
            break

        # Compute gradient: for each critical triangle, push vertices apart
        # to increase circumradius
        grad = np.zeros(2 * N)
        for k in critical:
            tri_idx = triangles[k]
            a, b, c = current_c[tri_idx]

            # Circumcenter
            d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
            if abs(d) < 1e-15:
                continue
            ux = ((a[0]**2 + a[1]**2) * (b[1] - c[1]) + (b[0]**2 + b[1]**2) * (c[1] - a[1]) + (c[0]**2 + c[1]**2) * (a[1] - b[1])) / d
            uy = ((a[0]**2 + a[1]**2) * (c[0] - b[0]) + (b[0]**2 + b[1]**2) * (a[0] - c[0]) + (c[0]**2 + c[1]**2) * (b[0] - a[0])) / d

            # Gradient of circumradius w.r.t. vertex positions
            # R = abc/(4A), dR/d(a) pushes a away from circumcenter
            # Simplified: push each vertex away from circumcenter
            for vi in tri_idx:
                diff = current_c[vi] - np.array([ux, uy])
                dist = np.linalg.norm(diff)
                if dist > 1e-12:
                    grad[2*vi:2*vi+2] += diff / dist

        # Also handle boundary constraints
        bdists = 0.5 - np.abs(current_c)
        min_bd = np.min(bdists)
        if min_bd < best_r * 1.05:
            critical_b = np.argwhere(bdists < min_bd * 1.1)
            for i, k in critical_b:
                if current_c[i, k] > 0:
                    grad[2*i + k] -= 1.0
                else:
                    grad[2*i + k] += 1.0

        # Apply small step
        step_size = 0.002
        c_new = current_c + step_size * grad.reshape(N, 2)
        c_new = np.clip(c_new, -0.49999, 0.49999)
        r_new = _compute_radius(c_new)

        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            current_c = c_new
        else:
            # Try smaller step
            c_new2 = current_c + step_size * 0.3 * grad.reshape(N, 2)
            c_new2 = np.clip(c_new2, -0.49999, 0.49999)
            r_new2 = _compute_radius(c_new2)
            if r_new2 > best_r:
                best_r = r_new2
                best_c = c_new2.copy()
                current_c = c_new2
            else:
                break

    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_radius = 0.0
    best_c = None

    # Step 1: Hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()

    # Step 2: Local optimization from hex
    if time.time() < deadline - 55:
        max_opt_time = min(35.0, deadline - time.time() - 20)
        c_opt, r_opt = _local_optimize(c_hex, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()

    # Step 3: Delaunay-based refinement
    if time.time() < deadline - 35:
        remaining = deadline - time.time() - 10
        if remaining > 10:
            c_delaunay, r_delaunay = _delaunay_refine(best_c, max_time=min(15.0, remaining))
            if r_delaunay > best_radius:
                best_radius = r_delaunay
                best_c = c_delaunay.copy()

    # Step 4: Random restarts
    np.random.seed(42)
    n_restarts = 2
    for trial in range(n_restarts):
        if time.time() > deadline - 30:
            break

        m = int(np.ceil(np.sqrt(N)))
        grid_x, grid_y = np.meshgrid(
            np.linspace(-0.45, 0.45, m),
            np.linspace(-0.45, 0.45, m)
        )
        grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        np.random.shuffle(grid)
        c_init = grid[:N].copy()
        c_init += np.random.normal(0, 0.01, size=(N, 2))
        c_init = np.clip(c_init, -0.49999, 0.49999)

        r_init = _compute_radius(c_init)
        if r_init > best_radius:
            best_radius = r_init
            best_c = c_init.copy()

        remaining = deadline - time.time() - 15
        if remaining > 5:
            max_opt_time = min(12.0, remaining)
            c_opt, r_opt = _local_optimize(c_init, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()

    # Step 5: SLSQP hard-constrained polish
    if time.time() < deadline - 20:
        remaining = deadline - time.time() - 5
        if remaining > 10:
            c_slsqp, r_slsqp = _slsqp_active_polish(best_c, max_time=min(25.0, remaining))
            if r_slsqp > best_radius:
                best_radius = r_slsqp
                best_c = c_slsqp.copy()

    # Step 6: High-precision polish
    if time.time() < deadline - 8:
        remaining = deadline - time.time() - 2
        if remaining > 5:
            c_pol, r_pol = _polish(best_c, max_time=min(12.0, remaining))
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()

    # Step 7: Final ultra-precise pass
    if time.time() < deadline - 3:
        remaining = deadline - time.time() - 1
        if remaining > 3:
            def fun_and_grad_final(params):
                return _lse_energy(params, beta=10.0)

            try:
                result = minimize(
                    fun_and_grad_final,
                    best_c.flatten(),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(-0.49999, 0.49999)] * (2 * N),
                    options={'maxiter': 10000, 'ftol': 1e-19, 'gtol': 1e-16, 'maxfun': 50000, 'maxls': 300}
                )
                c_final = np.clip(result.x.reshape(N, 2), -0.49999, 0.49999)
                r_final = _compute_radius(c_final)
                if r_final > best_radius:
                    best_radius = r_final
                    best_c = c_final
            except Exception:
                pass

    # Final validation
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)

    return best_c
# EVOLVE-BLOCK-END
