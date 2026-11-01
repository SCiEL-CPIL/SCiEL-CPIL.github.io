"""Circle packing in unit square, n=32.

Combines:
- Structural idea of the current program: one coherent L-BFGS-B pipeline,
  hard validity check, explicit deadline guard with safety margin.
- Refinement machinery from the strongest inspiration: diverse seed
  generators, center-only push-apart pre-refine, and a center-only final
  polish that preserves the radius sum.
- Physics-based repulsive potential (Lennard-Jones style) for a smooth
  energy landscape that gradient methods can navigate effectively.
- Final high-precision polish with multiple L-BFGS-B passes at decreasing
  tolerances and local perturbation search to extract additional gains.
"""

N = 32

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time


def construct():
    start_time = time.time()
    deadline = start_time + 135  # 135s out of 150s, safety margin

    best_flat = None
    best_score = -np.inf

    # ------------------------------------------------------------------
    # Parameter layout: [x0..xN, y0..yN, r0..rN]
    # ------------------------------------------------------------------
    def unpack(params):
        n = N
        x = params[:n]
        y = params[n:2*n]
        r = params[2*n:]
        return x, y, r

    # ------------------------------------------------------------------
    # Soft repulsive potential (Lennard-Jones style) + boundary terms
    # ------------------------------------------------------------------
    def potential(params, sigma=0.18, eps=1.0):
        """Return (energy, grad).

        Energy = sum over pairs of eps*( (sigma/d)^12 - 2*(sigma/d)^6 )
               + boundary spring terms - sum(r).
        """
        n = N
        x, y, r = unpack(params)
        energy = 0.0
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)

        # --- Pairwise Lennard-Jones potential ---
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                d2 = dx * dx + dy * dy
                if d2 < 1e-20:
                    d2 = 1e-20
                d = np.sqrt(d2)
                sd = sigma / d
                sd6 = sd ** 6
                sd12 = sd6 * sd6
                e = eps * (sd12 - 2.0 * sd6)
                energy += e
                coeff = 12.0 * eps * sd6 * (sd6 - 1.0) / d
                gx[i] -= coeff * dx
                gx[j] += coeff * dx
                gy[i] -= coeff * dy
                gy[j] += coeff * dy

        # --- Linear reward for larger radii ---
        energy -= np.sum(r)
        gr[:] -= 1.0

        # --- Boundary spring terms (keep circles inside [0,1]^2) ---
        k_boundary = 50.0
        for i in range(n):
            viol_l = max(0.0, r[i] - x[i])
            energy += k_boundary * viol_l ** 2
            if viol_l > 0:
                gx[i] -= 2.0 * k_boundary * viol_l
                gr[i] += 2.0 * k_boundary * viol_l
            viol_r = max(0.0, x[i] + r[i] - 1.0)
            energy += k_boundary * viol_r ** 2
            if viol_r > 0:
                gx[i] += 2.0 * k_boundary * viol_r
                gr[i] += 2.0 * k_boundary * viol_r
            viol_b = max(0.0, r[i] - y[i])
            energy += k_boundary * viol_b ** 2
            if viol_b > 0:
                gy[i] -= 2.0 * k_boundary * viol_b
                gr[i] += 2.0 * k_boundary * viol_b
            viol_t = max(0.0, y[i] + r[i] - 1.0)
            energy += k_boundary * viol_t ** 2
            if viol_t > 0:
                gy[i] += 2.0 * k_boundary * viol_t
                gr[i] += 2.0 * k_boundary * viol_t

        grad = np.concatenate([gx, gy, gr])
        return energy, grad

    # ------------------------------------------------------------------
    # Objective for L-BFGS-B: potential energy
    # ------------------------------------------------------------------
    def objective(params):
        energy, _ = potential(params)
        return energy

    # ------------------------------------------------------------------
    # Hard validity check
    # ------------------------------------------------------------------
    def is_valid(params, tol=1e-9):
        n = N
        x, y, r = unpack(params)
        if np.any(r < -tol):
            return False
        if np.any(x - r < -tol) or np.any(1.0 - x - r < -tol):
            return False
        if np.any(y - r < -tol) or np.any(1.0 - y - r < -tol):
            return False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False
        return True

    # ------------------------------------------------------------------
    # Score: sum of radii
    # ------------------------------------------------------------------
    def get_score(params):
        _, _, r = unpack(params)
        return float(np.sum(r))

    # ------------------------------------------------------------------
    # Center-only push-apart: separate overlapping circles without
    # changing radii, preserving the radius sum.
    # ------------------------------------------------------------------
    def push_apart(params, n_iters=80):
        n = N
        x, y, r = unpack(params)
        x = x.copy()
        y = y.copy()

        for _ in range(n_iters):
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    d = np.sqrt(dx * dx + dy * dy)
                    min_d = r[i] + r[j]
                    if d < min_d:
                        if d > 1e-12:
                            overlap = min_d - d
                            wi = r[i] / min_d if min_d > 1e-15 else 0.5
                            wj = 1.0 - wi
                            px = (dx / d) * overlap
                            py = (dy / d) * overlap
                            x[i] -= px * wi
                            y[i] -= py * wi
                            x[j] += px * wj
                            y[j] += py * wj
                            moved = True
                        else:
                            angle = np.random.uniform(0, 2 * np.pi)
                            x[i] -= 0.01 * np.cos(angle)
                            y[i] -= 0.01 * np.sin(angle)
                            x[j] += 0.01 * np.cos(angle)
                            y[j] += 0.01 * np.sin(angle)
                            moved = True
            for i in range(n):
                if x[i] < r[i]:
                    x[i] = r[i]
                    moved = True
                if x[i] > 1.0 - r[i]:
                    x[i] = 1.0 - r[i]
                    moved = True
                if y[i] < r[i]:
                    y[i] = r[i]
                    moved = True
                if y[i] > 1.0 - r[i]:
                    y[i] = 1.0 - r[i]
                    moved = True
            if not moved:
                break

        return np.concatenate([x, y, r])

    # ------------------------------------------------------------------
    # Multi-pass high-precision polish with decreasing tolerances and
    # local perturbation search.
    # ------------------------------------------------------------------
    def high_precision_polish(params, max_time=30.0):
        """Multiple L-BFGS-B passes at decreasing tolerances, with
        local perturbation between passes to escape shallow local minima."""
        x = params.copy()
        best_x = x.copy()
        best_e, _ = potential(x)

        tolerances = [1e-12, 1e-14, 1e-15]
        max_iters_list = [500, 800, 1000]

        for pass_idx in range(len(tolerances)):
            if time.time() > deadline - 5:
                break

            # Local perturbation to escape local minima (except first pass)
            if pass_idx > 0:
                perturbation_scale = 0.005 / (pass_idx + 1)
                x_perturbed = x + np.random.normal(0, perturbation_scale, size=x.shape)
                # Keep radii non-negative
                x_perturbed[2*N:] = np.maximum(x_perturbed[2*N:], 0.01)
                e_pert, _ = potential(x_perturbed)
                if e_pert < best_e:
                    x = x_perturbed
                    best_e = e_pert
                    best_x = x.copy()

            try:
                res = scipy.optimize.minimize(
                    objective, x, method='L-BFGS-B',
                    jac=lambda p: potential(p)[1],
                    options={'maxiter': max_iters_list[pass_idx],
                             'ftol': tolerances[pass_idx],
                             'gtol': 1e-13}
                )
                if res.fun < best_e:
                    best_e = res.fun
                    best_x = res.x.copy()
                    x = res.x.copy()
            except Exception:
                pass

        return best_x

    # ------------------------------------------------------------------
    # Diverse initial configurations
    # ------------------------------------------------------------------
    def get_initial_config(restart, rng):
        n = N
        side = int(np.ceil(np.sqrt(n)))

        if restart % 10 == 0:
            # Regular grid
            x = np.zeros(n)
            y = np.zeros(n)
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                x[i] = (col + 0.5) / side
                y[i] = (row + 0.5) / side
                r[i] = 0.5 / side - 0.002
        elif restart % 10 == 1:
            # Perturbed grid
            x = np.zeros(n)
            y = np.zeros(n)
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                x[i] = (col + 0.5) / side
                y[i] = (row + 0.5) / side
                r[i] = 0.5 / side - 0.002
            x += rng.uniform(-0.02, 0.02, size=n)
            y += rng.uniform(-0.02, 0.02, size=n)
        elif restart % 10 == 2:
            # Hexagonal packing (6 columns)
            x = np.zeros(n)
            y = np.zeros(n)
            r = np.zeros(n)
            for i in range(n):
                row = i // 6
                col = i % 6
                x[i] = (col + 0.5) / 6 + (0.25 if row % 2 == 1 else 0.0)
                y[i] = (row + 0.5) / 6
                r[i] = 0.08
        elif restart % 10 == 3:
            # Random uniform
            x = rng.uniform(0.08, 0.92, size=n)
            y = rng.uniform(0.08, 0.92, size=n)
            r = rng.uniform(0.05, 0.10, size=n)
        elif restart % 10 == 4:
            # 8x4 grid
            x = np.zeros(n)
            y = np.zeros(n)
            r = np.zeros(n)
            for i in range(n):
                row = i // 8
                col = i % 8
                x[i] = (col + 0.5) / 8
                y[i] = (row + 0.5) / 4
                r[i] = 0.5 / 8 - 0.002
            x += rng.uniform(-0.01, 0.01, size=n)
            y += rng.uniform(-0.01, 0.01, size=n)
        elif restart % 10 == 5:
            # 4x8 grid
            x = np.zeros(n)
            y = np.zeros(n)
            r = np.zeros(n)
            for i in range(n):
                row = i // 4
                col = i % 4
                x[i] = (col + 0.5) / 4
                y[i] = (row + 0.5) / 8
                r[i] = 0.5 / 4 - 0.002
            x += rng.uniform(-0.01, 0.01, size=n)
            y += rng.uniform(-0.01, 0.01, size=n)
        elif restart % 10 == 6:
            # Hexagonal variant
            x = np.zeros(n)
            y = np.zeros(n)
            r = np.zeros(n)
            for i in range(n):
                row = i // 6
                col = i % 6
                x[i] = (col + 0.5) / 6 + (0.2 if row % 2 == 0 else 0.0)
                y[i] = (row + 0.5) / 6
                r[i] = 0.075
        elif restart % 10 == 7:
            # Clustered: 4 groups of 8
            x = np.zeros(n)
            y = np.zeros(n)
            r = np.zeros(n)
            centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
            for i in range(n):
                group = i // 8
                cx, cy = centers[group]
                sub = i % 8
                sx = (sub % 3) / 3.0 + 0.5 / 3.0
                sy = (sub // 3) / 3.0 + 0.5 / 3.0
                x[i] = cx + (sx - 0.5) * 0.4
                y[i] = cy + (sy - 0.5) * 0.4
                r[i] = 0.06
        elif restart % 10 == 8:
            # Random medium
            x = rng.uniform(0.08, 0.92, size=n)
            y = rng.uniform(0.08, 0.92, size=n)
            r = rng.uniform(0.06, 0.09, size=n)
        else:
            # Random small
            x = rng.uniform(0.05, 0.95, size=n)
            y = rng.uniform(0.05, 0.95, size=n)
            r = rng.uniform(0.03, 0.06, size=n)

        return np.concatenate([x, y, r])

    # ------------------------------------------------------------------
    # SLSQP-based refinement: maximize sum(r) subject to hard constraints
    # ------------------------------------------------------------------
    def slsqp_refine(params, max_time=20.0):
        """Use SLSQP to refine a valid starting point by maximizing sum(r)
        with hard inequality constraints for non-overlap and boundaries."""
        n = N
        x, y, r = unpack(params)
        z = np.concatenate([x, y, r])

        def obj(z):
            return -np.sum(z[2*n:])

        def cons(z):
            xi = z[:n]
            yi = z[n:2*n]
            ri = z[2*n:]
            c = []
            # Boundary: x_i >= r_i
            c.extend(xi - ri)
            # Boundary: 1 - x_i >= r_i
            c.extend(1.0 - xi - ri)
            # Boundary: y_i >= r_i
            c.extend(yi - ri)
            # Boundary: 1 - y_i >= r_i
            c.extend(1.0 - yi - ri)
            # Pairwise: dist^2 >= (r_i + r_j)^2
            for i in range(n):
                for j in range(i + 1, n):
                    dx = xi[i] - xi[j]
                    dy = yi[i] - yi[j]
                    d2 = dx * dx + dy * dy
                    rs = ri[i] + ri[j]
                    c.append(d2 - rs * rs)
            return np.array(c)

        bounds = []
        for _ in range(n):
            bounds.append((0.0, 1.0))
        for _ in range(n):
            bounds.append((0.0, 1.0))
        for _ in range(n):
            bounds.append((0.001, 0.5))

        try:
            res = scipy.optimize.minimize(
                obj, z, method='SLSQP',
                bounds=bounds,
                constraints=[{'type': 'ineq', 'fun': cons}],
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            if res.success and is_valid(res.x):
                return res.x
            else:
                return params
        except Exception:
            return params

    # ------------------------------------------------------------------
    # Main search loop
    # ------------------------------------------------------------------
    rng = np.random.RandomState(12345)

    n_restarts = 60
    for restart in range(n_restarts):
        if time.time() > deadline:
            break

        params = get_initial_config(restart, rng)
        params = push_apart(params, n_iters=60)

        # Quick validity + score check on the pushed-apart config
        if is_valid(params):
            s = get_score(params)
            if s > best_score:
                best_score = s
                best_flat = params.copy()

        # L-BFGS-B optimization on the potential energy
        try:
            res = scipy.optimize.minimize(
                objective, params, method='L-BFGS-B',
                jac=lambda p: potential(p)[1],
                options={'maxiter': 800, 'ftol': 1e-13, 'gtol': 1e-11}
            )
            x_opt = res.x
        except Exception:
            x_opt = params

        if is_valid(x_opt):
            s = get_score(x_opt)
            if s > best_score:
                best_score = s
                best_flat = x_opt.copy()
        else:
            # Try center-only push-apart to recover validity
            x_fixed = push_apart(x_opt, n_iters=40)
            if is_valid(x_fixed):
                s = get_score(x_fixed)
                if s > best_score:
                    best_score = s
                    best_flat = x_fixed.copy()

        # SLSQP refinement on the best candidate so far
        if best_flat is not None and time.time() < deadline - 15:
            refined = slsqp_refine(best_flat)
            if is_valid(refined):
                s = get_score(refined)
                if s > best_score:
                    best_score = s
                    best_flat = refined.copy()

    # ------------------------------------------------------------------
    # Final high-precision polish
    # ------------------------------------------------------------------
    if best_flat is not None and time.time() < deadline - 10:
        try:
            x_polished = high_precision_polish(best_flat)
            if is_valid(x_polished):
                s = get_score(x_polished)
                if s > best_score:
                    best_score = s
                    best_flat = x_polished.copy()
            else:
                # Fallback: center-only push-apart
                x_fixed = push_apart(x_polished, n_iters=50)
                if is_valid(x_fixed):
                    s = get_score(x_fixed)
                    if s > best_score:
                        best_score = s
                        best_flat = x_fixed.copy()
        except Exception:
            pass

        # Final SLSQP pass if time remains
        if time.time() < deadline - 5:
            try:
                refined = slsqp_refine(best_flat)
                if is_valid(refined):
                    s = get_score(refined)
                    if s > best_score:
                        best_score = s
                        best_flat = refined.copy()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Safety: if no valid solution was found, return a valid grid packing
    # ------------------------------------------------------------------
    if best_flat is None:
        side = 6
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side - 0.002
        best_flat = np.concatenate([x, y, r])

    # ------------------------------------------------------------------
    # Format output: (N, 3) array of (x, y, radius)
    # ------------------------------------------------------------------
    x, y, r = unpack(best_flat)
    result = np.zeros((N, 3))
    result[:, 0] = x
    result[:, 1] = y
    result[:, 2] = r

    return result

# EVOLVE-BLOCK-END
