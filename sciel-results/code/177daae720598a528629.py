"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit

    def is_valid(x, tol=1e-8):
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        for i in range(N):
            if xs[i] - rs[i] < -tol or xs[i] + rs[i] > 1 + tol:
                return False
            if ys[i] - rs[i] < -tol or ys[i] + rs[i] > 1 + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True

    def compute_radii_from_centers(centers):
        """
        Compute maximum non-overlapping radii from centers using
        iterative proportional scaling on pairwise constraints.
        """
        centers = np.asarray(centers)
        n = len(centers)

        d_left = centers[:, 0]
        d_right = 1.0 - centers[:, 0]
        d_bottom = centers[:, 1]
        d_top = 1.0 - centers[:, 1]
        radii = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))

        dx = centers[:, 0:1] - centers[:, 0:1].T
        dy = centers[:, 1:2] - centers[:, 1:2].T
        dists = np.sqrt(dx*dx + dy*dy)
        np.fill_diagonal(dists, np.inf)

        for _ in range(100):
            changed = False
            for i in range(n):
                for j in range(i+1, n):
                    d = dists[i, j]
                    s = radii[i] + radii[j]
                    if s > d and s > 0:
                        scale = d / s
                        radii[i] *= scale
                        radii[j] *= scale
                        changed = True
            if not changed:
                break

        return np.maximum(0, radii)

    def build_contact_graph(centers, radii, tol=1e-6):
        """
        Build contact graph: edges where circles are tangent (within tol).
        Returns list of (i, j) pairs that are in contact.
        """
        contacts = []
        n = len(centers)
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                if abs(d - (radii[i] + radii[j])) < tol:
                    contacts.append((i, j))
        
        # Also add boundary contacts
        boundary_contacts = []
        for i in range(n):
            if abs(centers[i, 0] - radii[i]) < tol:
                boundary_contacts.append((i, 'L'))
            if abs(1.0 - centers[i, 0] - radii[i]) < tol:
                boundary_contacts.append((i, 'R'))
            if abs(centers[i, 1] - radii[i]) < tol:
                boundary_contacts.append((i, 'B'))
            if abs(1.0 - centers[i, 1] - radii[i]) < tol:
                boundary_contacts.append((i, 'T'))
        
        return contacts, boundary_contacts

    def newton_solve_contact_system(centers, radii, max_iters=50, tol=1e-10):
        """
        Newton-Raphson solver for contact-constrained packing.
        We fix the contact graph and solve for positions/radii that
        exactly satisfy all tangency constraints.
        
        Variables: x_i, y_i, r_i for each circle i (3N variables)
        Constraints: 
          - For each contact (i,j): (x_i-x_j)^2 + (y_i-y_j)^2 = (r_i+r_j)^2
          - For each boundary contact (i, side): appropriate equation
          - Non-contact pairs: (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2 (inequality, handled separately)
        
        We use a damped Newton method on the equality constraints.
        """
        n = len(centers)
        contacts, boundary_contacts = build_contact_graph(centers, radii, tol=1e-5)
        
        if len(contacts) + len(boundary_contacts) < 2 * n:
            # Not enough constraints for a well-determined system
            # Use least-squares approach
            pass
        
        # Build residual function
        def residual(vars):
            """Return residual vector for contact constraints."""
            xs = vars[0:n]
            ys = vars[n:2*n]
            rs = vars[2*n:3*n]
            res = []
            
            for (i, j) in contacts:
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d_sq = dx*dx + dy*dy
                r_sum = rs[i] + rs[j]
                res.append(d_sq - r_sum*r_sum)
            
            for (i, side) in boundary_contacts:
                if side == 'L':
                    res.append(xs[i] - rs[i])
                elif side == 'R':
                    res.append(1.0 - xs[i] - rs[i])
                elif side == 'B':
                    res.append(ys[i] - rs[i])
                elif side == 'T':
                    res.append(1.0 - ys[i] - rs[i])
            
            return np.array(res)
        
        def jacobian(vars):
            """Return Jacobian of residual w.r.t. variables."""
            xs = vars[0:n]
            ys = vars[n:2*n]
            rs = vars[2*n:3*n]
            
            n_constraints = len(contacts) + len(boundary_contacts)
            J = np.zeros((n_constraints, 3*n))
            
            row = 0
            for (i, j) in contacts:
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                r_sum = rs[i] + rs[j]
                
                # d/dx_i of (dx^2 + dy^2 - r_sum^2) = 2*dx
                J[row, i] = 2*dx
                J[row, n+i] = 2*dy
                J[row, 2*n+i] = -2*r_sum
                
                # d/dx_j = -2*dx, etc.
                J[row, j] = -2*dx
                J[row, n+j] = -2*dy
                J[row, 2*n+j] = -2*r_sum
                
                row += 1
            
            for (i, side) in boundary_contacts:
                if side == 'L':
                    J[row, i] = 1.0
                    J[row, 2*n+i] = -1.0
                elif side == 'R':
                    J[row, i] = -1.0
                    J[row, 2*n+i] = -1.0
                elif side == 'B':
                    J[row, n+i] = 1.0
                    J[row, 2*n+i] = -1.0
                elif side == 'T':
                    J[row, n+i] = -1.0
                    J[row, 2*n+i] = -1.0
                row += 1
            
            return J
        
        # Initial guess
        x0 = np.concatenate([centers[:, 0], centers[:, 1], radii])
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            r = residual(x0)
            if np.max(np.abs(r)) < tol:
                break
                
            J = jacobian(x0)
            
            # Solve J * dx = -r using least squares (system may be under/over-determined)
            try:
                dx, residuals, rank, sv = np.linalg.lstsq(J, -r, rcond=None)
            except:
                break
            
            # Damped step
            alpha = 1.0
            for _ in range(20):
                new_x = x0 + alpha * dx
                new_r = residual(new_x)
                if np.linalg.norm(new_r) < np.linalg.norm(r):
                    break
                alpha *= 0.5
            
            x0 = x0 + alpha * dx
        
        new_centers = np.column_stack([x0[0:n], x0[n:2*n]])
        new_radii = x0[2*n:3*n]
        
        # Verify validity
        x_full = np.column_stack([new_centers, new_radii])
        if is_valid(x_full, tol=1e-6) and np.all(new_radii > 0):
            return new_centers, new_radii
        
        return None

    def compute_radii_vectorized(centers):
        """
        Vectorized computation of maximum non-overlapping radii from centers.
        Uses iterative proportional scaling on pairwise constraints.
        """
        centers = np.asarray(centers)
        n = len(centers)

        d_left = centers[:, 0]
        d_right = 1.0 - centers[:, 0]
        d_bottom = centers[:, 1]
        d_top = 1.0 - centers[:, 1]
        radii = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))

        dx = centers[:, 0:1] - centers[:, 0:1].T
        dy = centers[:, 1:2] - centers[:, 1:2].T
        dists = np.sqrt(dx*dx + dy*dy)
        np.fill_diagonal(dists, np.inf)

        for _ in range(100):
            changed = False
            for i in range(n):
                for j in range(i+1, n):
                    d = dists[i, j]
                    s = radii[i] + radii[j]
                    if s > d and s > 0:
                        scale = d / s
                        radii[i] *= scale
                        radii[j] *= scale
                        changed = True
            if not changed:
                break

        return np.maximum(0, radii)

    def eval_packing(centers):
        """Evaluate the sum of radii for given centers."""
        radii = compute_radii_vectorized(centers)
        return np.sum(radii), radii

    def contact_graph_optimization(centers, radii, max_newton_iters=30):
        """
        Optimize packing using contact graph constraints with Newton-Raphson.
        This reduces the problem to solving a system of nonlinear equations
        where all contacts are exactly satisfied.
        """
        result = newton_solve_contact_system(centers, radii, max_iters=max_newton_iters)
        if result is not None:
            new_centers, new_radii = result
            new_score = np.sum(new_radii)
            old_score = np.sum(radii)
            if new_score > old_score:
                return new_centers, new_radii
        return centers, radii

    def objective_smooth(centers_flat, beta=200.0):
        """
        Fully vectorized smooth objective using soft-min via log-sum-exp.
        For each circle i, radius_i = soft_min(boundary_dists_i, {d_ij/2 for j!=i})
        """
        centers = centers_flat.reshape(-1, 2)
        n = len(centers)

        d_left = centers[:, 0]
        d_right = 1.0 - centers[:, 0]
        d_bottom = centers[:, 1]
        d_top = 1.0 - centers[:, 1]

        dx = centers[:, 0:1] - centers[:, 0:1].T
        dy = centers[:, 1:2] - centers[:, 1:2].T
        dists = np.sqrt(dx*dx + dy*dy)
        np.fill_diagonal(dists, np.inf)
        half_dists = dists / 2.0

        constraints = np.empty((n, n + 3))
        constraints[:, 0] = d_left
        constraints[:, 1] = d_right
        constraints[:, 2] = d_bottom
        constraints[:, 3] = d_top
        constraints[:, 4:] = half_dists

        c_min = np.min(constraints, axis=1, keepdims=True)
        shifted = -beta * (constraints - c_min)
        shifted = np.clip(shifted, -500, 500)
        log_sum = np.log(np.sum(np.exp(shifted), axis=1))
        soft_min_vals = c_min.squeeze() - log_sum / beta

        return -np.sum(soft_min_vals)

    def objective_smooth_grad(centers_flat, beta=200.0):
        """
        Vectorized smooth objective AND gradient.
        """
        centers = centers_flat.reshape(-1, 2)
        n = len(centers)

        d_left = centers[:, 0]
        d_right = 1.0 - centers[:, 0]
        d_bottom = centers[:, 1]
        d_top = 1.0 - centers[:, 1]

        dx = centers[:, 0:1] - centers[:, 0:1].T
        dy = centers[:, 1:2] - centers[:, 1:2].T
        dists = np.sqrt(dx*dx + dy*dy)
        np.fill_diagonal(dists, np.inf)
        half_dists = dists / 2.0

        constraints = np.empty((n, n + 3))
        constraints[:, 0] = d_left
        constraints[:, 1] = d_right
        constraints[:, 2] = d_bottom
        constraints[:, 3] = d_top
        constraints[:, 4:] = half_dists

        c_min = np.min(constraints, axis=1, keepdims=True)
        shifted = -beta * (constraints - c_min)
        shifted = np.clip(shifted, -500, 500)
        exp_vals = np.exp(shifted)
        sum_exp = np.sum(exp_vals, axis=1, keepdims=True)
        weights = exp_vals / sum_exp

        soft_min_vals = c_min.squeeze() - np.log(sum_exp.squeeze()) / beta

        grad = np.zeros_like(centers)

        grad[:, 0] += -(weights[:, 0] - weights[:, 1])
        grad[:, 1] += -(weights[:, 2] - weights[:, 3])

        for i in range(n):
            for j in range(i+1, n):
                d = dists[i, j]
                if d < 1e-12:
                    continue
                w_ij = weights[i, j + 4]
                w_ji = weights[j, i + 4]

                factor_x = (centers[i, 0] - centers[j, 0]) / (2.0 * d)
                factor_y = (centers[i, 1] - centers[j, 1]) / (2.0 * d)

                grad[i, 0] += -w_ij * factor_x
                grad[i, 1] += -w_ij * factor_y
                grad[j, 0] += w_ji * factor_x
                grad[j, 1] += w_ji * factor_y

        return -np.sum(soft_min_vals), grad.flatten()

    def gradient_polish(centers, max_iters=300, beta=200.0):
        """
        Final high-precision polish using L-BFGS-B with smooth objective,
        followed by contact-graph Newton refinement.
        """
        try:
            result = minimize(
                lambda x: objective_smooth(x, beta=beta),
                centers.flatten(),
                method='L-BFGS-B',
                bounds=[(0.01, 0.99)] * (2 * N),
                options={'maxiter': max_iters, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            opt_centers = result.x.reshape(-1, 2)
            radii = compute_radii_vectorized(opt_centers)
            
            # Try contact-graph Newton refinement
            cg_centers, cg_radii = contact_graph_optimization(opt_centers, radii, max_newton_iters=30)
            if np.sum(cg_radii) > np.sum(radii):
                opt_centers, radii = cg_centers, cg_radii
            
            x_full = np.column_stack([opt_centers, radii])
            if is_valid(x_full, tol=1e-6):
                return x_full
        except Exception:
            pass
        return None

    def pairwise_gradient_polish(centers, max_iters=500, lr=0.001):
        """
        Direct subgradient polish on the nonsmooth objective.
        Uses the fact that at optimum, the active constraints determine the gradient.
        """
        current = centers.copy()
        current_score, current_radii = eval_packing(current)
        best = current.copy()
        best_score = current_score

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Compute active constraints for each circle
            n = len(current)
            radii = current_radii.copy()

            # Determine which constraints are active (within tolerance)
            d_left = current[:, 0]
            d_right = 1.0 - current[:, 0]
            d_bottom = current[:, 1]
            d_top = 1.0 - current[:, 1]

            # Subgradient: for each circle, the direction to increase radius
            # is determined by the active constraints
            grad = np.zeros_like(current)

            for i in range(n):
                # Boundary active constraints
                tol_act = 1e-7
                if abs(d_left[i] - radii[i]) < tol_act:
                    grad[i, 0] += -1.0  # moving right increases r
                if abs(d_right[i] - radii[i]) < tol_act:
                    grad[i, 0] += 1.0  # moving left increases r
                if abs(d_bottom[i] - radii[i]) < tol_act:
                    grad[i, 1] += -1.0
                if abs(d_top[i] - radii[i]) < tol_act:
                    grad[i, 1] += 1.0

                # Pairwise active constraints
                for j in range(i+1, n):
                    dx = current[i, 0] - current[j, 0]
                    dy = current[i, 1] - current[j, 1]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < 1e-12:
                        continue
                    if abs(radii[i] + radii[j] - d) < tol_act:
                        # Active: moving i away from j increases r_i
                        # and moving j away from i increases r_j
                        ux = dx / d
                        uy = dy / d
                        grad[i, 0] += ux
                        grad[i, 1] += uy
                        grad[j, 0] -= ux
                        grad[j, 1] -= uy

            # Normalize gradient
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-10:
                break

            # Adaptive step size
            step = lr / (1.0 + 0.1 * it)
            new_centers = current + step * grad / gnorm
            new_centers = np.clip(new_centers, 0.01, 0.99)

            new_score, new_radii = eval_packing(new_centers)

            if new_score > current_score:
                current = new_centers
                current_score = new_score
                current_radii = new_radii
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()
            else:
                # Reduce step size
                lr *= 0.9

        return best, best_score

    def simulated_annealing(centers, max_iters=50000, T_start=0.05, T_end=0.001, seed=0):
        """
        Simulated annealing to optimize circle center positions.
        """
        rng = np.random.RandomState(seed)
        current = centers.copy()
        current_score, _ = eval_packing(current)

        best = current.copy()
        best_score = current_score

        n_iters = max_iters
        for it in range(n_iters):
            if time.time() > deadline - 5:
                break

            # Cooling schedule: geometric
            T = T_start * (T_end / T_start) ** (it / max(1, n_iters - 1))

            # Pick a random circle to move
            i = rng.randint(N)
            # Adaptive step size
            step = 0.02 * (T / T_start) ** 0.5 + 0.002
            dx = rng.normal(0, step)
            dy = rng.normal(0, step)

            new_centers = current.copy()
            new_centers[i, 0] = np.clip(current[i, 0] + dx, 0.01, 0.99)
            new_centers[i, 1] = np.clip(current[i, 1] + dy, 0.01, 0.99)

            new_score, _ = eval_packing(new_centers)
            delta = new_score - current_score

            # Acceptance criterion
            if delta > 0 or rng.uniform() < np.exp(delta / max(T, 1e-10)):
                current = new_centers
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()

        return best, best_score

    def local_search(centers, max_iters=3000, step_size=0.01, seed=0):
        """
        Best-fit local search: only accept improvements.
        """
        rng = np.random.RandomState(seed)
        current = centers.copy()
        current_score, _ = eval_packing(current)
        best = current.copy()
        best_score = current_score

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            i = rng.randint(N)
            dx = rng.uniform(-step_size, step_size)
            dy = rng.uniform(-step_size, step_size)

            new_centers = current.copy()
            new_centers[i, 0] = np.clip(current[i, 0] + dx, 0.01, 0.99)
            new_centers[i, 1] = np.clip(current[i, 1] + dy, 0.01, 0.99)

            new_score, _ = eval_packing(new_centers)

            if new_score > current_score + 1e-12:
                current = new_centers
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()

        return best, best_score

    def grid_init():
        """Structured grid initialization."""
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([(i + 0.5) / side, (j + 0.5) / side])
        return np.array(centers[:N])

    def hex_init():
        """Hexagonal-like initialization."""
        centers = []
        for i in range(6):
            for j in range(5):
                if len(centers) < N:
                    x = 0.1 + i * 0.16 + (0.08 if j % 2 else 0)
                    y = 0.1 + j * 0.17
                    if x < 0.9 and y < 0.9:
                        centers.append([x, y])
        if len(centers) < N:
            rng = np.random.RandomState(42)
            while len(centers) < N:
                centers.append(list(rng.uniform(0.1, 0.9, 2)))
        return np.array(centers[:N])

    def random_init(seed):
        rng = np.random.RandomState(seed)
        return rng.uniform(0.1, 0.9, size=(N, 2))

    # Main optimization loop
    best_result = None
    best_score = -np.inf

    # Multiple initialization strategies
    inits = []
    inits.append(('grid', grid_init()))
    inits.append(('hex', hex_init()))
    for s in [1, 7, 13, 23, 37, 53, 71, 89, 101, 131, 151, 173]:
        inits.append((f'rand_{s}', random_init(s)))

    # Run SA + local search + gradient polish + contact-graph Newton from each init
    for init_name, init_centers in inits:
        if time.time() > deadline - 10:
            break

        # Phase 1: Simulated Annealing
        sa_centers, sa_score = simulated_annealing(
            init_centers,
            max_iters=20000,
            T_start=0.04,
            T_end=0.002,
            seed=hash(init_name) % 10000
        )

        # Phase 2: Local search polish
        ls_centers, ls_score = local_search(
            sa_centers,
            max_iters=2000,
            step_size=0.008,
            seed=hash(init_name) % 10000 + 1
        )

        # Phase 3: Gradient polish with smooth objective (includes contact-graph Newton)
        polished = gradient_polish(ls_centers, max_iters=200, beta=200.0)

        if polished is not None:
            p_score = np.sum(polished[:, 2])
            if p_score > best_score:
                best_score = p_score
                best_result = polished.copy()
        else:
            # Use local search result if polish failed
            if ls_score > best_score:
                best_score = ls_score
                radii = compute_radii_vectorized(ls_centers)
                best_result = np.column_stack([ls_centers, radii])

        # Also check SA result directly
        if sa_score > best_score:
            best_score = sa_score
            radii = compute_radii_vectorized(sa_centers)
            best_result = np.column_stack([sa_centers, radii])

    # If no valid result, fall back to grid
    if best_result is None:
        centers = grid_init()
        radii = compute_radii_vectorized(centers)
        best_result = np.column_stack([centers, radii])

    # Final refinement: multiple SA restarts from best solution
    if best_result is not None and time.time() < deadline - 20:
        for trial in range(15):
            if time.time() > deadline - 5:
                break

            # Perturb best solution
            rng = np.random.RandomState(500 + trial)
            perturbed = best_result[:, :2].copy()
            perturbed += rng.normal(0, 0.012, size=perturbed.shape)
            perturbed = np.clip(perturbed, 0.01, 0.99)

            # SA from perturbed
            sa_centers, sa_score = simulated_annealing(
                perturbed,
                max_iters=10000,
                T_start=0.02,
                T_end=0.001,
                seed=1000 + trial
            )

            # Local search
            ls_centers, ls_score = local_search(
                sa_centers,
                max_iters=1000,
                step_size=0.005,
                seed=2000 + trial
            )

            # Pairwise gradient polish (direct on nonsmooth)
            pg_centers, pg_score = pairwise_gradient_polish(ls_centers, max_iters=300, lr=0.002)
            if pg_score > ls_score:
                ls_centers, ls_score = pg_centers, pg_score

            # Smooth gradient polish (includes contact-graph Newton)
            polished = gradient_polish(ls_centers, max_iters=150, beta=200.0)

            if polished is not None:
                p_score = np.sum(polished[:, 2])
                if p_score > best_score:
                    best_score = p_score
                    best_result = polished.copy()
            elif ls_score > best_score:
                best_score = ls_score
                radii = compute_radii_vectorized(ls_centers)
                best_result = np.column_stack([ls_centers, radii])

    # Second pass: more local search from best result with finer steps
    if best_result is not None and time.time() < deadline - 10:
        for trial in range(8):
            if time.time() > deadline - 5:
                break

            # Slight perturbation
            rng = np.random.RandomState(3000 + trial)
            perturbed = best_result[:, :2].copy()
            perturbed += rng.normal(0, 0.006, size=perturbed.shape)
            perturbed = np.clip(perturbed, 0.01, 0.99)

            # Fine local search
            ls_centers, ls_score = local_search(
                perturbed,
                max_iters=1200,
                step_size=0.003,
                seed=4000 + trial
            )

            # Pairwise gradient polish
            pg_centers, pg_score = pairwise_gradient_polish(ls_centers, max_iters=400, lr=0.001)
            if pg_score > ls_score:
                ls_centers, ls_score = pg_centers, pg_score

            # Smooth gradient polish (includes contact-graph Newton)
            polished = gradient_polish(ls_centers, max_iters=150, beta=300.0)

            if polished is not None:
                p_score = np.sum(polished[:, 2])
                if p_score > best_score:
                    best_score = p_score
                    best_result = polished.copy()
            elif ls_score > best_score:
                best_score = ls_score
                radii = compute_radii_vectorized(ls_centers)
                best_result = np.column_stack([ls_centers, radii])

    # Third pass: dedicated contact-graph Newton refinement
    if best_result is not None and time.time() < deadline - 8:
        for trial in range(10):
            if time.time() > deadline - 4:
                break

            # Slightly perturb to escape potential local minima in contact graph
            rng = np.random.RandomState(5000 + trial)
            perturbed = best_result[:, :2].copy()
            perturbed += rng.normal(0, 0.003, size=perturbed.shape)
            perturbed = np.clip(perturbed, 0.01, 0.99)

            radii = compute_radii_vectorized(perturbed)

            # Apply contact-graph Newton with more iterations
            cg_centers, cg_radii = contact_graph_optimization(perturbed, radii, max_newton_iters=50)

            if np.sum(cg_radii) > best_score:
                x_full = np.column_stack([cg_centers, cg_radii])
                if is_valid(x_full, tol=1e-6):
                    best_score = np.sum(cg_radii)
                    best_result = x_full

    # Ensure final validity with small margin
    if best_result is not None:
        # Apply tiny margin to ensure strict validity
        radii = compute_radii_vectorized(best_result[:, :2])
        radii = np.maximum(0, radii - 1e-9)
        best_result = np.column_stack([best_result[:, :2], radii])

        if not is_valid(best_result, tol=1e-6):
            # Fallback: conservative grid
            centers = grid_init()
            radii = compute_radii_vectorized(centers)
            radii = np.maximum(0, radii - 0.001)
            best_result = np.column_stack([centers, radii])

    return best_result

# EVOLVE-BLOCK-END
