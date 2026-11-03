"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

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

    def compute_radii_fast(centers):
        """Fast vectorized radius computation using full pairwise constraint."""
        centers = np.asarray(centers)
        n = len(centers)
        # Boundary
        radii = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                           np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
        # Pairwise: r_i + r_j <= d_ij  =>  r_i <= d_ij - r_j
        # Iterative relaxation (few passes)
        D = cdist(centers, centers)
        np.fill_diagonal(D, np.inf)
        for _ in range(50):
            # r_i <= min_j (D[i,j] - r_j)
            new_radii = radii.copy()
            for i in range(n):
                # d_ij - r_j for all j != i
                vals = D[i] - radii
                vals[i] = np.inf
                new_radii[i] = min(radii[i], np.min(vals))
            if np.allclose(new_radii, radii, atol=1e-12):
                radii = new_radii
                break
            radii = new_radii
        return np.maximum(0, radii)

    def compute_radii_vectorized(centers):
        """
        Vectorized computation of maximum non-overlapping radii from centers.
        Uses iterative relaxation for the coupled constraint r_i + r_j <= d_ij.
        """
        return compute_radii_fast(centers)

    def eval_packing(centers):
        """Evaluate the sum of radii for given centers."""
        radii = compute_radii_vectorized(centers)
        return np.sum(radii), radii

    def contact_graph_newton(centers, radii, max_iters=50, tol=1e-10):
        """
        Newton-Raphson on contact constraints.
        Identifies active contacts (r_i + r_j ≈ d_ij or boundary contacts)
        and solves the system to maximize sum of radii.
        """
        n = len(centers)
        centers = centers.copy()
        radii = radii.copy()

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Build contact graph
            D = cdist(centers, centers)
            np.fill_diagonal(D, np.inf)

            # Find active contacts: r_i + r_j >= d_ij - tol
            contacts = []
            for i in range(n):
                for j in range(i+1, n):
                    if D[i, j] <= radii[i] + radii[j] + 1e-8:
                        contacts.append((i, j))

            # Boundary contacts
            boundary_contacts = []
            for i in range(n):
                if abs(centers[i, 0] - radii[i]) < 1e-6:
                    boundary_contacts.append((i, 'L'))
                if abs(1.0 - centers[i, 0] - radii[i]) < 1e-6:
                    boundary_contacts.append((i, 'R'))
                if abs(centers[i, 1] - radii[i]) < 1e-6:
                    boundary_contacts.append((i, 'B'))
                if abs(1.0 - centers[i, 1] - radii[i]) < 1e-6:
                    boundary_contacts.append((i, 'T'))

            # Solve: for each contact (i,j): (x_i-x_j)^2 + (y_i-y_j)^2 = (r_i+r_j)^2
            # and boundary contacts: x_i = r_i, etc.
            # We use a simple iterative approach: adjust radii to fill gaps

            # For each pair in contact, try to increase radii
            improved = False
            for _ in range(20):
                changed = False
                # Update radii based on current geometry
                new_radii = compute_radii_fast(centers)
                if np.max(np.abs(new_radii - radii)) > 1e-12:
                    radii = new_radii
                    changed = True
                if not changed:
                    break

            # Try to move centers to create more space
            # For each circle, compute gradient of sum(radii) w.r.t. position
            # and take a small step
            for i in range(n):
                # Compute how moving center i affects radii
                best_move = np.array([0.0, 0.0])
                best_gain = 0.0
                for dx in [-0.005, -0.002, 0.0, 0.002, 0.005]:
                    for dy in [-0.005, -0.002, 0.0, 0.002, 0.005]:
                        nc = centers.copy()
                        nc[i, 0] = np.clip(centers[i, 0] + dx, 0.01, 0.99)
                        nc[i, 1] = np.clip(centers[i, 1] + dy, 0.01, 0.99)
                        nr = compute_radii_fast(nc)
                        gain = np.sum(nr) - np.sum(radii)
                        if gain > best_gain + 1e-10:
                            best_gain = gain
                            best_move = np.array([dx, dy])
                if best_gain > 1e-10:
                    centers[i] += best_move
                    improved = True

            if not improved:
                break

        return centers, compute_radii_fast(centers)

    def delaunay_radii(centers):
        """Compute radii using Delaunay + full pairwise relaxation."""
        return compute_radii_fast(centers)

    def objective_smooth(centers_flat, beta=50.0):
        """
        Smooth objective using soft-min approximation.
        """
        centers = centers_flat.reshape(-1, 2)
        n = len(centers)

        total_sum = 0.0
        for i in range(n):
            constraints = np.array([centers[i, 0], 1.0 - centers[i, 0],
                                    centers[i, 1], 1.0 - centers[i, 1]])
            for j in range(n):
                if j != i:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    d = np.sqrt(dx*dx + dy*dy) / 2.0
                    constraints = np.append(constraints, d)

            c_min = np.min(constraints)
            log_sum = np.log(np.sum(np.exp(-beta * (constraints - c_min))))
            soft_min_val = c_min - np.log(log_sum) / beta
            total_sum += soft_min_val

        return -total_sum

    def gradient_polish(centers, max_iters=200, beta=50.0):
        """
        Final high-precision polish using L-BFGS-B with smooth objective,
        followed by contact-graph refinement.
        """
        try:
            result = minimize(
                lambda x: objective_smooth(x, beta=beta),
                centers.flatten(),
                method='L-BFGS-B',
                bounds=[(0.01, 0.99)] * (2 * N),
                options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            opt_centers = result.x.reshape(-1, 2)
            radii = compute_radii_vectorized(opt_centers)
            x_full = np.column_stack([opt_centers, radii])
            if is_valid(x_full, tol=1e-6):
                # Contact graph refinement
                opt_centers2, radii2 = contact_graph_newton(opt_centers, radii, max_iters=30)
                x_full2 = np.column_stack([opt_centers2, radii2])
                if is_valid(x_full2, tol=1e-6) and np.sum(radii2) > np.sum(radii):
                    return x_full2
                return x_full
        except Exception:
            pass
        return None

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
        Includes multi-point moves and contact-aware adjustments.
        """
        rng = np.random.RandomState(seed)
        current = centers.copy()
        current_score, _ = eval_packing(current)
        best = current.copy()
        best_score = current_score

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Occasionally move 2 circles simultaneously
            if rng.random() < 0.3:
                idx = rng.choice(N, size=2, replace=False)
                for i in idx:
                    dx = rng.uniform(-step_size, step_size)
                    dy = rng.uniform(-step_size, step_size)
                    current[i, 0] = np.clip(current[i, 0] + dx, 0.01, 0.99)
                    current[i, 1] = np.clip(current[i, 1] + dy, 0.01, 0.99)
            else:
                i = rng.randint(N)
                dx = rng.uniform(-step_size, step_size)
                dy = rng.uniform(-step_size, step_size)
                current[i, 0] = np.clip(current[i, 0] + dx, 0.01, 0.99)
                current[i, 1] = np.clip(current[i, 1] + dy, 0.01, 0.99)

            new_score, _ = eval_packing(current)

            if new_score > current_score + 1e-12:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()

        return best, best_score

    def grid_init():
        """Structured grid initialization with jitter for better packing."""
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        rng = np.random.RandomState(42)
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    x = (i + 0.5) / side + rng.uniform(-0.02, 0.02)
                    y = (j + 0.5) / side + rng.uniform(-0.02, 0.02)
                    centers.append([np.clip(x, 0.05, 0.95), np.clip(y, 0.05, 0.95)])
        return np.array(centers[:N])

    def anchor_circle_init(seed=0):
        """
        Anchor-circle initialization: place a few large anchor circles
        and fill remaining space with smaller circles.
        """
        rng = np.random.RandomState(seed)
        centers = []
        radii = []

        # Place 4 anchor circles in corners
        anchors = [
            [0.15, 0.15, 0.12],
            [0.85, 0.15, 0.12],
            [0.15, 0.85, 0.12],
            [0.85, 0.85, 0.12]
        ]
        for a in anchors:
            centers.append(a[:2])
            radii.append(a[2])

        # Fill remaining N-4 circles
        for _ in range(N - 4):
            # Find best position for new circle
            best_pos = None
            best_r = 0
            for _ in range(200):
                x = rng.uniform(0.1, 0.9)
                y = rng.uniform(0.1, 0.9)
                # Compute max radius at this position
                r = min(x, 1-x, y, 1-y)
                for k in range(len(centers)):
                    dx = x - centers[k][0]
                    dy = y - centers[k][1]
                    d = np.sqrt(dx*dx + dy*dy)
                    r = min(r, d - radii[k])
                if r > best_r:
                    best_r = r
                    best_pos = [x, y]

            if best_pos is not None and best_r > 0.01:
                centers.append(best_pos)
                radii.append(best_r)
            else:
                # Fallback random
                x = rng.uniform(0.1, 0.9)
                y = rng.uniform(0.1, 0.9)
                centers.append([x, y])
                radii.append(0.03)

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

    def contact_graph_optimize(centers, radii, max_iters=100):
        """
        Optimize using contact graph: identify which circles are in contact
        and solve the tangency equations via Newton's method.
        """
        n = len(centers)
        centers = centers.copy()
        radii = radii.copy()

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            D = cdist(centers, centers)
            np.fill_diagonal(D, np.inf)

            # Find contact pairs
            contacts = []
            for i in range(n):
                for j in range(i+1, n):
                    if D[i, j] <= radii[i] + radii[j] + 1e-6:
                        contacts.append((i, j))

            # For each non-contact pair, try to bring them closer
            # to enable larger radii
            improved = False
            for _ in range(10):
                # Recompute radii
                new_radii = compute_radii_fast(centers)
                if np.max(np.abs(new_radii - radii)) < 1e-12:
                    break
                radii = new_radii

            # Try small moves to improve packing
            for i in range(n):
                best_gain = 0
                best_dx, best_dy = 0, 0
                for dx in [-0.003, -0.001, 0.001, 0.003]:
                    for dy in [-0.003, -0.001, 0.001, 0.003]:
                        nc = centers.copy()
                        nc[i, 0] = np.clip(centers[i, 0] + dx, 0.01, 0.99)
                        nc[i, 1] = np.clip(centers[i, 1] + dy, 0.01, 0.99)
                        nr = compute_radii_fast(nc)
                        gain = np.sum(nr) - np.sum(radii)
                        if gain > best_gain + 1e-10:
                            best_gain = gain
                            best_dx, best_dy = dx, dy
                if best_gain > 1e-10:
                    centers[i, 0] += best_dx
                    centers[i, 1] += best_dy
                    improved = True

            if not improved:
                break

        return centers, compute_radii_fast(centers)

    def random_init(seed):
        rng = np.random.RandomState(seed)
        return rng.uniform(0.1, 0.9, size=(N, 2))

    def multi_start_contact_graph():
        """
        Multi-start Newton-Raphson on contact graph constraints.
        Try multiple initial configurations and solve the tangency system.
        """
        results = []
        rng = np.random.RandomState(123)

        for trial in range(8):
            if time.time() > deadline - 15:
                break

            # Generate diverse initial configurations
            if trial == 0:
                centers = grid_init()
            elif trial == 1:
                centers = hex_init()
            elif trial == 2:
                centers = anchor_circle_init(seed=trial)
            else:
                centers = rng.uniform(0.1, 0.9, size=(N, 2))

            radii = compute_radii_fast(centers)

            # Contact graph optimization
            centers_opt, radii_opt = contact_graph_optimize(centers, radii, max_iters=50)
            score = np.sum(radii_opt)
            results.append((score, centers_opt, radii_opt))

        if results:
            results.sort(key=lambda x: -x[0])
            return results[0]
        return None

    # Main optimization loop
    best_result = None
    best_score = -np.inf

    # Phase 0: Contact graph multi-start (fast, may find good local optima)
    cg_result = multi_start_contact_graph()
    if cg_result is not None:
        cg_score, cg_centers, cg_radii = cg_result
        cg_full = np.column_stack([cg_centers, cg_radii])
        if is_valid(cg_full, tol=1e-6):
            best_score = cg_score
            best_result = cg_full.copy()

    # Multiple initialization strategies
    inits = []
    inits.append(('grid', grid_init()))
    inits.append(('hex', hex_init()))
    inits.append(('anchor', anchor_circle_init(seed=0)))
    for s in [1, 7, 13, 23, 37, 53, 71, 89, 101, 131, 151, 173]:
        inits.append((f'rand_{s}', random_init(s)))

    # Run SA + local search + gradient polish from each init
    for init_name, init_centers in inits:
        if time.time() > deadline - 15:
            break

        # Phase 1: Simulated Annealing
        sa_iters = 12000 if time.time() < deadline - 40 else 6000
        sa_centers, sa_score = simulated_annealing(
            init_centers,
            max_iters=sa_iters,
            T_start=0.04,
            T_end=0.002,
            seed=hash(init_name) % 10000
        )

        # Phase 2: Local search polish
        ls_iters = 1200 if time.time() < deadline - 20 else 600
        ls_centers, ls_score = local_search(
            sa_centers,
            max_iters=ls_iters,
            step_size=0.008,
            seed=hash(init_name) % 10000 + 1
        )

        # Phase 3: Gradient polish with smooth objective + contact graph
        polished = gradient_polish(ls_centers, max_iters=120, beta=50.0)

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
    if best_result is not None and time.time() < deadline - 25:
        for trial in range(8):
            if time.time() > deadline - 8:
                break

            # Perturb best solution
            rng = np.random.RandomState(500 + trial)
            perturbed = best_result[:, :2].copy()
            perturbed += rng.normal(0, 0.015, size=perturbed.shape)
            perturbed = np.clip(perturbed, 0.01, 0.99)

            # SA from perturbed
            sa_iters = 6000 if time.time() < deadline - 20 else 3000
            sa_centers, sa_score = simulated_annealing(
                perturbed,
                max_iters=sa_iters,
                T_start=0.02,
                T_end=0.001,
                seed=1000 + trial
            )

            # Local search
            ls_iters = 600 if time.time() < deadline - 10 else 300
            ls_centers, ls_score = local_search(
                sa_centers,
                max_iters=ls_iters,
                step_size=0.005,
                seed=2000 + trial
            )

            # Gradient polish + contact graph
            polished = gradient_polish(ls_centers, max_iters=80, beta=50.0)

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
    if best_result is not None and time.time() < deadline - 12:
        for trial in range(4):
            if time.time() > deadline - 6:
                break

            # Slight perturbation
            rng = np.random.RandomState(3000 + trial)
            perturbed = best_result[:, :2].copy()
            perturbed += rng.normal(0, 0.008, size=perturbed.shape)
            perturbed = np.clip(perturbed, 0.01, 0.99)

            # Fine local search
            ls_iters = 800 if time.time() < deadline - 8 else 400
            ls_centers, ls_score = local_search(
                perturbed,
                max_iters=ls_iters,
                step_size=0.004,
                seed=4000 + trial
            )

            # Gradient polish + contact graph
            polished = gradient_polish(ls_centers, max_iters=80, beta=50.0)

            if polished is not None:
                p_score = np.sum(polished[:, 2])
                if p_score > best_score:
                    best_score = p_score
                    best_result = polished.copy()
            elif ls_score > best_score:
                best_score = ls_score
                radii = compute_radii_vectorized(ls_centers)
                best_result = np.column_stack([ls_centers, radii])

    # Final contact graph refinement on best result
    if best_result is not None and time.time() < deadline - 3:
        cg_centers, cg_radii = contact_graph_optimize(
            best_result[:, :2], best_result[:, 2], max_iters=30
        )
        cg_full = np.column_stack([cg_centers, cg_radii])
        if is_valid(cg_full, tol=1e-6) and np.sum(cg_radii) > best_score:
            best_result = cg_full
            best_score = np.sum(cg_radii)

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
