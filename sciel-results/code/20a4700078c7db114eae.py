import numpy as np
import time
from scipy.optimize import linprog

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    def initial_grid(n):
        """Create initial grid placement with slightly smaller radii for feasibility."""
        side = int(np.ceil(np.sqrt(n)))
        xs = np.zeros(n)
        ys = np.zeros(n)
        rs = np.zeros(n)
        for i in range(n):
            row = i // side
            col = i % side
            xs[i] = (col + 0.5) / side
            ys[i] = (row + 0.5) / side
            rs[i] = 0.5 / side - 1e-4
        return xs, ys, rs

    def check_validity(xs, ys, rs, tol=1e-9):
        """Check if circles are valid (inside square, non-overlapping)."""
        for i in range(N):
            if xs[i] < rs[i] - tol or xs[i] > 1.0 - rs[i] + tol:
                return False
            if ys[i] < rs[i] - tol or ys[i] > 1.0 - rs[i] + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[j] - xs[i]
                dy = ys[j] - ys[i]
                dist_sq = dx * dx + dy * dy
                min_dist = rs[i] + rs[j]
                if dist_sq < min_dist * min_dist - tol * (min_dist * min_dist):
                    return False
        return True

    def pbd_relax(xs, ys, rs, max_iter=100, damping=1.0):
        """
        Position-based dynamics relaxation: project onto boundary and non-overlap constraints.
        Returns (xs, ys, converged).
        """
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            moved = False

            # Boundary constraints
            for i in range(N):
                r = rs[i]
                if xs[i] < r:
                    xs[i] = r
                    moved = True
                elif xs[i] > 1.0 - r:
                    xs[i] = 1.0 - r
                    moved = True
                if ys[i] < r:
                    ys[i] = r
                    moved = True
                elif ys[i] > 1.0 - r:
                    ys[i] = 1.0 - r
                    moved = True

            # Pairwise non-overlap constraints
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[j] - xs[i]
                    dy = ys[j] - ys[i]
                    dist_sq = dx * dx + dy * dy
                    min_dist = rs[i] + rs[j]

                    if dist_sq < min_dist * min_dist:
                        dist = np.sqrt(dist_sq) if dist_sq > 1e-15 else 1e-8
                        overlap = min_dist - dist
                        if dist < 1e-9:
                            angle = np.random.rand() * 2 * np.pi
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist

                        push = overlap * 0.5 * 1.0
                        xs[i] -= dx * push * damping
                        ys[i] -= dy * push * damping
                        xs[j] += dx * push * damping
                        ys[j] += dy * push * damping
                        moved = True

            if not moved:
                return xs, ys, True

        return xs, ys, False

    def grow_and_relax(xs, ys, rs, growth_factor, max_iter=60):
        """
        Try to grow all radii by growth_factor, then relax.
        Returns (xs, ys, rs, success).
        """
        new_rs = rs * growth_factor

        # Check if any radius exceeds 0.5
        if np.max(new_rs) > 0.5:
            return xs, ys, rs, False

        rs_new = new_rs.copy()
        xs_new = xs.copy()
        ys_new = ys.copy()

        # Relax with new radii
        xs_new, ys_new, converged = pbd_relax(xs_new, ys_new, rs_new, max_iter)

        if converged:
            # Verify validity
            if check_validity(xs_new, ys_new, rs_new):
                return xs_new, ys_new, rs_new, True

        # Revert
        return xs, ys, rs, False

    def simulate_annealing(xs, ys, rs, time_limit=60.0, max_iter=100000):
        """
        Simulated annealing to explore configuration space.
        Jointly optimizes positions and radii.
        """
        sa_start = time.time()
        current_temp = 0.01
        cooling_rate = 0.999
        min_temp = 1e-6

        best_score = np.sum(rs)
        best_xs = xs.copy()
        best_ys = ys.copy()
        best_rs = rs.copy()

        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            if time.time() - sa_start > time_limit:
                break

            # Random perturbation
            new_xs = xs.copy()
            new_ys = ys.copy()
            new_rs = rs.copy()

            # Perturb a random circle's position
            idx = np.random.randint(N)
            new_xs[idx] += np.random.randn() * 0.01
            new_ys[idx] += np.random.randn() * 0.01

            # Perturb a random circle's radius
            if np.random.rand() < 0.3:
                r_idx = np.random.randint(N)
                new_rs[r_idx] *= (1.0 + np.random.randn() * 0.005)
                new_rs[r_idx] = np.clip(new_rs[r_idx], 1e-4, 0.5)

            # Relax the new configuration
            new_xs, new_ys, converged = pbd_relax(new_xs, new_ys, new_rs, max_iter=30)

            if not converged:
                # Reject if relaxation fails
                continue

            if not check_validity(new_xs, new_ys, new_rs):
                continue

            new_score = np.sum(new_rs)
            current_score = np.sum(rs)
            delta_score = new_score - current_score

            # Metropolis criterion
            if delta_score > 0 or np.random.rand() < np.exp(delta_score / current_temp):
                xs, ys, rs = new_xs, new_ys, new_rs

                if new_score > best_score:
                    best_score = new_score
                    best_xs = new_xs.copy()
                    best_ys = new_ys.copy()
                    best_rs = new_rs.copy()

            # Cool down
            current_temp *= cooling_rate
            if current_temp < min_temp:
                break

        return best_xs, best_ys, best_rs, best_score

    def local_refine(xs, ys, rs, time_limit=30.0):
        """
        Local refinement using greedy growth and relaxation.
        """
        refine_start = time.time()
        growth = 1.001
        min_growth = 1.0001

        best_score = np.sum(rs)
        best_xs = xs.copy()
        best_ys = ys.copy()
        best_rs = rs.copy()

        while time.time() < deadline - 5 and time.time() - refine_start < time_limit:
            new_xs, new_ys, new_rs, success = grow_and_relax(xs, ys, rs, growth, max_iter=40)

            if success:
                current_score = np.sum(new_rs)
                if current_score > best_score:
                    best_score = current_score
                    best_xs = new_xs.copy()
                    best_ys = new_ys.copy()
                    best_rs = new_rs.copy()
                xs, ys, rs = new_xs, new_ys, new_rs
                growth = min(1.003, growth * 1.005)
            else:
                growth *= 0.95
                if growth < min_growth:
                    break

        return best_xs, best_ys, best_rs, best_score

    # --- Main optimization loop ---

    # Phase 1: Initial grid setup
    xs, ys, rs = initial_grid(N)
    xs, ys, _ = pbd_relax(xs, ys, rs, max_iter=100)

    best_score = np.sum(rs)
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_rs = rs.copy()

    # Phase 2: Greedy growth
    growth_factor = 1.002
    min_growth = 1.0001
    growth_deadline = deadline - 80

    while time.time() < growth_deadline:
        xs, ys, rs, success = grow_and_relax(xs, ys, rs, growth_factor, max_iter=50)

        if success:
            current_score = np.sum(rs)
            if current_score > best_score:
                best_score = current_score
                best_xs = xs.copy()
                best_ys = ys.copy()
                best_rs = rs.copy()
            growth_factor = min(1.004, growth_factor * 1.01)
        else:
            growth_factor *= 0.93
            if growth_factor < min_growth:
                break

    # Phase 3: Simulated annealing to escape local minima
    sa_xs, sa_ys, sa_rs, sa_score = simulate_annealing(
        best_xs, best_ys, best_rs,
        time_limit=60.0,
        max_iter=50000
    )

    if sa_score > best_score:
        best_score = sa_score
        best_xs = sa_xs.copy()
        best_ys = sa_ys.copy()
        best_rs = sa_rs.copy()

    # Phase 4: Local refinement
    local_xs, local_ys, local_rs, local_score = local_refine(
        best_xs, best_ys, best_rs,
        time_limit=30.0
    )

    if local_score > best_score:
        best_score = local_score
        best_xs = local_xs.copy()
        best_ys = local_ys.copy()
        best_rs = local_rs.copy()

    # Final validation
    if not check_validity(best_xs, best_ys, best_rs):
        # Fallback: return a valid grid
        xs_f, ys_f, rs_f = initial_grid(N)
        xs_f, ys_f, _ = pbd_relax(xs_f, ys_f, rs_f, max_iter=200)
        best_xs, best_ys, best_rs = xs_f, ys_f, rs_f

    return np.column_stack((best_xs, best_ys, best_rs))
