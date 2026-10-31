"""Rewritten circle packing program with soft-penalty L-BFGS-B + radius growth + SLSQP polish."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    def elapsed():
        return time.time() - start_time

    def time_left():
        return deadline - elapsed()

    # --- Soft-max / soft-min helpers ---
    def _soft_max(vals, beta=50.0):
        """Soft max via log-sum-exp."""
        m = np.max(vals)
        if m < -1e10:
            return m
        return m + np.log(np.sum(np.exp(vals - m)) / len(vals))

    def _soft_min(vals, beta=50.0):
        return -_soft_max(-vals, beta)

    # --- Penalty function: smooth surrogate for constraints ---
    def _penalty_function(v, beta=50.0):
        """Returns penalty value (lower is better) and gradient."""
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        penalty = 0.0
        grad = np.zeros(3 * N)

        # Boundary violations: x_i < r_i, x_i > 1-r_i, y_i < r_i, y_i > 1-r_i
        # Use soft-max of violation magnitudes
        viol = np.concatenate([
            r - x,           # x < r
            x + r - 1.0,     # x > 1-r
            r - y,           # y < r
            y + r - 1.0      # y > 1-r
        ])
        sm = _soft_max(viol, beta)
        if sm > 0:
            penalty += sm * sm
            # Gradient of soft_max
            exps = np.exp(viol - sm)
            weights = exps / np.sum(exps)
            # d(sm)/d(viol_k) = weights_k
            # d(penalty)/d(viol_k) = 2*sm*weights_k
            g_viol = 2.0 * sm * weights
            # Map back to v
            grad[0:N] += g_viol[0:N] * (-1.0)       # x < r: viol = r - x, d/dx = -1
            grad[0:N] += g_viol[N:2*N] * (1.0)       # x > 1-r: viol = x+r-1, d/dx = 1
            grad[0:N] += 0.0
            grad[0:N] += g_viol[2*N:3*N] * (1.0)     # wait, let me redo this carefully

        # Let me redo boundary penalties more carefully
        penalty = 0.0
        grad = np.zeros(3 * N)

        # For each circle i, 4 boundary violations:
        # b1_i = r_i - x_i  (want <= 0)
        # b2_i = x_i + r_i - 1  (want <= 0)
        # b3_i = r_i - y_i  (want <= 0)
        # b4_i = y_i + r_i - 1  (want <= 0)
        # Collect all boundary violations
        b = np.empty(4 * N)
        b[0:N] = r - x
        b[N:2*N] = x + r - 1.0
        b[2*N:3*N] = r - y
        b[3*N:4*N] = y + r - 1.0

        sm_b = _soft_max(b, beta)
        if sm_b > 0:
            penalty += sm_b * sm_b
            exps = np.exp(b - sm_b)
            w = exps / np.sum(exps)
            g_b = 2.0 * sm_b * w
            # g_b[k] is d(penalty)/d(b_k)
            # b[0:N] = r - x  => d/dx_i = -1, d/dr_i = 1
            grad[0:N] += -g_b[0:N]
            grad[2*N:3*N] += g_b[0:N]
            # b[N:2*N] = x + r - 1 => d/dx_i = 1, d/dr_i = 1
            grad[0:N] += g_b[N:2*N]
            grad[2*N:3*N] += g_b[N:2*N]
            # b[2*N:3*N] = r - y => d/dy_i = -1, d/dr_i = 1
            grad[N:2*N] += -g_b[2*N:3*N]
            grad[2*N:3*N] += g_b[2*N:3*N]
            # b[3*N:4*N] = y + r - 1 => d/dy_i = 1, d/dr_i = 1
            grad[N:2*N] += g_b[3*N:4*N]
            grad[2*N:3*N] += g_b[3*N:4*N]

        # Pairwise overlap violations:
        # For pair (i,j): d_ij^2 - (r_i + r_j)^2 should be >= 0
        # Violation: (r_i + r_j)^2 - d_ij^2 (want <= 0)
        pair_viol = np.empty(N * (N - 1) // 2)
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                dr = r[i] + r[j]
                pair_viol[idx] = dr * dr - d2
                idx += 1

        sm_p = _soft_max(pair_viol, beta)
        if sm_p > 0:
            penalty += sm_p * sm_p
            exps = np.exp(pair_viol - sm_p)
            w = exps / np.sum(exps)
            g_p = 2.0 * sm_p * w
            # Now map gradient back
            idx = 0
            for i in range(N):
                for j in range(i + 1, N):
                    gp = g_p[idx]
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d2 = dx * dx + dy * dy
                    dr = r[i] + r[j]
                    # viol = dr^2 - d2
                    # d(viol)/dx_i = -2*dx, d(viol)/dx_j = 2*dx
                    # d(viol)/dy_i = -2*dy, d(viol)/dy_j = 2*dy
                    # d(viol)/dr_i = 2*dr, d(viol)/dr_j = 2*dr
                    grad[i] += gp * (-2.0 * dx)
                    grad[j] += gp * (2.0 * dx)
                    grad[N + i] += gp * (-2.0 * dy)
                    grad[N + j] += gp * (2.0 * dy)
                    grad[2*N + i] += gp * (2.0 * dr)
                    grad[2*N + j] += gp * (2.0 * dr)
                    idx += 1

        # Radius lower bound: r_i >= eps
        eps = 1e-6
        r_viol = np.maximum(eps - r, 0.0)
        penalty += np.sum(r_viol * r_viol)
        grad[2*N:3*N] += 2.0 * r_viol * (-1.0)

        return penalty, grad

    # --- Objective: maximize sum of radii, minimize penalty ---
    def _objective(v, penalty_weight=100.0):
        """Combined objective: -sum(r) + weight * penalty"""
        pen, _ = _penalty_function(v)
        return -np.sum(v[2*N:3*N]) + penalty_weight * pen

    def _objective_grad(v, penalty_weight=100.0):
        g = np.zeros(3 * N)
        g[2*N:3*N] = -1.0
        _, gp = _penalty_function(v)
        g += penalty_weight * gp
        return g

    # --- Validity check ---
    def _is_valid(v, tol=1e-7):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        if np.any(r < 1e-8):
            return False
        if np.any(x < r - tol) or np.any(x > 1 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                    return False
        return True

    # --- Post-process: project to feasibility ---
    def _post_process(v):
        """Project solution to be strictly feasible."""
        x = v[0:N].copy()
        y = v[N:2*N].copy()
        r = v[2*N:3*N].copy()
        # Clip radii
        r = np.maximum(r, 1e-6)
        # Clip positions to stay in bounds
        for i in range(N):
            x[i] = np.clip(x[i], r[i] + 1e-8, 1 - r[i] - 1e-8)
            y[i] = np.clip(y[i], r[i] + 1e-8, 1 - r[i] - 1e-8)
        # Resolve overlaps by shrinking radii
        for _ in range(50):
            changed = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    required = r[i] + r[j]
                    if d < required - 1e-9:
                        # Shrink the larger radius
                        shrink = (required - d + 1e-9) / 2.0
                        if r[i] >= r[j]:
                            r[i] -= shrink
                            x[i] += dx / (d + 1e-12) * shrink * 0.5
                            y[i] += dy / (d + 1e-12) * shrink * 0.5
                        else:
                            r[j] -= shrink
                            x[j] -= dx / (d + 1e-12) * shrink * 0.5
                            y[j] -= dy / (d + 1e-12) * shrink * 0.5
                        r[i] = max(r[i], 1e-6)
                        r[j] = max(r[j], 1e-6)
                        changed = True
            if not changed:
                break
        return np.concatenate([x, y, r])

    # --- Initializations ---
    def _grid_init(seed=0):
        rng = np.random.RandomState(seed)
        side = int(np.ceil(np.sqrt(N)))  # 6
        v = np.zeros(3 * N)
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                v[count] = (i + 0.5) / side + rng.uniform(-0.02, 0.02)
                v[N + count] = (j + 0.5) / side + rng.uniform(-0.02, 0.02)
                v[2*N + count] = 0.5 / side - 0.005
                count += 1
        v[0:N] = np.clip(v[0:N], 0.02, 0.98)
        v[N:2*N] = np.clip(v[N:2*N], 0.02, 0.98)
        v[2*N:3*N] = np.clip(v[2*N:3*N], 0.005, 0.49)
        return v

    def _hex_init(seed=0):
        """Hexagonal packing initialization."""
        rng = np.random.RandomState(seed)
        v = np.zeros(3 * N)
        # Hexagonal arrangement
        r0 = 0.07
        dx = 2 * r0
        dy = np.sqrt(3) * r0
        count = 0
        row = 0
        while count < N:
            for col in range(6):
                if count >= N:
                    break
                cx = r0 + col * dx + (row % 2) * r0
                cy = r0 + row * dy
                if cx + r0 > 1 - r0 or cy + r0 > 1 - r0:
                    continue
                v[count] = cx + rng.uniform(-0.005, 0.005)
                v[N + count] = cy + rng.uniform(-0.005, 0.005)
                v[2*N + count] = r0
                count += 1
            row += 1
        v[0:N] = np.clip(v[0:N], 0.02, 0.98)
        v[N:2*N] = np.clip(v[N:2*N], 0.02, 0.98)
        v[2*N:3*N] = np.clip(v[2*N:3*N], 0.005, 0.49)
        return v

    def _random_init(seed=0):
        rng = np.random.RandomState(seed)
        v = np.zeros(3 * N)
        v[0:N] = rng.uniform(0.1, 0.9, N)
        v[N:2*N] = rng.uniform(0.1, 0.9, N)
        v[2*N:3*N] = rng.uniform(0.03, 0.12, N)
        return v

    # --- L-BFGS-B optimization with soft penalty ---
    def _lbfgsb_pass(v0, maxiter=2000, penalty_weight=100.0):
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))
        for i in range(N):
            bounds.append((0.0, 1.0))
        for i in range(N):
            bounds.append((1e-6, 0.5))

        try:
            res = minimize(
                lambda v: _objective(v, penalty_weight),
                v0,
                jac=lambda v: _objective_grad(v, penalty_weight),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50}
            )
            return res.x
        except Exception:
            return v0

    # --- Radius growth pass: fix positions, maximize radii ---
    def _radius_growth_pass(v, maxiter=500):
        """Fix x,y and maximize sum of r subject to constraints."""
        x = v[0:N].copy()
        y = v[N:2*N].copy()
        r = v[2*N:3*N].copy()

        # SLSQP on radii only
        def obj_r(rv):
            return -np.sum(rv)

        def obj_r_grad(rv):
            return -np.ones(N)

        constraints = []
        # Boundary: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda rv, i=i: x[i] - rv[i],
                                'jac': lambda rv, i=i: -np.eye(N)[i]})
            constraints.append({'type': 'ineq', 'fun': lambda rv, i=i: 1 - x[i] - rv[i],
                                'jac': lambda rv, i=i: -np.eye(N)[i]})
            constraints.append({'type': 'ineq', 'fun': lambda rv, i=i: y[i] - rv[i],
                                'jac': lambda rv, i=i: -np.eye(N)[i]})
            constraints.append({'type': 'ineq', 'fun': lambda rv, i=i: 1 - y[i] - rv[i],
                                'jac': lambda rv, i=i: -np.eye(N)[i]})
        # Pairwise: (r_i + r_j)^2 <= (x_i-x_j)^2 + (y_i-y_j)^2
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                if d2 > 0:
                    d = np.sqrt(d2)
                    constraints.append({'type': 'ineq', 'fun': lambda rv, i=i, j=j, d=d: d - rv[i] - rv[j],
                                        'jac': lambda rv, i=i, j=j: -np.eye(N)[i] - np.eye(N)[j]})

        bounds_r = [(1e-6, 0.5)] * N
        try:
            res = minimize(obj_r, r, jac=obj_r_grad, method='SLSQP',
                          bounds=bounds_r, constraints=constraints,
                          options={'maxiter': maxiter, 'ftol': 1e-12})
            if res.success:
                r = res.x
        except Exception:
            pass

        return np.concatenate([x, y, r])

    # --- SLSQP final polish ---
    def _slsqp_polish(v0, maxiter=1000):
        """Full SLSQP with all constraints for final precision."""
        def make_constraints():
            cons = []
            for i in range(N):
                A = np.zeros(3 * N)
                A[i] = 1.0
                A[2*N + i] = -1.0
                cons.append({'type': 'ineq', 'fun': lambda v, A=A: A @ v, 'jac': lambda v, A=A: A})
                A = np.zeros(3 * N)
                A[i] = -1.0
                A[2*N + i] = -1.0
                cons.append({'type': 'ineq', 'fun': lambda v, A=A: 1.0 - A @ v, 'jac': lambda v, A=A: -A})
                A = np.zeros(3 * N)
                A[N + i] = 1.0
                A[2*N + i] = -1.0
                cons.append({'type': 'ineq', 'fun': lambda v, A=A: A @ v, 'jac': lambda v, A=A: A})
                A = np.zeros(3 * N)
                A[N + i] = -1.0
                A[2*N + i] = -1.0
                cons.append({'type': 'ineq', 'fun': lambda v, A=A: 1.0 - A @ v, 'jac': lambda v, A=A: -A})

            for i in range(N):
                for j in range(i + 1, N):
                    def pf(v, i=i, j=j):
                        dx = v[i] - v[j]
                        dy = v[N+i] - v[N+j]
                        dr = v[2*N+i] + v[2*N+j]
                        return dx*dx + dy*dy - dr*dr
                    def pj(v, i=i, j=j):
                        g = np.zeros(3*N)
                        dx = v[i] - v[j]
                        dy = v[N+i] - v[N+j]
                        dr = v[2*N+i] + v[2*N+j]
                        g[i] = 2*dx
                        g[j] = -2*dx
                        g[N+i] = 2*dy
                        g[N+j] = -2*dy
                        g[2*N+i] = -2*dr
                        g[2*N+j] = -2*dr
                        return g
                    cons.append({'type': 'ineq', 'fun': pf, 'jac': pj})

            for i in range(N):
                A = np.zeros(3 * N)
                A[2*N + i] = 1.0
                cons.append({'type': 'ineq', 'fun': lambda v, A=A: A @ v - 1e-6, 'jac': lambda v, A=A: A})
            return cons

        bounds = []
        for _ in range(N):
            bounds.append((0.0, 1.0))
        for _ in range(N):
            bounds.append((0.0, 1.0))
        for _ in range(N):
            bounds.append((1e-6, 0.5))

        try:
            res = minimize(
                lambda v: -np.sum(v[2*N:3*N]),
                v0,
                jac=lambda v: np.concatenate([np.zeros(N), np.zeros(N), -np.ones(N)]),
                method='SLSQP',
                bounds=bounds,
                constraints=make_constraints(),
                options={'maxiter': maxiter, 'ftol': 1e-14}
            )
            if res.success and _is_valid(res.x):
                return res.x
        except Exception:
            pass
        return None

    # --- Local search: perturb and re-optimize ---
    def _local_search(v0, n_perturb=5, sigma=0.01):
        best = v0.copy()
        best_val = np.sum(best[2*N:3*N])
        rng = np.random.RandomState(int(time.time() * 1000) % 10000)
        for _ in range(n_perturb):
            if time_left() < 20:
                break
            v_pert = best + rng.normal(0, sigma, 3 * N)
            v_pert[0:N] = np.clip(v_pert[0:N], 0.01, 0.99)
            v_pert[N:2*N] = np.clip(v_pert[N:2*N], 0.01, 0.99)
            v_pert[2*N:3*N] = np.clip(v_pert[2*N:3*N], 0.005, 0.49)
            v_opt = _lbfgsb_pass(v_pert, maxiter=500, penalty_weight=50.0)
            v_opt = _radius_growth_pass(v_opt)
            if _is_valid(v_opt):
                val = np.sum(v_opt[2*N:3*N])
                if val > best_val:
                    best_val = val
                    best = v_opt.copy()
        return best

    # --- Main optimization pipeline ---
    def _optimize(v0):
        """Full pipeline: L-BFGS-B -> radius growth -> SLSQP polish -> local search"""
        # Phase 1: L-BFGS-B with soft penalty
        v1 = _lbfgsb_pass(v0, maxiter=2000, penalty_weight=100.0)
        if time_left() < 25:
            v1 = _post_process(v1)
            return v1

        # Phase 2: Reduce penalty weight for better radius maximization
        v2 = _lbfgsb_pass(v1, maxiter=1000, penalty_weight=10.0)

        # Phase 3: Radius growth (fix positions)
        v3 = _radius_growth_pass(v2, maxiter=500)

        # Phase 4: SLSQP polish
        v4 = _slsqp_polish(v3, maxiter=500)
        if v4 is not None:
            v3 = v4

        # Phase 5: Local search
        if time_left() > 30:
            v5 = _local_search(v3, n_perturb=3, sigma=0.005)
            if _is_valid(v5):
                v3 = v5

        # Phase 6: Final radius growth
        v6 = _radius_growth_pass(v3, maxiter=200)
        if _is_valid(v6):
            v3 = v6

        # Final SLSQP
        if time_left() > 15:
            v7 = _slsqp_polish(v3, maxiter=300)
            if v7 is not None and _is_valid(v7):
                v3 = v7

        return _post_process(v3)

    # --- Run multiple restarts ---
    best_v = None
    best_val = -np.inf

    inits = [
        _grid_init(0),
        _grid_init(1),
        _hex_init(0),
        _hex_init(1),
        _random_init(42),
        _random_init(123),
    ]

    for idx, v0 in enumerate(inits):
        if time_left() < 25:
            break
        result = _optimize(v0)
        if result is not None and _is_valid(result):
            val = np.sum(result[2*N:3*N])
            if val > best_val:
                best_val = val
                best_v = result.copy()

    # Additional restarts perturbing best
    if best_v is not None:
        for extra in range(4):
            if time_left() < 25:
                break
            rng = np.random.RandomState(extra * 777 + 13)
            v0 = best_v + rng.normal(0, 0.02, 3 * N)
            v0[0:N] = np.clip(v0[0:N], 0.01, 0.99)
            v0[N:2*N] = np.clip(v0[N:2*N], 0.01, 0.99)
            v0[2*N:3*N] = np.clip(v0[2*N:3*N], 0.005, 0.49)
            result = _optimize(v0)
            if result is not None and _is_valid(result):
                val = np.sum(result[2*N:3*N])
                if val > best_val:
                    best_val = val
                    best_v = result.copy()

    if best_v is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_v[0:N]
    y = best_v[N:2*N]
    r = best_v[2*N:3*N]
    return np.column_stack([x, y, r])

# EVOLVE-BLOCK-END
