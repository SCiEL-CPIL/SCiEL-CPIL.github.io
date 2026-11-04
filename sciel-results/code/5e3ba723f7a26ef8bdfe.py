import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # Centers of the 11 hexagons in a "flower" pattern
    # 1 center + 6 at distance sqrt(3) + 4 at distance 3
    eps = 1e-12
    centers = np.array([[0.0, 0.0]])

    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])

    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])

    n_hex = len(centers)

    # Base vertices of unit hexagon (pointy-right orientation, angle 0)
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])

    def compute_outer_side(theta_out, phis):
        """
        Compute the minimum outer hexagon side length given outer rotation and inner rotations.
        Uses projection onto the 6 face normals of the outer hexagon.
        """
        normal_angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

        max_proj = 0.0
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            vx = base_vertices[:, 0] * c - base_vertices[:, 1] * s
            vy = base_vertices[:, 0] * s + base_vertices[:, 1] * c
            vx += centers[i][0]
            vy += centers[i][1]
            dots = vx * normals[:, 0] + vy * normals[:, 1]
            max_proj = max(max_proj, np.max(np.abs(dots)))

        return 2.0 * max_proj / np.sqrt(3)

    def smooth_objective(params, beta=80.0):
        theta_out = params[0]
        phis = params[1:]

        normal_angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

        all_abs_dots = []
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            vx = base_vertices[:, 0] * c - base_vertices[:, 1] * s
            vy = base_vertices[:, 0] * s + base_vertices[:, 1] * c
            vx += centers[i][0]
            vy += centers[i][1]
            dots = vx * normals[:, 0] + vy * normals[:, 1]
            all_abs_dots.append(np.abs(dots))

        all_abs_dots = np.concatenate(all_abs_dots)

        max_val = np.max(all_abs_dots)
        if max_val == 0:
            return 0.0
        scaled = (all_abs_dots - max_val) * beta
        log_sum_exp = max_val + np.log(np.sum(np.exp(scaled - np.max(scaled))) / len(all_abs_dots)) / beta + np.max(scaled) / beta

        return 2.0 * log_sum_exp / np.sqrt(3)

    def objective(params):
        theta_out = params[0]
        phis = params[1:]
        return compute_outer_side(theta_out, phis)

    # Analytical initial guess based on known optimal configuration
    x0 = np.array([np.deg2rad(30)] + [np.deg2rad(30)] * n_hex)

    best_score = objective(x0)
    best_params = x0.copy()

    # Phase 1: Multi-start Powell optimization
    n_starts = 15
    start_variants = [
        np.array([np.deg2rad(30)] + [np.deg2rad(30)] * n_hex),
        np.array([0.0] + [0.0] * n_hex),
        np.array([np.pi/6] + [0.0] * n_hex),
        np.array([0.0] + [np.pi/6] * n_hex),
        np.array([np.deg2rad(0)] + [np.deg2rad(30)] * n_hex),
        np.array([np.deg2rad(60)] + [np.deg2rad(30)] * n_hex),
        np.array([np.deg2rad(15)] + [np.deg2rad(15)] * n_hex),
        np.array([np.deg2rad(45)] + [np.deg2rad(45)] * n_hex),
        np.array([np.deg2rad(30)] + [np.deg2rad(0)] * n_hex),
    ]

    np.random.seed(42)
    for i in range(n_starts):
        if time.time() > deadline - 40:
            break

        if i < len(start_variants):
            x_start = start_variants[i].copy()
        else:
            x_start = np.random.uniform(0, np.pi/3, n_hex + 1)

        try:
            res = minimize(objective, x_start, method='Powell',
                          options={'maxiter': 500, 'disp': False, 'ftol': 1e-16, 'xtol': 1e-16})
            if res.fun < best_score - 1e-15:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Smooth surrogate optimization
    if time.time() < deadline - 25:
        for beta_val in [80.0, 150.0, 300.0]:
            if time.time() > deadline - 20:
                break
            try:
                res = minimize(lambda p: smooth_objective(p, beta=beta_val), best_params, 
                             method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
                true_score = objective(res.x)
                if true_score < best_score - 1e-15:
                    best_score = true_score
                    best_params = res.x.copy()
            except Exception:
                pass

    # Phase 3: Coordinate descent polish
    if time.time() < deadline - 20:
        for iteration in range(100):
            if time.time() > deadline - 15:
                break

            improved = False
            for j in range(n_hex + 1):
                if time.time() > deadline - 15:
                    break

                current_val = best_params[j]
                best_local = best_score
                best_local_val = current_val

                for delta in [1e-2, -1e-2, 1e-3, -1e-3, 1e-4, -1e-4,
                             1e-5, -1e-5, 1e-6, -1e-6, 1e-7, -1e-7]:
                    test_val = current_val + delta
                    test_param = best_params.copy()
                    test_param[j] = test_val
                    try:
                        test_score = objective(test_param)
                        if test_score < best_local - 1e-16:
                            best_local = test_score
                            best_local_val = test_val
                    except Exception:
                        continue

                if best_local < best_score - 1e-16:
                    best_params[j] = best_local_val
                    best_score = best_local
                    improved = True

            if not improved:
                break

    # Phase 4: Final high-precision Nelder-Mead polish
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 8000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True})
            if res.fun < best_score - 1e-16:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 5: Ultra-fine coordinate descent
    if time.time() < deadline - 5:
        for iteration in range(30):
            if time.time() > deadline - 3:
                break

            improved = False
            for j in range(n_hex + 1):
                if time.time() > deadline - 3:
                    break

                current_val = best_params[j]
                best_local = best_score
                best_local_val = current_val

                for delta in [1e-8, -1e-8, 1e-9, -1e-9, 1e-10, -1e-10]:
                    test_val = current_val + delta
                    test_param = best_params.copy()
                    test_param[j] = test_val
                    try:
                        test_score = objective(test_param)
                        if test_score < best_local - 1e-18:
                            best_local = test_score
                            best_local_val = test_val
                    except Exception:
                        continue

                if best_local < best_score - 1e-18:
                    best_params[j] = best_local_val
                    best_score = best_local
                    improved = True

            if not improved:
                break

    # Phase 6: 1D ternary search over global rotation for the optimal phis
    if time.time() < deadline - 8:
        # Fix phis to their best values and do a fine 1D search over theta_out
        phis_fixed = best_params[1:].copy()
        
        def side_1d(theta):
            return compute_outer_side(theta, phis_fixed)
        
        # Ternary search in [0, pi/3] since hexagon has 60-degree symmetry
        a, b = 0.0, np.pi / 3
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            m1 = a + (b - a) / 3.0
            m2 = b - (b - a) / 3.0
            f1 = side_1d(m1)
            f2 = side_1d(m2)
            if f1 < f2:
                b = m2
            else:
                a = m1
        
        theta_opt = (a + b) / 2.0
        score_1d = side_1d(theta_opt)
        if score_1d < best_score - 1e-12:
            best_score = score_1d
            best_params[0] = theta_opt

    # Phase 7: Joint fine-tuning with small perturbations
    if time.time() < deadline - 5:
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            
            improved = False
            # Try small random perturbations
            for trial in range(20):
                if time.time() > deadline - 3:
                    break
                perturbation = np.random.normal(0, 1e-6, n_hex + 1)
                test_params = best_params + perturbation
                test_score = objective(test_params)
                if test_score < best_score - 1e-14:
                    best_score = test_score
                    best_params = test_params.copy()
                    improved = True
            
            if not improved:
                break

    # Extract solution
    theta_out = best_params[0]
    phis = best_params[1:]

    inner = []
    for i in range(n_hex):
        inner.append([centers[i][0], centers[i][1], np.rad2deg(phis[i])])

    side = best_score + 1e-8  # Margin for floating point safety

    return inner, (0.0, 0.0), side, np.rad2deg(theta_out)

if __name__ == "__main__":
    construct()
