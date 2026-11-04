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
        Vectorized for speed.
        """
        normal_angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Vectorize over all hexagons
        cs = np.cos(phis)
        sn = np.sin(phis)
        
        # Rotate vertices for each hexagon: (n_hex, 6, 2)
        vx = np.outer(cs, base_vertices[:, 0]) - np.outer(sn, base_vertices[:, 1])
        vy = np.outer(cs, base_vertices[:, 0]) + np.outer(sn, base_vertices[:, 1])
        
        # Add centers: (n_hex, 6, 2)
        vx += centers[:, 0:1]
        vy += centers[:, 1:2]
        
        # Project onto normals: (n_hex, 6, 6)
        dots = vx @ normals[:, 0] + vy @ normals[:, 1]
        
        max_proj = np.max(np.abs(dots))
        return 2.0 * max_proj / np.sqrt(3)

    def smooth_objective(params, beta=80.0):
        theta_out = params[0]
        phis = params[1:]

        normal_angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Vectorize over all hexagons
        cs = np.cos(phis)
        sn = np.sin(phis)
        
        vx = np.outer(cs, base_vertices[:, 0]) - np.outer(sn, base_vertices[:, 1])
        vy = np.outer(cs, base_vertices[:, 0]) + np.outer(sn, base_vertices[:, 1])
        
        vx += centers[:, 0:1]
        vy += centers[:, 1:2]
        
        dots = vx @ normals[:, 0] + vy @ normals[:, 1]
        all_abs_dots = np.abs(dots).flatten()

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

    # Phase 3: Adaptive coordinate descent with line search
    if time.time() < deadline - 20:
        for iteration in range(200):
            if time.time() > deadline - 15:
                break

            improved = False
            for j in range(n_hex + 1):
                if time.time() > deadline - 15:
                    break

                current_val = best_params[j]
                best_local = best_score
                best_local_val = current_val
                
                # Adaptive step sizes based on current scale
                base_step = 1e-2
                for step_scale in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                    delta = base_step * step_scale
                    for sign in [1, -1]:
                        test_val = current_val + sign * delta
                        test_param = best_params.copy()
                        test_param[j] = test_val
                        try:
                            test_score = objective(test_param)
                            if test_score < best_local - 1e-16:
                                best_local = test_score
                                best_local_val = test_val
                                # Try to extend in the same direction
                                for extend in range(5):
                                    next_val = test_val + sign * delta * (0.5 ** extend)
                                    next_param = best_params.copy()
                                    next_param[j] = next_val
                                    next_score = objective(next_param)
                                    if next_score < best_local - 1e-16:
                                        best_local = next_score
                                        best_local_val = next_val
                                        test_val = next_val
                                    else:
                                        break
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
                          options={'maxiter': 15000, 'xatol': 1e-16, 'fatol': 1e-16, 'adaptive': True})
            if res.fun < best_score - 1e-16:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        # Additional L-BFGS-B attempt
        if time.time() < deadline - 8:
            try:
                res2 = minimize(objective, best_params, method='L-BFGS-B',
                               options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-16})
                if res2.fun < best_score - 1e-16:
                    best_score = res2.fun
                    best_params = res2.x.copy()
            except Exception:
                pass

    # Phase 5: Ultra-fine coordinate descent with golden section
    if time.time() < deadline - 5:
        for iteration in range(50):
            if time.time() > deadline - 3:
                break

            improved = False
            for j in range(n_hex + 1):
                if time.time() > deadline - 3:
                    break

                current_val = best_params[j]
                best_local = best_score
                best_local_val = current_val
                
                # Golden section search in a small interval
                span = 1e-6
                a = current_val - span
                b = current_val + span
                
                # Evaluate at golden points
                gr = (np.sqrt(5) - 1) / 2
                c = b - gr * (b - a)
                d = a + gr * (b - a)
                
                f_c = objective(np.concatenate([[c], best_params[1:]])) if j == 0 else objective(np.concatenate([best_params[:1], [c]] + list(best_params[2:])))
                f_d = objective(np.concatenate([[d], best_params[1:]])) if j == 0 else objective(np.concatenate([best_params[:1], [d]] + list(best_params[2:])))
                
                # Iterate golden section
                for _ in range(30):
                    if time.time() > deadline - 3:
                        break
                    if f_c < f_d:
                        b = d
                        d = c
                        f_d = f_c
                        c = b - gr * (b - a)
                        f_c = objective(np.concatenate([[c], best_params[1:]])) if j == 0 else objective(np.concatenate([best_params[:1], [c]] + list(best_params[2:])))
                    else:
                        a = c
                        c = d
                        f_c = f_d
                        d = a + gr * (b - a)
                        f_d = objective(np.concatenate([[d], best_params[1:]])) if j == 0 else objective(np.concatenate([best_params[:1], [d]] + list(best_params[2:])))
                
                mid = (a + b) / 2
                test_score = objective(np.concatenate([[mid], best_params[1:]])) if j == 0 else objective(np.concatenate([best_params[:1], [mid]] + list(best_params[2:])))
                if test_score < best_local - 1e-18:
                    best_local = test_score
                    best_local_val = mid

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
        for _ in range(300):
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
        
        # Also try local refinement around theta_opt
        for delta in [1e-4, -1e-4, 1e-5, -1e-5, 1e-6, -1e-6]:
            test_theta = theta_opt + delta
            test_score = side_1d(test_theta)
            if test_score < best_score - 1e-14:
                best_score = test_score
                best_params[0] = test_theta

    # Phase 7: Joint fine-tuning with adaptive perturbations
    if time.time() < deadline - 5:
        for _ in range(100):
            if time.time() > deadline - 3:
                break
            
            improved = False
            # Try small random perturbations with decreasing scale
            scale = 1e-5 * (0.8 ** (_ // 10))
            for trial in range(30):
                if time.time() > deadline - 3:
                    break
                perturbation = np.random.normal(0, scale, n_hex + 1)
                test_params = best_params + perturbation
                test_score = objective(test_params)
                if test_score < best_score - 1e-14:
                    best_score = test_score
                    best_params = test_params.copy()
                    improved = True
            
            if not improved and scale < 1e-8:
                break

    # Extract solution
    theta_out = best_params[0]
    phis = best_params[1:]

    inner = []
    for i in range(n_hex):
        inner.append([centers[i][0], centers[i][1], np.rad2deg(phis[i])])

    side = best_score + 1e-10  # Tighter margin for better score

    return inner, (0.0, 0.0), side, np.rad2deg(theta_out)

if __name__ == "__main__":
    construct()
