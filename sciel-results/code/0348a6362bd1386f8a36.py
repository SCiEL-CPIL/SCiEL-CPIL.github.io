"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random

N = 21

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145

    idx_i, idx_j = np.triu_indices(N, k=1)

    def get_rect_dims(t):
        t = np.clip(t, 0.15, 0.85)
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        return w, h

    def evaluate(state):
        t = state[0]
        w, h = get_rect_dims(t)
        penalty = 0.0
        coords = np.zeros((N, 3))
        for i in range(N):
            coords[i, 0] = state[1 + 3*i]
            coords[i, 1] = state[1 + 3*i+1]
            coords[i, 2] = state[1 + 3*i+2]
        radii = coords[:, 2]
        sum_r = np.sum(radii)
        x = coords[:, 0]
        y = coords[:, 1]
        viol_l = np.maximum(0, radii - x)
        viol_r = np.maximum(0, x + radii - w)
        viol_b = np.maximum(0, radii - y)
        viol_t = np.maximum(0, y + radii - h)
        penalty += 500.0 * (np.sum(viol_l) + np.sum(viol_r) + np.sum(viol_b) + np.sum(viol_t))
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        min_dist = radii[idx_i] + radii[idx_j]
        overlap_mask = dist_sq < min_dist * min_dist
        if np.any(overlap_mask):
            dist = np.sqrt(dist_sq[overlap_mask])
            overlaps = min_dist[overlap_mask] - dist
            penalty += 1000.0 * np.sum(overlaps)
        score = sum_r - penalty
        return score, penalty

    def check_valid(state, eps=1e-7):
        t = state[0]
        w, h = get_rect_dims(t)
        coords = np.zeros((N, 3))
        for i in range(N):
            coords[i, 0] = state[1 + 3*i]
            coords[i, 1] = state[1 + 3*i+1]
            coords[i, 2] = state[1 + 3*i+2]
        radii = coords[:, 2]
        if np.any(radii < 1e-6):
            return False
        x = coords[:, 0]
        y = coords[:, 1]
        if np.any(x - radii < -eps) or np.any(x + radii > w + eps) or \
           np.any(y - radii < -eps) or np.any(y + radii > h + eps):
            return False
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        min_dist = radii[idx_i] + radii[idx_j] - eps
        if np.any(dist_sq < min_dist * min_dist):
            return False
        return True

    def make_hexagonal(t_init, equal_r=True, jitter=0.0):
        w, h = get_rect_dims(t_init)
        state = np.zeros(1 + 3*N)
        state[0] = t_init
        r_est = np.sqrt(0.9069 * w * h / (N * np.pi))
        positions = []
        r_current = r_est
        for attempt in range(30):
            positions = []
            y = r_current
            row_idx = 0
            while y <= h - r_current + 1e-9:
                if row_idx % 2 == 0:
                    x_start = r_current
                else:
                    x_start = r_current + r_current
                x = x_start
                while x <= w - r_current + 1e-9:
                    positions.append((x, y, r_current))
                    x += 2.0 * r_current
                y += np.sqrt(3) * r_current
                row_idx += 1
            if len(positions) >= N:
                break
            r_current *= 0.95
        if len(positions) < N:
            side = 5
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) >= N:
                        break
                    gx = (i + 0.5) / side * w
                    gy = (j + 0.5) / side * h
                    r = 0.5 / side * min(w, h) * 0.95
                    positions.append((gx, gy, r))
        for i in range(N):
            if i < len(positions):
                x, y, r = positions[i]
                if jitter > 0:
                    x += random.uniform(-jitter, jitter)
                    y += random.uniform(-jitter, jitter)
                state[1 + 3*i] = x
                state[1 + 3*i + 1] = y
                state[1 + 3*i + 2] = r
            else:
                state[1 + 3*i] = w/2
                state[1 + 3*i + 1] = h/2
                state[1 + 3*i + 2] = 0.01
        return state

    def make_grid(t_init, jitter=0.0):
        w, h = get_rect_dims(t_init)
        state = np.zeros(1 + 3*N)
        state[0] = t_init
        best_positions = None
        best_area = 0
        for rows in range(3, 9):
            for cols in range(3, 9):
                if rows * cols < N:
                    continue
                r = min(w / cols, h / rows) * 0.48
                positions = []
                for i in range(rows):
                    for j in range(cols):
                        if len(positions) >= N:
                            break
                        x = (j + 0.5) * w / cols
                        y = (i + 0.5) * h / rows
                        positions.append((x, y, r))
                if len(positions) >= N:
                    total_area = sum(r**2 for _, _, r in positions)
                    if total_area > best_area:
                        best_area = total_area
                        best_positions = positions
        if best_positions is None:
            return make_hexagonal(t_init, jitter)
        for i in range(N):
            x, y, r = best_positions[i]
            if jitter > 0:
                x += random.uniform(-jitter, jitter)
                y += random.uniform(-jitter, jitter)
            state[1 + 3*i] = x
            state[1 + 3*i + 1] = y
            state[1 + 3*i + 2] = r
        return state

    def polish_gradient(state, max_iter=3000, beta_b=0.005, beta_o=0.005,
                        weight_b=2000.0, weight_o=10000.0, equal_r_weight=5.0):
        from scipy.optimize import minimize

        def objective_and_grad(s):
            t = s[0]
            w, h = get_rect_dims(t)
            coords = np.zeros((N, 3))
            for i in range(N):
                coords[i, 0] = s[1 + 3*i]
                coords[i, 1] = s[1 + 3*i+1]
                coords[i, 2] = s[1 + 3*i+2]
            radii = coords[:, 2]
            x = coords[:, 0]
            y = coords[:, 1]

            obj = -np.sum(radii)

            # Equal-radius penalty (variance)
            mean_r = np.mean(radii)
            obj += equal_r_weight * np.sum((radii - mean_r)**2)

            viol_l = radii - x
            viol_r = x + radii - w
            viol_b = radii - y
            viol_t = y + radii - h

            sl = beta_b * np.logaddexp(0, viol_l / beta_b)
            sr = beta_b * np.logaddexp(0, viol_r / beta_b)
            sb = beta_b * np.logaddexp(0, viol_b / beta_b)
            st = beta_b * np.logaddexp(0, viol_t / beta_b)
            obj += weight_b * (np.sum(sl**2) + np.sum(sr**2) + np.sum(sb**2) + np.sum(st**2))

            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[idx_i] + radii[idx_j]
            dist = np.sqrt(dist_sq + 1e-12)
            overlap = min_dist - dist
            so = beta_o * np.logaddexp(0, overlap / beta_o)
            obj += weight_o * np.sum(so**2)

            grad = np.zeros_like(s)
            grad[2::3] -= 1.0
            # Equal radius gradient
            grad[2::3] -= 2 * equal_r_weight * (radii - mean_r) / N

            sig_l = 1.0 / (1.0 + np.exp(-viol_l / beta_b))
            sig_r = 1.0 / (1.0 + np.exp(-viol_r / beta_b))
            sig_b = 1.0 / (1.0 + np.exp(-viol_b / beta_b))
            sig_t = 1.0 / (1.0 + np.exp(-viol_t / beta_b))
            grad[0::3] += 2 * weight_b * (-sl * sig_l + sr * sig_r)
            grad[1::3] += 2 * weight_b * (-sb * sig_b + st * sig_t)
            grad[2::3] += 2 * weight_b * (sl * sig_l + sr * sig_r + sb * sig_b + st * sig_t)

            sig_o = 1.0 / (1.0 + np.exp(-overlap / beta_o))
            factor = 2 * weight_o * so * sig_o
            grad_x_contrib = np.zeros(N)
            grad_y_contrib = np.zeros(N)
            grad_r_contrib = np.zeros(N)
            np.add.at(grad_x_contrib, idx_i, -dx / dist * factor)
            np.add.at(grad_x_contrib, idx_j, dx / dist * factor)
            np.add.at(grad_y_contrib, idx_i, -dy / dist * factor)
            np.add.at(grad_y_contrib, idx_j, dy / dist * factor)
            np.add.at(grad_r_contrib, idx_i, factor)
            np.add.at(grad_r_contrib, idx_j, factor)
            grad[0::3] += grad_x_contrib
            grad[1::3] += grad_y_contrib
            grad[2::3] += grad_r_contrib

            return obj, grad

        def objective(s):
            return objective_and_grad(s)[0]

        def gradient(s):
            return objective_and_grad(s)[1]

        bounds = [(0.15, 0.85)]
        for i in range(N):
            bounds.append((0, 2.0))
            bounds.append((0, 2.0))
            bounds.append((1e-4, 0.45))

        try:
            result = minimize(
                objective, state, jac=gradient, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100}
            )
            if result.fun < objective(state):
                return result.x
        except Exception:
            pass
        return state

    def force_relax(state, max_iter=3000, equal_r_weight=2.0):
        best_state = state.copy()
        best_score, _ = evaluate(best_state)
        step_size = 0.004
        patience = 0
        max_patience = 150

        for it in range(max_iter):
            s = best_state.copy()
            t = s[0]
            w, h = get_rect_dims(t)
            coords = np.zeros((N, 3))
            for i in range(N):
                coords[i, 0] = s[1 + 3*i]
                coords[i, 1] = s[1 + 3*i+1]
                coords[i, 2] = s[1 + 3*i+2]
            radii = coords[:, 2]
            x = coords[:, 0].copy()
            y = coords[:, 1].copy()
            fx = np.zeros(N)
            fy = np.zeros(N)
            fr = np.zeros(N)

            viol_l = np.maximum(0, radii - x)
            fx += viol_l * 10.0
            viol_r = np.maximum(0, x + radii - w)
            fx -= viol_r * 10.0
            viol_b = np.maximum(0, radii - y)
            fy += viol_b * 10.0
            viol_t = np.maximum(0, y + radii - h)
            fy -= viol_t * 10.0

            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq + 1e-12)
            target = radii[idx_i] + radii[idx_j]
            diff = dist - target
            k_repel = 50.0
            k_attract = 2.0
            k_eff = np.where(diff < 0, k_repel, k_attract)
            force_mag = np.where(diff < 0, k_repel * (-diff), -k_attract * diff)
            dir_x = dx / dist
            dir_y = dy / dist
            f_x_pair = force_mag * dir_x
            f_y_pair = force_mag * dir_y
            np.add.at(fx, idx_i, f_x_pair)
            np.add.at(fx, idx_j, -f_x_pair)
            np.add.at(fy, idx_i, f_y_pair)
            np.add.at(fy, idx_j, -f_y_pair)

            # Equal radius force
            mean_r = np.mean(radii)
            fr += equal_r_weight * (mean_r - radii) * 0.1

            x_new = x + step_size * fx
            y_new = y + step_size * fy
            r_new = radii + step_size * 0.5 * fr
            for i in range(N):
                x_new[i] = max(radii[i], min(w - radii[i], x_new[i]))
                y_new[i] = max(radii[i], min(h - radii[i], y_new[i]))
                r_new[i] = max(1e-4, min(0.45 * min(w, h), r_new[i]))

            new_state = s.copy()
            for i in range(N):
                new_state[1 + 3*i] = x_new[i]
                new_state[1 + 3*i + 1] = y_new[i]
                new_state[1 + 3*i + 2] = r_new[i]
            new_score, _ = evaluate(new_state)
            if new_score > best_score:
                best_state = new_state
                best_score = new_score
                patience = 0
                step_size = min(0.01, step_size * 1.02)
            else:
                patience += 1
                step_size = max(0.001, step_size * 0.95)
            if patience > max_patience:
                break
        return best_state

    def local_optimize(state, max_iter=5000):
        best_state = state.copy()
        best_score, _ = evaluate(best_state)
        T = 0.3
        alpha = 0.9997
        for _ in range(max_iter):
            new_state = best_state.copy()
            var_idx = random.randint(0, len(new_state)-1)
            if var_idx == 0:
                new_state[0] += random.gauss(0, 0.015)
                if new_state[0] < 0.15 or new_state[0] > 0.85:
                    T *= alpha
                    continue
            else:
                circle_idx = (var_idx - 1) // 3
                comp = (var_idx - 1) % 3
                scale = T * 0.1 + 0.008
                if comp == 2:
                    scale *= 0.3
                new_state[var_idx] += random.gauss(0, scale)
                if comp == 2:
                    if new_state[var_idx] < 1e-4:
                        new_state[var_idx] = 1e-4
                    w, h = get_rect_dims(new_state[0])
                    max_r = 0.45 * min(w, h)
                    if new_state[var_idx] > max_r:
                        new_state[var_idx] = max_r
            new_score, _ = evaluate(new_state)
            if new_score > best_score or random.random() < np.exp((new_score - best_score) / T):
                best_state = new_state
                best_score = new_score
            T *= alpha
        return best_state, best_score

    def optimize_equal_radius(state, max_iter=2000):
        """Optimize assuming all circles have equal radius r, then find best positions."""
        from scipy.optimize import minimize

        t = state[0]
        w, h = get_rect_dims(t)

        def eq_obj(c):
            r = c[0]
            x = c[1:N+1]
            y = c[N+1:2*N+1]
            obj = -N * r
            # Boundary
            viol_l = r - x
            viol_r = x + r - w
            viol_b = r - y
            viol_t = y + r - h
            beta = 0.01
            sl = beta * np.logaddexp(0, viol_l / beta)
            sr = beta * np.logaddexp(0, viol_r / beta)
            sb = beta * np.logaddexp(0, viol_b / beta)
            st = beta * np.logaddexp(0, viol_t / beta)
            obj += 5000.0 * (np.sum(sl**2) + np.sum(sr**2) + np.sum(sb**2) + np.sum(st**2))
            # Overlap
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq + 1e-12)
            overlap = 2*r - dist
            so = beta * np.logaddexp(0, overlap / beta)
            obj += 20000.0 * np.sum(so**2)
            return obj

        def eq_grad(c):
            r = c[0]
            x = c[1:N+1]
            y = c[N+1:2*N+1]
            grad = np.zeros(2*N+1)
            grad[0] = -N
            beta = 0.01
            viol_l = r - x
            viol_r = x + r - w
            viol_b = r - y
            viol_t = y + r - h
            sl = beta * np.logaddexp(0, viol_l / beta)
            sr = beta * np.logaddexp(0, viol_r / beta)
            sb = beta * np.logaddexp(0, viol_b / beta)
            st = beta * np.logaddexp(0, viol_t / beta)
            sig_l = 1.0 / (1.0 + np.exp(-viol_l / beta))
            sig_r = 1.0 / (1.0 + np.exp(-viol_r / beta))
            sig_b = 1.0 / (1.0 + np.exp(-viol_b / beta))
            sig_t = 1.0 / (1.0 + np.exp(-viol_t / beta))
            grad[1:N+1] += 2*5000.0 * (-sl*sig_l + sr*sig_r)
            grad[N+1:2*N+1] += 2*5000.0 * (-sb*sig_b + st*sig_t)
            grad[0] += 2*5000.0 * (sl*sig_l + sr*sig_r + sb*sig_b + st*sig_t)
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq + 1e-12)
            overlap = 2*r - dist
            so = beta * np.logaddexp(0, overlap / beta)
            sig_o = 1.0 / (1.0 + np.exp(-overlap / beta))
            factor = 2*20000.0 * so * sig_o
            gx = np.zeros(N)
            gy = np.zeros(N)
            np.add.at(gx, idx_i, -dx/dist * factor)
            np.add.at(gx, idx_j, dx/dist * factor)
            np.add.at(gy, idx_i, -dy/dist * factor)
            np.add.at(gy, idx_j, dy/dist * factor)
            grad[1:N+1] += gx
            grad[N+1:2*N+1] += gy
            grad[0] += 2*factor.sum()
            return grad

        # Initial guess
        r0 = np.sqrt(0.9069 * w * h / (N * np.pi))
        c0 = np.zeros(2*N+1)
        c0[0] = r0
        for i in range(N):
            c0[1+i] = (i % 5 + 0.5) * w / 5
            c0[N+1+i] = (i // 5 + 0.5) * h / 5

        bounds = [(1e-4, 0.4*min(w,h))] + [(0, w)]*N + [(0, h)]*N
        try:
            result = minimize(eq_obj, c0, jac=eq_grad, method='L-BFGS-B',
                             bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-14})
            if result.fun < eq_obj(c0):
                r_opt = result.x[0]
                x_opt = result.x[1:N+1]
                y_opt = result.x[N+1:2*N+1]
                new_state = state.copy()
                new_state[0] = t
                for i in range(N):
                    new_state[1+3*i] = x_opt[i]
                    new_state[1+3*i+1] = y_opt[i]
                    new_state[1+3*i+2] = r_opt
                return new_state
        except Exception:
            pass
        return state

    # Main
    best_state = None
    best_score = -np.inf
    num_restarts = 10
    t_candidates = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.52, 0.48, 0.53]

    for restart in range(num_restarts):
        if time.time() > deadline - 50:
            break
        t_init = t_candidates[restart % len(t_candidates)] + random.uniform(-0.02, 0.02)
        if restart % 2 == 0:
            state = make_hexagonal(t_init, jitter=0.005)
        else:
            state = make_grid(t_init, jitter=0.005)

        remaining = deadline - 50 - time.time()
        if remaining < 10:
            break

        # Phase 1: SA
        sa_iters = min(3000, max(500, int(remaining * 80)))
        state, score = local_optimize(state, sa_iters)

        # Phase 2: Equal radius optimization
        if time.time() < deadline - 30:
            eq_state = optimize_equal_radius(state, max_iter=1500)
            eq_score, _ = evaluate(eq_state)
            if eq_score > score:
                state = eq_state
                score = eq_score

        # Phase 3: Force relax
        if time.time() < deadline - 25:
            force_state = force_relax(state, max_iter=2000)
            force_score, _ = evaluate(force_state)
            if force_score > score:
                state = force_state
                score = force_score

        # Phase 4: Gradient polish with equal-r
        if time.time() < deadline - 15:
            for pi in range(2):
                if time.time() > deadline - 10:
                    break
                beta_val = 0.005 / (pi + 1)
                polished = polish_gradient(state, max_iter=2000, beta_b=beta_val, beta_o=beta_val,
                                           weight_b=3000.0, weight_o=15000.0, equal_r_weight=5.0)
                pol_score, _ = evaluate(polished)
                if pol_score > score:
                    state = polished
                    score = pol_score
                else:
                    break

        if score > best_score:
            best_score = score
            best_state = state.copy()

    # Final polish
    if best_state is not None and time.time() < deadline - 5:
        for pp in range(3):
            if time.time() > deadline - 2:
                break
            beta_final = 0.002 / (pp + 1)
            final_polished = polish_gradient(best_state, max_iter=3000, beta_b=beta_final, beta_o=beta_final,
                                            weight_b=5000.0, weight_o=25000.0, equal_r_weight=8.0)
            fp_score, _ = evaluate(final_polished)
            if fp_score > best_score:
                best_state = final_polished
                best_score = fp_score

    if best_state is None:
        best_state = make_hexagonal(0.5)
    elif not check_valid(best_state):
        repair = best_state.copy()
        for shrink in range(50):
            if check_valid(repair):
                break
            for i in range(N):
                repair[1 + 3*i + 2] *= 0.95
            t_r = repair[0]
            w_r, h_r = get_rect_dims(t_r)
            for i in range(N):
                r_r = repair[1 + 3*i + 2]
                repair[1 + 3*i] = max(r_r, min(w_r - r_r, repair[1 + 3*i]))
                repair[1 + 3*i + 1] = max(r_r, min(h_r - r_r, repair[1 + 3*i + 1]))
        if check_valid(repair):
            best_state = repair
        else:
            best_state = make_hexagonal(0.5)

    result = np.zeros((N, 3))
    t_final = best_state[0]
    w_final, h_final = get_rect_dims(t_final)
    for i in range(N):
        x = best_state[1 + 3*i]
        y = best_state[1 + 3*i + 1]
        r = best_state[1 + 3*i + 2]
        x = max(r, min(w_final - r, x))
        y = max(r, min(h_final - r, y))
        r = max(1e-6, min(0.5 * min(w_final, h_final), r))
        result[i, 0] = x
        result[i, 1] = y
        result[i, 2] = r

    return result

# EVOLVE-BLOCK-END
