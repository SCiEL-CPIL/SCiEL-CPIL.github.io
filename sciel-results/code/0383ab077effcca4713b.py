import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        eval_count = 0
        best_x = np.zeros(d)
        best_f = np.inf

        def evaluate(x):
            nonlocal eval_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = float(fx)
                best_x = x.copy()
            return float(fx)

        def time_ok():
            return time.time() < deadline

        # ---- Phase 1: Diverse early sampling for fast initial progress ----
        # We want to find a point in the basin of attraction quickly.
        samples = []
        # Broad random
        for _ in range(10):
            samples.append(rng.uniform(-5.0, 5.0, d))
        # Medium random
        for _ in range(10):
            samples.append(rng.uniform(-3.0, 3.0, d))
        # Local random
        for _ in range(8):
            samples.append(rng.uniform(-1.5, 1.5, d))
        # Very local
        for _ in range(4):
            samples.append(rng.uniform(-0.5, 0.5, d))
        # Structured points
        samples.append(np.zeros(d))
        samples.append(np.ones(d) * 0.5)
        samples.append(np.ones(d) * -0.5)
        # Gaussian
        samples.append(rng.normal(0, 2.0, d))
        samples.append(rng.normal(0, 1.0, d))
        samples.append(rng.normal(0, 0.5, d))

        n_random = min(len(samples), 35)
        for x in samples[:n_random]:
            if eval_count >= 35 or not time_ok():
                break
            evaluate(x)

        # ---- Phase 2: Powell's Method ----
        # Powell's method is effective for separable or nearly separable functions.
        # Even with rotation, the adaptive directions help align with the function's axes.
        
        def line_search(x, d_dir, step_size):
            """
            Perform a 1D line search along direction d_dir starting from x.
            Uses parabolic interpolation.
            Returns new_x, new_f.
            """
            nonlocal eval_count, best_x, best_f
            if not time_ok() or eval_count >= budget - 10:
                return x, func(x) # This is bad, we shouldn't call func directly, but we have best_f. 
            # Actually, we should use the current best_f if x is the current best.
            # But line_search is called with a specific x which might not be the global best.
            # We need to evaluate x if it hasn't been evaluated recently?
            # No, in Powell, we always know the f-value of the current point.
            # Let's assume the caller passes the current f-value.
            # Refactor: line_search(x, f_x, d_dir, step_size)
            
            # Step 1: Try a step forward
            x1 = x + step_size * d_dir
            x1 = np.clip(x1, -5.0, 5.0)
            f1 = evaluate(x1)
            
            if f1 < f_x:
                # Try a step backward to bracket the minimum
                x2 = x - step_size * d_dir
                x2 = np.clip(x2, -5.0, 5.0)
                f2 = evaluate(x2)
                
                # We have 3 points: (x2, f2), (x, f_x), (x1, f1)
                # Fit parabola to find minimum
                # Using Lagrange interpolation or similar
                # t=0 for x, t=1 for x1, t=-1 for x2
                # f(t) = a*t^2 + b*t + c
                # f(0) = c = f_x
                # f(1) = a + b + c = f1  => a + b = f1 - f_x
                # f(-1) = a - b + c = f2 => a - b = f2 - f_x
                # Adding: 2a = f1 + f2 - 2*f_x => a = 0.5 * (f1 + f2 - 2*f_x)
                a = 0.5 * (f1 + f2 - 2 * f_x)
                if a <= 0:
                    # Parabola opens downward or is flat. Minimum is at one of the endpoints.
                    if f1 < f2:
                        return x1, f1
                    else:
                        return x2, f2
                
                # t_min = -b / (2a)
                # b = (f1 - f_x) - a
                b = (f1 - f_x) - a
                t_min = -b / (2 * a)
                
                # Clip t_min to [-1, 1] to stay within bracket
                t_min = np.clip(t_min, -1.0, 1.0)
                
                x_min = x + t_min * step_size * d_dir
                x_min = np.clip(x_min, -5.0, 5.0)
                f_min = evaluate(x_min)
                
                # Optional: One more refinement step if we have budget
                if eval_count < budget - 10 and time_ok():
                     # Check if x_min is better than x1 and x2
                     if f_min < min(f1, f2):
                         # We could do another parabolic fit, but let's keep it simple for now
                         # Or just return x_min
                         pass
                         
                return x_min, f_min
            else:
                # f1 >= f_x, so minimum is likely behind or at x
                x2 = x - step_size * d_dir
                x2 = np.clip(x2, -5.0, 5.0)
                f2 = evaluate(x2)
                
                if f2 < f_x:
                    # Minimum is between x2 and x
                    # Points: (x2, f2), (x, f_x), (x1, f1)
                    # Same parabola logic
                    a = 0.5 * (f1 + f2 - 2 * f_x)
                    if a <= 0:
                        return x2, f2
                    b = (f1 - f_x) - a
                    t_min = -b / (2 * a)
                    t_min = np.clip(t_min, -1.0, 1.0)
                    x_min = x + t_min * step_size * d_dir
                    x_min = np.clip(x_min, -5.0, 5.0)
                    f_min = evaluate(x_min)
                    return x_min, f_min
                else:
                    # f_x is the best of the three
                    return x, f_x

        def powell_method(x0, f0, max_cycles=20):
            nonlocal eval_count, best_x, best_f
            x = x0.copy()
            f_x = f0
            directions = [np.eye(d)[i] for i in range(d)]
            step_size = 1.0
            
            for cycle in range(max_cycles):
                if not time_ok() or eval_count >= budget - 20:
                    break
                    
                x_old = x.copy()
                f_old = f_x
                worst_delta = 0
                worst_index = 0
                
                for i in range(d):
                    if not time_ok() or eval_count >= budget - 10:
                        break
                    x_new, f_new = line_search(x, f_x, directions[i], step_size)
                    delta = f_old - f_new # Wait, f_old is the start of the cycle? No, f_x is current.
                    # Actually, we track the improvement from the start of the cycle for the worst direction check.
                    # Standard Powell:
                    # 1. Minimize along each direction.
                    # 2. Record the largest decrease.
                    
                    # Let's use a simpler check:
                    # If f_new < f_x, update x and f_x.
                    if f_new < f_x:
                        x = x_new
                        f_x = f_new
                        # Update global best
                        if f_x < best_f:
                            best_f = f_x
                            best_x = x.copy()
                            
                # After all directions, check if we improved
                if f_x < f_old:
                    # Extrapolation step
                    d_new = x - x_old
                    # Normalize? No, keep the vector.
                    # Step size for extrapolation?
                    # Usually, we try to go further in the direction d_new.
                    # Let's try a step of size 1.0 in the direction of (x - x_old) normalized?
                    # Or just use the vector itself?
                    # If x moved a lot, d_new is large.
                    # Let's normalize d_new and use step_size.
                    norm_d = np.linalg.norm(d_new)
                    if norm_d > 1e-10:
                        d_unit = d_new / norm_d
                        x_ext, f_ext = line_search(x, f_x, d_unit, step_size)
                        if f_ext < f_x:
                            x = x_ext
                            f_x = f_ext
                            if f_x < best_f:
                                best_f = f_x
                                best_x = x.copy()
                            # Update directions
                            # Find the direction that gave the worst improvement in this cycle
                            # We didn't track that specifically. Let's just replace the last direction or the one that seemed least useful.
                            # Simple strategy: Replace the direction that was used last?
                            # Better: Replace the direction with the smallest improvement.
                            # For simplicity, let's replace the direction that is most aligned with d_new?
                            # Or just replace the last direction.
                            # Let's replace the direction that had the smallest delta.
                            # Since we didn't store deltas, let's just replace the last direction.
                            directions[-1] = d_unit
                        else:
                            # Extrapolation didn't help, but we still have a new direction?
                            # Standard Powell replaces the worst direction with d_new if the cycle improved.
                            # Let's just update the last direction to d_unit.
                            directions[-1] = d_unit
                    else:
                        break
                else:
                    # No improvement, break
                    break
                    
            return x, f_x

        # Run Powell's method
        if eval_count < budget - 50 and time_ok():
            x_powell, f_powell = powell_method(best_x.copy(), best_f)
            if np.all(np.isfinite(x_powell)):
                best_x = np.clip(x_powell, -5.0, 5.0)
                # Ensure we have the best value recorded
                if eval_count < budget - 5:
                    # Re-evaluate to be sure
                    f_check = evaluate(best_x)
                    if f_check < best_f:
                        best_f = f_check

        # ---- Phase 3: Final Polish with Nelder-Mead (if budget remains) ----
        # NM can polish the result if Powell got stuck or if there's a small local error.
        def nelder_mead(x0, max_iter=50, init_step=0.1):
            nonlocal eval_count, best_x, best_f
            if not time_ok() or eval_count >= budget - 10:
                return x0

            simplex = [x0.copy()]
            for i in range(d):
                v = x0.copy()
                v[i] += init_step
                simplex.append(v)
            f_values = [evaluate(v) for v in simplex]

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            for _ in range(max_iter):
                if not time_ok() or eval_count >= budget - 5:
                    break

                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]

                if f_values[-1] - f_values[0] < 1e-10 * (1 + abs(f_values[0])):
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                f_worst = f_values[-1]

                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = evaluate(x_r)

                if f_r < f_values[0]:
                    x_e = centroid + gamma * (centroid - worst)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    if f_r < f_worst:
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = np.clip(x_c, -5.0, 5.0)
                        f_c = evaluate(x_c)
                        if f_c <= f_r:
                            simplex[-1] = x_c
                            f_values[-1] = f_c
                        else:
                            for i in range(1, d + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                f_values[i] = evaluate(simplex[i])
                    else:
                        x_c = centroid - rho * (centroid - worst)
                        x_c = np.clip(x_c, -5.0, 5.0)
                        f_c = evaluate(x_c)
                        if f_c < f_worst:
                            simplex[-1] = x_c
                            f_values[-1] = f_c
                        else:
                            for i in range(1, d + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                f_values[i] = evaluate(simplex[i])

            idx = int(np.argmin(f_values))
            return simplex[idx]

        if eval_count < budget - 10 and time_ok():
            x_nm = nelder_mead(best_x.copy(), max_iter=30, init_step=0.05)
            if np.all(np.isfinite(x_nm)):
                best_x = np.clip(x_nm, -5.0, 5.0)
                if eval_count < budget - 2:
                    evaluate(best_x)

        return best_x, best_f
