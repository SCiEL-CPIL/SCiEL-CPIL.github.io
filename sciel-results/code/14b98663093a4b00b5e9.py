import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def evaluate(pt):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or time.time() > deadline:
                return best_f
            pt = np.asarray(pt, dtype=np.float64)
            if pt.shape != (dim,):
                pt = pt.reshape(dim)
            val = func(pt)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = pt.copy()
            return val

        # Phase 1: Exact quadratic model fitting
        # f(x) = a*||x||^2 + q^T x + c
        # We use a moderate step size d for the initial fit to avoid numerical issues with too small steps
        # but large enough to be robust.
        d = 2.0
        # Add a few random points to improve robustness against numerical issues
        rng = np.random.default_rng(42)
        extra_points = [rng.uniform(-4, 4, dim) for _ in range(5)]
        points = [np.zeros(dim)]
        for i in range(dim):
            p1 = np.zeros(dim); p1[i] = d
            p2 = np.zeros(dim); p2[i] = -d
            points.append(p1)
            points.append(p2)
        points.extend(extra_points)

        values = np.array([evaluate(p) for p in points])

        c = values[0]
        a_vals = []
        q_vec = np.zeros(dim)

        for i in range(dim):
            val_pos = values[1 + 2*i]
            val_neg = values[2 + 2*i]
            a_i = (val_pos + val_neg - 2*c) / (2 * d * d)
            q_i = (val_pos - val_neg) / (2 * d)
            a_vals.append(a_i)
            q_vec[i] = q_i

        # Use least-squares fit with all points for better robustness
        # For each point x, f(x) = a*||x||^2 + q^T x + c
        # We can set up a linear system: [||x||^2, x_1, ..., x_n, 1] * [a, q_1, ..., q_n, c]^T = f(x)
        n_total = len(points)
        A = np.zeros((n_total, dim + 2))
        b = values
        for idx, p in enumerate(points):
            A[idx, 0] = np.dot(p, p)
            A[idx, 1:dim+1] = p
            A[idx, dim+1] = 1.0
        
        # Solve least squares
        coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        a = coeffs[0]
        q_vec = coeffs[1:dim+1]

        if abs(a) < 1e-12:
            a = 1e-12 if a >= 0 else -1e-12

        # Estimated minimum
        x_opt = -q_vec / (2 * a)
        x_opt = np.clip(x_opt, -5, 5)

        f_opt_est = evaluate(x_opt)
        
        # If the quadratic fit gives a poor result, fall back to random search
        if f_opt_est > best_f + 1.0:
            # Do some random sampling
            for _ in range(20):
                if calls_made >= budget or time.time() > deadline:
                    break
                x_rand = rng.uniform(-4, 4, dim)
                f_rand = evaluate(x_rand)

        # Phase 2: Iterative refinement with decreasing step sizes
        # For a pure quadratic, one refinement should be exact, but numerical
        # errors may require a second pass with a much smaller step.
        x = best_x.copy()
        fx = best_f
        
        for d_refine in [0.1, 0.01, 0.001]:
            if calls_made + (2 * dim + 2) > budget or time.time() > deadline:
                break
                
            points_refine = [x]
            for i in range(dim):
                p1 = x.copy(); p1[i] += d_refine
                p2 = x.copy(); p2[i] -= d_refine
                points_refine.append(p1)
                points_refine.append(p2)
            
            values_refine = np.array([evaluate(p) for p in points_refine])
            c_ref = values_refine[0]
            a_vals_ref = []
            q_vec_ref = np.zeros(dim)
            
            for i in range(dim):
                val_pos = values_refine[1 + 2*i]
                val_neg = values_refine[2 + 2*i]
                a_i = (val_pos + val_neg - 2*c_ref) / (2 * d_refine * d_refine)
                q_i = (val_pos - val_neg) / (2 * d_refine)
                a_vals_ref.append(a_i)
                q_vec_ref[i] = q_i
            
            a_ref = np.mean(a_vals_ref)
            
            if abs(a_ref) > 1e-14:
                x_opt_ref = x - q_vec_ref / (2 * a_ref)
                x_opt_ref = np.clip(x_opt_ref, -5, 5)
                f_ref = evaluate(x_opt_ref)
                if f_ref < fx:
                    x = x_opt_ref
                    fx = f_ref
                else:
                    # If the refinement didn't help, try a smaller step
                    break

        # Phase 3: Final polish with very small steps if budget allows
        # This catches any residual error from the quadratic fit
        if calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            
            for d_polish in [0.0001, 0.00001]:
                if calls_made + (2 * dim + 2) > budget or time.time() > deadline:
                    break
                    
                points_polish = [x]
                for i in range(dim):
                    p1 = x.copy(); p1[i] += d_polish
                    p2 = x.copy(); p2[i] -= d_polish
                    points_polish.append(p1)
                    points_polish.append(p2)
                
                values_polish = np.array([evaluate(p) for p in points_polish])
                c_pol = values_polish[0]
                a_vals_pol = []
                q_vec_pol = np.zeros(dim)
                
                for i in range(dim):
                    val_pos = values_polish[1 + 2*i]
                    val_neg = values_polish[2 + 2*i]
                    a_i = (val_pos + val_neg - 2*c_pol) / (2 * d_polish * d_polish)
                    q_i = (val_pos - val_neg) / (2 * d_polish)
                    a_vals_pol.append(a_i)
                    q_vec_pol[i] = q_i
                
                a_pol = np.mean(a_vals_pol)
                
                if abs(a_pol) > 1e-14:
                    x_opt_pol = x - q_vec_pol / (2 * a_pol)
                    x_opt_pol = np.clip(x_opt_pol, -5, 5)
                    f_pol = evaluate(x_opt_pol)
                    if f_pol < fx:
                        x = x_opt_pol
                        fx = f_pol
                    else:
                        break

        return best_x, best_f
