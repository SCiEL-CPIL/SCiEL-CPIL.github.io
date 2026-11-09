# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 610

    def elapsed():
        return time.time() - start_time

    def is_valid(centers, sides, tol=1e-7):
        n = len(sides)
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0 - tol
                if dx < t and dy < t:
                    return False
        return True

    best_centers = None
    best_sides = None
    best_score = -np.inf

    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if not is_valid(centers, sides):
            return
        s = np.sum(sides)
        if s > best_score:
            best_score = s
            best_centers = centers.copy()
            best_sides = sides.copy()

    # ============================================================
    # Vectorized smooth penalty objective
    # ============================================================
    def smooth_penalty(params, beta=50.0):
        centers = params[:2*n].reshape(n, 2)
        sides = params[2*n:]

        def softplus(x, b):
            return np.log1p(np.exp(-b * x)) / b

        obj = -np.sum(sides)
        W = 100.0

        # Boundary penalties (vectorized)
        v_l = -(centers[:, 0] - sides/2)
        v_r = (centers[:, 0] + sides/2) - 1
        v_b = -(centers[:, 1] - sides/2)
        v_t = (centers[:, 1] + sides/2) - 1
        obj += W * (np.sum(softplus(v_l, beta)) + np.sum(softplus(v_r, beta)) +
                    np.sum(softplus(v_b, beta)) + np.sum(softplus(v_t, beta)))

        # Overlap penalties (vectorized over pairs)
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0 - 1e-7
                if t <= 0:
                    continue
                a = t - dx
                b = t - dy
                a_c = np.clip(a, -50/beta, 50/beta)
                b_c = np.clip(b, -50/beta, 50/beta)
                sm = -np.log(np.exp(-beta * a_c) + np.exp(-beta * b_c)) / beta
                obj += W * softplus(sm, beta)

        return obj

    # ============================================================
    # Build initial topologies
    # ============================================================
    starts = []

    # Topology 1: 2 large (0.5) top, 8 small (0.25) bottom 2x4
    c1 = np.zeros((n, 2))
    s1 = np.zeros(n)
    a, b = 0.5, 0.25
    c1[0] = [a/2, 1 - a/2]; s1[0] = a
    c1[1] = [1 - a/2, 1 - a/2]; s1[1] = a
    for r in range(2):
        for col in range(4):
            idx = 2 + r*4 + col
            c1[idx, 0] = b/2 + col*b
            c1[idx, 1] = b/2 + r*b
            s1[idx] = b
    update_best(c1, s1)
    starts.append((c1.copy(), s1.copy()))

    # Topology 2: 1 large (0.5) left, 9 in right 0.5x1 region
    c2 = np.zeros((n, 2))
    s2 = np.zeros(n)
    c2[0] = [0.25, 0.5]; s2[0] = 0.5
    b2 = 0.5/3
    for r in range(3):
        for col in range(3):
            idx = 1 + r*3 + col
            c2[idx, 0] = 0.5 + b2/2 + col*b2
            c2[idx, 1] = b2/2 + r*b2
            s2[idx] = b2
    update_best(c2, s2)
    starts.append((c2.copy(), s2.copy()))

    # Topology 3: 3 large (1/3) in a row at top, 7 small below
    c3 = np.zeros((n, 2))
    s3 = np.zeros(n)
    a3 = 1.0/3
    for i in range(3):
        c3[i, 0] = a3/2 + i*a3
        c3[i, 1] = 1 - a3/2
        s3[i] = a3
    b3a = 0.25
    b3b = 1.0/3
    for col in range(4):
        idx = 3 + col
        c3[idx, 0] = b3a/2 + col*b3a
        c3[idx, 1] = (2.0/3) - b3a/2
        s3[idx] = b3a
    for col in range(3):
        idx = 7 + col
        c3[idx, 0] = b3b/2 + col*b3b
        c3[idx, 1] = b3b/2
        s3[idx] = b3b
    update_best(c3, s3)
    starts.append((c3.copy(), s3.copy()))

    # Topology 4: Uniform 2x5 grid
    c4 = np.zeros((n, 2))
    s4 = np.zeros(n)
    for i in range(n):
        r = i // 5
        c = i % 5
        c4[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        s4[i] = 0.2
    update_best(c4, s4)
    starts.append((c4.copy(), s4.copy()))

    # Topology 5: 4 corners + 6 center
    c5 = np.zeros((n, 2))
    s5 = np.zeros(n)
    a5 = 0.3
    b5 = (1 - 2*a5)/3
    c5[0] = [a5/2, a5/2]; s5[0] = a5
    c5[1] = [1-a5/2, a5/2]; s5[1] = a5
    c5[2] = [a5/2, 1-a5/2]; s5[2] = a5
    c5[3] = [1-a5/2, 1-a5/2]; s5[3] = a5
    for r in range(2):
        for col in range(3):
            idx = 4 + r*3 + col
            c5[idx, 0] = a5 + b5/2 + col*b5
            c5[idx, 1] = a5 + b5/2 + r*b5
            s5[idx] = b5
    update_best(c5, s5)
    starts.append((c5.copy(), s5.copy()))

    # Topology 6: 5+5 split
    c6 = np.zeros((n, 2))
    s6 = np.zeros(n)
    for r in range(2):
        for col in range(2):
            idx = r*2 + col
            c6[idx, 0] = 0.125 + col*0.25
            c6[idx, 1] = 0.125 + r*0.25
            s6[idx] = 0.25
    c6[4, 0] = 0.25; c6[4, 1] = 0.75; s6[4] = 0.25
    for r in range(2):
        for col in range(2):
            idx = 5 + r*2 + col
            c6[idx, 0] = 0.625 + col*0.25
            c6[idx, 1] = 0.125 + r*0.25
            s6[idx] = 0.25
    c6[9, 0] = 0.75; c6[9, 1] = 0.75; s6[9] = 0.25
    update_best(c6, s6)
    starts.append((c6.copy(), s6.copy()))

    # Topology 7: 2x5 but with varying sizes - top row larger
    c7 = np.zeros((n, 2))
    s7 = np.zeros(n)
    # Top row: 5 squares of size 0.2
    # Bottom row: 5 squares of size 0.2
    # Try asymmetric: top 3 of 0.3, bottom 7 split
    a7 = 0.3
    for i in range(3):
        c7[i, 0] = a7/2 + i*a7
        c7[i, 1] = 1 - a7/2
        s7[i] = a7
    # Remaining 7 in bottom 0.7 height, 1 width
    # 4 of 0.25 + 3 of 0.25? No, 7 squares
    # Try: 4 of 0.2 in top sub-row, 3 of 0.233 in bottom
    b7a = 0.2
    for col in range(4):
        idx = 3 + col
        c7[idx, 0] = b7a/2 + col*b7a
        c7[idx, 1] = 0.7 - b7a/2
        s7[idx] = b7a
    b7b = 1.0/3
    for col in range(3):
        idx = 7 + col
        c7[idx, 0] = b7b/2 + col*b7b
        c7[idx, 1] = b7b/2
        s7[idx] = b7b
    update_best(c7, s7)
    starts.append((c7.copy(), s7.copy()))

    # Topology 8: 4 large (0.4) in corners, 6 small in center
    c8 = np.zeros((n, 2))
    s8 = np.zeros(n)
    a8 = 0.4
    c8[0] = [a8/2, a8/2]; s8[0] = a8
    c8[1] = [1-a8/2, a8/2]; s8[1] = a8
    c8[2] = [a8/2, 1-a8/2]; s8[2] = a8
    c8[3] = [1-a8/2, 1-a8/2]; s8[3] = a8
    # Center region: (1-2*0.4) x (1-2*0.4) = 0.2 x 0.2
    # 6 squares in 0.2x0.2: 2x3 grid of 0.2/3 ≈ 0.0667
    b8 = 0.2/3
    for r in range(2):
        for col in range(3):
            idx = 4 + r*3 + col
            c8[idx, 0] = a8 + b8/2 + col*b8
            c8[idx, 1] = a8 + b8/2 + r*b8
            s8[idx] = b8
    update_best(c8, s8)
    starts.append((c8.copy(), s8.copy()))

    # Add perturbed versions
    np.random.seed(42)
    for base_c, base_s in starts[:4]:
        for _ in range(4):
            c_p = base_c + np.random.normal(0, 0.02, (n, 2))
            s_p = base_s + np.random.normal(0, 0.01, n)
            s_p = np.clip(s_p, 0.05, 0.55)
            starts.append((c_p, s_p))

    # ============================================================
    # Differential Evolution global search
    # ============================================================
    def de_search(num_gen=30, pop_size=40, F=0.8, CR=0.9):
        """Differential evolution to find good starting points."""
        np.random.seed(12345)
        
        # Bounds: centers in [0,1], sides in [0.01, 0.5]
        lb = np.concatenate([np.zeros(2*n), np.full(n, 0.01)])
        ub = np.concatenate([np.ones(2*n), np.full(n, 0.5)])
        
        # Initialize population
        pop = np.random.uniform(lb, ub, (pop_size, 3*n))
        
        # Evaluate initial population
        scores = np.array([smooth_penalty(p, beta=50.0) for p in pop])
        best_idx = np.argmin(scores)
        best_params = pop[best_idx].copy()
        best_score_de = scores[best_idx]
        
        for gen in range(num_gen):
            if elapsed() > 300:
                break
            
            new_pop = pop.copy()
            for i in range(pop_size):
                # Select three distinct individuals
                a_idx, b_idx, c_idx = np.random.choice(pop_size, 3, replace=False)
                while a_idx == i or b_idx == i or c_idx == i:
                    a_idx, b_idx, c_idx = np.random.choice(pop_size, 3, replace=False)
                
                # Mutation
                v = pop[a_idx] + F * (pop[b_idx] - pop[c_idx])
                v = np.clip(v, lb, ub)
                
                # Crossover
                trial = np.where(np.random.random(3*n) < CR, v, pop[i])
                trial = np.clip(trial, lb, ub)
                
                # Selection
                trial_score = smooth_penalty(trial, beta=50.0)
                if trial_score < scores[i]:
                    new_pop[i] = trial
                    scores[i] = trial_score
            
            pop = new_pop
            gen_best_idx = np.argmin(scores)
            if scores[gen_best_idx] < best_score_de:
                best_score_de = scores[gen_best_idx]
                best_params = pop[gen_best_idx].copy()
            
            # Track valid solutions
            c_try = best_params[:2*n].reshape(n, 2)
            s_try = np.maximum(best_params[2*n:], 1e-6)
            update_best(c_try, s_try)
        
        return best_params

    # Run DE if we have time
    if elapsed() < 250:
        de_params = de_search(num_gen=25, pop_size=35, F=0.8, CR=0.9)
        c_de = de_params[:2*n].reshape(n, 2)
        s_de = np.maximum(de_params[2*n:], 1e-6)
        update_best(c_de, s_de)
        starts.append((c_de.copy(), s_de.copy()))

    # ============================================================
    # Optimization loop with adaptive beta
    # ============================================================
    for start_idx, (c_init, s_init) in enumerate(starts):
        if elapsed() > 560:
            break

        params_init = np.concatenate([c_init.flatten(), s_init])

        # Multi-stage: coarse to fine
        for stage, (beta, maxiter, ftol) in enumerate([(30.0, 200, 1e-8),
                                                        (80.0, 300, 1e-10),
                                                        (200.0, 200, 1e-12)]):
            if elapsed() > 570:
                break
            try:
                res = minimize(smooth_penalty, params_init, args=(beta,),
                               method='SLSQP',
                               options={'maxiter': maxiter, 'ftol': ftol, 'disp': False})

                params_opt = res.x
                c_opt = params_opt[:2*n].reshape(n, 2)
                s_opt = np.maximum(params_opt[2*n:], 1e-6)

                update_best(c_opt, s_opt)
                params_init = params_opt

            except Exception:
                break

        # Final high-precision polish
        if elapsed() < 580:
            try:
                c_pol = params_init[:2*n].reshape(n, 2)
                s_pol = np.maximum(params_init[2*n:], 1e-6)
                if is_valid(c_pol, s_pol):
                    res_pol = minimize(smooth_penalty, params_init, args=(500.0,),
                                       method='SLSQP',
                                       options={'maxiter': 500, 'ftol': 1e-14, 'disp': False})
                    params_pol = res_pol.x
                    c_pol = params_pol[:2*n].reshape(n, 2)
                    s_pol = np.maximum(params_pol[2*n:], 1e-6)
                    update_best(c_pol, s_pol)
            except Exception:
                pass

    # ============================================================
    # Local search: small random perturbations around best
    # ============================================================
    if best_centers is not None and elapsed() < 590:
        np.random.seed(999)
        for _ in range(150):
            if elapsed() > 600:
                break
            c_try = best_centers + np.random.normal(0, 0.005, (n, 2))
            s_try = best_sides + np.random.normal(0, 0.003, n)
            s_try = np.clip(s_try, 0.01, 0.6)
            params_try = np.concatenate([c_try.flatten(), s_try])
            try:
                res = minimize(smooth_penalty, params_try, args=(100.0,),
                               method='SLSQP',
                               options={'maxiter': 100, 'ftol': 1e-10, 'disp': False})
                c_r = res.x[:2*n].reshape(n, 2)
                s_r = np.maximum(res.x[2*n:], 1e-6)
                update_best(c_r, s_r)
            except Exception:
                pass

    if best_centers is None:
        best_centers = c1
        best_sides = s1

    return best_centers, best_sides

# EVOLVE-BLOCK-END
