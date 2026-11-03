"""
Improved sphere packing: Vectorized smooth surrogate + aggressive local search.
Uses fully vectorized log-sum-exp soft-min gradients for speed, allowing for
more iterations and restarts. Includes a specialized "critical constraint"
polish that focuses on the active constraints (closest pairs/walls).
"""
import numpy as np
import time
import scipy.optimize as opt
import scipy.special as sp

N = 50
CUBE_HALF = 0.5
DEADLINE_SECONDS = 145.0  # Safety margin before 150 s kill

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def remaining():
        return DEADLINE_SECONDS - elapsed()

    # ------------------------------------------------------------------
    # Exact radius
    # ------------------------------------------------------------------
    def compute_radius(positions):
        positions = np.asarray(positions, dtype=np.float64).reshape(N, 3)
        if not np.all(np.isfinite(positions)):
            return 0.0
        if np.any(np.abs(positions) > 0.5 + 1e-9):
            return 0.0
        
        # Pairwise distances
        diff = positions[:, None, :] - positions[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
        
        # Wall distances
        walls = np.min(0.5 - np.abs(positions), axis=1)
        min_wall = np.min(walls)
        
        return float(min(min_pair, min_wall))

    # ------------------------------------------------------------------
    # Vectorized Smooth Surrogate with Analytic Gradient
    # ------------------------------------------------------------------
    tri_i, tri_j = np.triu_indices(N, k=1)
    n_pairs = len(tri_i)
    
    # Precompute indices for wall gradients
    # Walls: 0.5 - c, c + 0.5 for each dim
    # We want grad of -soft_min(g) where g are the distances.
    # f = -1/beta * log(sum(exp(-beta*g)))
    # df/dc = sum_k w_k * (1/beta) * (-beta * dg_k/dc) = - sum_k w_k * dg_k/dc
    # Wait, we want to MAXIMIZE min(g). So we MINIMIZE -min(g).
    # Let F = -soft_min(g). We want to minimize F.
    # grad F = - grad soft_min(g)
    # grad soft_min(g) = sum w_k * grad g_k
    # So grad F = - sum w_k * grad g_k

    def smooth_obj_grad(x_flat, beta):
        """Return (value, grad_flat) for F = -soft_min(g)."""
        c = x_flat.reshape(N, 3)

        # Pairwise distances / 2
        diff = c[tri_i] - c[tri_j]
        norms = np.linalg.norm(diff, axis=1)
        norms_safe = np.where(norms < 1e-12, 1e-12, norms)
        g_pairs = 0.5 * norms_safe

        # Wall distances (6 per point)
        # g_walls shape (6*N, )
        # Order: x-, x+, y-, y+, z-, z+
        g_walls = np.concatenate([
            0.5 - c[:, 0], c[:, 0] + 0.5,
            0.5 - c[:, 1], c[:, 1] + 0.5,
            0.5 - c[:, 2], c[:, 2] + 0.5,
        ])

        g_all = np.concatenate([g_pairs, g_walls])

        # Soft min  f = -(1/beta) log sum exp(-beta g)
        log_sum_exp = sp.logsumexp(-beta * g_all)
        val = -log_sum_exp / beta

        # Weights  w_k = exp(-beta g_k) / sum exp(-beta g)
        log_w = -beta * g_all - log_sum_exp
        weights = np.exp(log_w)

        # Gradient of soft_min(g) w.r.t. c
        # grad_soft_min = sum w_k * grad g_k
        # We want grad of F = -soft_min, so grad_F = -grad_soft_min
        
        grad_c = np.zeros_like(c)

        # Pairwise part:
        # g_k = 0.5 * ||c_i - c_j||
        # dg_k/dc_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
        # dg_k/dc_j = 0.5 * (c_j - c_i) / ||c_i - c_j||
        # contrib to grad_soft_min:
        # grad_c[i] += w_k * 0.5 * (c_i - c_j) / norm
        # grad_c[j] += w_k * 0.5 * (c_j - c_i) / norm
        
        # Vectorized accumulation
        # factor_k = 0.5 * w_k / norm_k
        factors = 0.5 * weights[:n_pairs] / norms_safe
        # diff[k] is c[tri_i[k]] - c[tri_j[k]]
        # grad_c[tri_i] += factors[:, None] * diff
        # grad_c[tri_j] -= factors[:, None] * diff
        
        np.add.at(grad_c, tri_i, factors[:, None] * diff)
        np.add.at(grad_c, tri_j, -factors[:, None] * diff)

        # Wall part:
        # g_x_minus = 0.5 - c_x  -> dg/dc_x = -1
        # g_x_plus = c_x + 0.5   -> dg/dc_x = +1
        # grad_soft_min[c_x] += w_x_minus * (-1) + w_x_plus * (+1)
        
        ws = weights[n_pairs:]
        # ws[0:N] are x_minus, ws[N:2N] are x_plus, etc.
        
        # grad_c[:, 0] += -ws[0:N] + ws[N:2*N]
        # grad_c[:, 1] += -ws[2*N:3*N] + ws[3*N:4*N]
        # grad_c[:, 2] += -ws[4*N:5*N] + ws[5*N:6*N]
        
        grad_c[:, 0] += -ws[0:N] + ws[N:2*N]
        grad_c[:, 1] += -ws[2*N:3*N] + ws[3*N:4*N]
        grad_c[:, 2] += -ws[4*N:5*N] + ws[5*N:6*N]

        # We minimize F = -soft_min, so grad_F = -grad_soft_min
        return val, (-grad_c).flatten()

    # ------------------------------------------------------------------
    # Local optimisation routines
    # ------------------------------------------------------------------
    bounds = [(-0.499999, 0.499999)] * (N * 3)

    def local_smooth(c_init, beta, maxiter=200):
        if remaining() < 5:
            return c_init
        x0 = c_init.flatten()

        def f(x):
            v, _ = smooth_obj_grad(x, beta)
            return v

        def g(x):
            _, gr = smooth_obj_grad(x, beta)
            return gr

        try:
            res = opt.minimize(f, x0, jac=g, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            return np.clip(res.x.reshape(N, 3), -0.499999, 0.499999)
        except Exception:
            return c_init

    def local_direct(c_init, maxiter=300):
        if remaining() < 5:
            return c_init
        x0 = c_init.flatten()

        def f(x):
            return -compute_radius(x.reshape(N, 3))

        try:
            res = opt.minimize(f, x0, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50})
            return np.clip(res.x.reshape(N, 3), -0.499999, 0.499999)
        except Exception:
            return c_init

    def critical_polish(c_init, maxiter=100):
        """
        Polish by focusing on the active constraints.
        We compute the gradient of the exact min-radius using subgradients
        but with a step size that adapts to the violation.
        """
        if remaining() < 5:
            return c_init
        pos = c_init.copy()
        lr = 0.005
        for it in range(maxiter):
            if remaining() < 2:
                break
            
            # Compute radius and active constraints
            diff = pos[:, None, :] - pos[None, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            min_pair_val = np.min(d) / 2.0
            walls = 0.5 - np.abs(pos)
            min_wall_val = np.min(walls)
            
            r = min(min_pair_val, min_wall_val)
            
            # If we are stuck, reduce learning rate
            if it > 20:
                lr *= 0.95
                
            grad = np.zeros_like(pos)
            
            # Find all pairs within a small epsilon of the min distance
            # This provides a smoother "average" subgradient
            eps = 1e-4
            close_pairs = np.argwhere(np.abs(d/2.0 - min_pair_val) < eps)
            if len(close_pairs) > 0:
                for i, j in close_pairs:
                    if i == j: continue
                    dv = pos[i] - pos[j]
                    dn = np.linalg.norm(dv)
                    if dn > 1e-12:
                        u = dv / dn
                        grad[i] += 0.5 * u
                        grad[j] -= 0.5 * u
            
            close_walls = np.argwhere(np.abs(walls - min_wall_val) < eps)
            if len(close_walls) > 0:
                for i, dim in close_walls:
                    if pos[i, dim] > 0:
                        grad[i, dim] -= 1.0
                    else:
                        grad[i, dim] += 1.0
            
            # Normalize gradient to prevent huge steps
            g_norm = np.linalg.norm(grad)
            if g_norm > 1e-12:
                grad /= g_norm
            
            pos = np.clip(pos + lr * grad, -0.499999, 0.499999)
            
            # Check for convergence
            if g_norm < 1e-6:
                break
                
        return pos

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def make_bcc():
        pts = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    pts.append([x, y, z])
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    pts.append([x + 0.5, y + 0.5, z + 0.5])
        arr = np.array(pts) - np.array([1.5, 1.5, 1.5])
        d = np.linalg.norm(arr, axis=1)
        sel = arr[np.argsort(d)[:N]]
        mx = np.max(np.linalg.norm(sel, axis=1))
        if mx > 0:
            sel *= 0.45 / mx
        return sel

    def make_fcc():
        # FCC lattice points
        pts = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    pts.append([x, y, z])
        # Face centers
        for x in range(4):
            for y in range(3):
                pts.append([x, y + 0.5, 0])
                pts.append([x, y + 0.5, 3])
            for z in range(3):
                pts.append([x, 0, z + 0.5])
                pts.append([x, 3, z + 0.5])
        for y in range(3):
            for z in range(3):
                pts.append([0, y + 0.5, z + 0.5])
                pts.append([3, y + 0.5, z + 0.5])
        
        arr = np.array(pts) - np.array([1.5, 1.5, 1.5])
        d = np.linalg.norm(arr, axis=1)
        sel = arr[np.argsort(d)[:N]]
        mx = np.max(np.linalg.norm(sel, axis=1))
        if mx > 0:
            sel *= 0.45 / mx
        return sel

    def make_grid():
        m = 4
        t = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in t for y in t for x in t])
        return grid[:N].copy()

    def make_random(seed):
        return np.random.RandomState(seed).uniform(-0.45, 0.45, size=(N, 3))

    # ------------------------------------------------------------------
    # Main search loop
    # ------------------------------------------------------------------
    best_pos = None
    best_r = -1.0

    def update(pos):
        nonlocal best_pos, best_r
        r = compute_radius(pos)
        if r > best_r:
            best_r = r
            best_pos = pos.copy()
        return r

    # --- Multi-start with smooth + direct + critical polish ---
    # Use diverse initializations
    inits = [
        make_bcc(), 
        make_fcc(), 
        make_grid(),
        make_random(0), 
        make_random(42), 
        make_random(123), 
        make_random(777),
        make_random(1000),
        make_random(2000)
    ]

    for init in inits:
        if remaining() < 10:
            break
        pos = np.clip(init, -0.499999, 0.499999)

        # Beta-annealed smooth optimization
        # Start with low beta (smooth) to escape bad local minima,
        # then increase beta to sharpen the minimum.
        if remaining() > 20:
            for beta in [2.0, 5.0, 10.0, 20.0, 40.0]:
                if remaining() < 5:
                    break
                pos = local_smooth(pos, beta, maxiter=100)

        # Direct L-BFGS-B polish on the exact non-smooth objective
        if remaining() > 10:
            pos = local_direct(pos, maxiter=200)

        # Critical constraint polish
        if remaining() > 5:
            pos = critical_polish(pos, maxiter=50)

        update(pos)

    # --- Perturbation refinement around best ---
    if best_pos is not None and remaining() > 20:
        for k in range(10):
            if remaining() < 5:
                break
            rng = np.random.RandomState(5000 + k)
            # Decreasing perturbation scale
            scale = 0.02 * (1.0 - k / 12.0)
            pert = np.clip(best_pos + rng.normal(0, scale, size=best_pos.shape),
                           -0.499999, 0.499999)

            if remaining() > 15:
                # Quick smooth pass
                for beta in [10.0, 30.0]:
                    if remaining() < 3:
                        break
                    pert = local_smooth(pert, beta, maxiter=50)

            if remaining() > 8:
                pert = local_direct(pert, maxiter=100)

            if remaining() > 4:
                pert = critical_polish(pert, maxiter=30)

            update(pert)

    # --- Final high-precision polish ---
    if best_pos is not None and remaining() > 10:
        try:
            fp = local_direct(best_pos, maxiter=300)
            fp = critical_polish(fp, maxiter=100)
            update(fp)
        except Exception:
            pass

    # --- Fallback ---
    if best_pos is None:
        best_pos = np.clip(make_bcc(), -0.499999, 0.499999)

    return best_pos
