# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Optimizes the 1D base positions to minimize the union volume.
    Uses a hybrid strategy:
    1. Smooth L-BFGS-B optimization on a softplus-approximated volume.
    2. Exact Nelder-Mead polish on the true piecewise-linear volume.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 30.0

    def _union_length_at_c(z, c):
        """
        Compute the exact union length of intervals [z[i] + c*i/N, z[i] + c*i/N + 1/N].
        """
        m = np.arange(N)
        starts = z + c * m / N
        ends = starts + 1.0 / N
        
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        
        L = e_sorted[-1] - s_sorted[0]
        
        for k in range(N - 1):
            gap = s_sorted[k+1] - e_sorted[k]
            if gap > 0:
                L -= gap
        
        return L

    def _union_length_grad_at_c(z, c):
        """
        Compute the gradient of union length w.r.t z at a given c.
        """
        m = np.arange(N)
        starts = z + c * m / N
        ends = starts + 1.0 / N
        
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        
        grad_z = np.zeros(N)
        grad_z[order[-1]] += 1.0
        grad_z[order[0]] -= 1.0
        
        for k in range(N - 1):
            gap = s_sorted[k+1] - e_sorted[k]
            if gap > 0:
                grad_z[order[k+1]] -= 1.0
                grad_z[order[k]] += 1.0
        
        return grad_z

    def _find_break_points(z):
        """
        Find all c values in [0,1] where the ordering of interval starts/ends changes.
        A break occurs when start_i(c) = start_j(c) or start_i(c) = end_j(c) or end_i(c) = end_j(c).
        start_i(c) = z[i] + c*i/N
        end_i(c) = z[i] + c*i/N + 1/N
        """
        breaks = set([0.0, 1.0])
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # start_i(c) = start_j(c): z[i] + c*i/N = z[j] + c*j/N
                # c*(i-j)/N = z[j] - z[i]
                if i != j:
                    c_val = (z[j] - z[i]) * N / (i - j)
                    if 0.0 <= c_val <= 1.0:
                        breaks.add(c_val)
                
                # start_i(c) = end_j(c): z[i] + c*i/N = z[j] + c*j/N + 1/N
                # c*(i-j)/N = z[j] + 1/N - z[i]
                if i != j:
                    c_val = (z[j] + 1.0/N - z[i]) * N / (i - j)
                    if 0.0 <= c_val <= 1.0:
                        breaks.add(c_val)
                
                # end_i(c) = end_j(c): z[i] + c*i/N + 1/N = z[j] + c*j/N + 1/N
                # c*(i-j)/N = z[j] - z[i] (same as start_i = start_j)
                # Already covered above
        
        return sorted(breaks)

    def volume_exact(z):
        """
        Compute the exact volume by integrating L(c)^2 over c in [0,1].
        L(c) is piecewise linear, so L(c)^2 is piecewise quadratic.
        """
        breaks = _find_break_points(z)
        
        total = 0.0
        for idx in range(len(breaks) - 1):
            c0 = breaks[idx]
            c1 = breaks[idx + 1]
            dc = c1 - c0
            if dc <= 0:
                continue
            
            # Sample at midpoint to determine the linear pieces
            c_mid = 0.5 * (c0 + c1)
            L_mid = _union_length_at_c(z, c_mid)
            
            # For piecewise linear L(c), L(c)^2 is quadratic
            # We need L at c0, c_mid, c1 to integrate exactly
            L0 = _union_length_at_c(z, c0)
            L1 = _union_length_at_c(z, c1)
            
            # Simpson's rule for quadratic: integral = dc/6 * (f0 + 4*f_mid + f1)
            total += dc / 6.0 * (L0*L0 + 4.0*L_mid*L_mid + L1*L1)
        
        return total

    def volume_smooth(z, eps=0.01):
        """
        Compute a smooth approximation of the volume and gradient using softplus.
        """
        # Use adaptive quadrature points
        c_quad, w_quad = np.polynomial.legendre.leggauss(100)
        c_quad = 0.5 * (c_quad + 1)
        w_quad = 0.5 * w_quad
        
        m = np.arange(N)
        cost = 0.0
        grad = np.zeros(N)
        
        for c, w in zip(c_quad, w_quad):
            starts = z + c * m / N
            ends = starts + 1.0 / N
            
            order = np.argsort(starts)
            s_sorted = starts[order]
            e_sorted = ends[order]
            
            gaps = s_sorted[1:] - e_sorted[:-1]
            L = e_sorted[-1] - s_sorted[0] - np.sum(eps * np.logaddexp(0, gaps / eps))
            
            grad_c = np.zeros(N)
            grad_c[order[-1]] += 1.0
            grad_c[order[0]] -= 1.0
            
            sig = 1.0 / (1.0 + np.exp(-gaps / eps))
            np.add.at(grad_c, order[1:], -sig)
            np.add.at(grad_c, order[:-1], sig)
            
            cost += w * L * L
            grad += w * 2.0 * L * grad_c
        
        return cost, grad

    # Initial guess: Keich binary construction
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)
    
    # Shift so z_0 = 0
    keich = keich - keich[0]
    x0 = keich[1:].copy()
    
    best_z = x0.copy()
    z_full_init = np.zeros(N)
    z_full_init[1:] = best_z
    best_cost = volume_exact(z_full_init)
    
    rng = np.random.default_rng(12345)
    
    # Optimization loop with exact volume for Nelder-Mead
    def exact_cost_params(params):
        z = np.zeros(N)
        z[1:] = params
        return volume_exact(z)
    
    def cost_smooth_params(params, eps):
        z = np.zeros(N)
        z[1:] = params
        cost, grad_full = volume_smooth(z, eps)
        return cost, grad_full[1:]

    # Try multiple restarts with emphasis on exact Nelder-Mead
    for run in range(40):
        if time.time() > deadline - safety_margin:
            break
        
        if run == 0:
            start = best_z.copy()
        else:
            # Random restart with adaptive noise
            scale = 0.02 + 0.01 * (run % 7)
            start = best_z + rng.normal(0, scale, N - 1)
        
        current_z = start.copy()
        
        try:
            # Stage 1: Quick smooth optimization to get close
            if time.time() < deadline - safety_margin - 5:
                res1 = scipy.optimize.minimize(
                    lambda p: cost_smooth_params(p, 0.05)[0],
                    current_z, 
                    jac=lambda p: cost_smooth_params(p, 0.05)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'gtol': 1e-3, 'ftol': 1e-5}
                )
                if res1.fun < best_cost:
                    current_z = res1.x
            
            # Stage 2: Exact Nelder-Mead with many iterations
            res2 = scipy.optimize.minimize(
                exact_cost_params,
                current_z,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-11}
            )
            
            if res2.fun < best_cost:
                best_cost = res2.fun
                best_z = res2.x.copy()
                
        except Exception:
            pass

    # Final intensive polish on the best found
    if time.time() < deadline - safety_margin:
        try:
            # Multiple rounds of exact Nelder-Mead with tightening tolerances
            for polish_round in range(3):
                if time.time() > deadline - safety_margin:
                    break
                
                tol_x = 1e-9 / (polish_round + 1)
                tol_f = 1e-12 / (polish_round + 1)
                
                res_fin = scipy.optimize.minimize(
                    exact_cost_params,
                    best_z,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': tol_x, 'fatol': tol_f}
                )
                
                if res_fin.fun < best_cost:
                    best_cost = res_fin.fun
                    best_z = res_fin.x.copy()
        except Exception:
            pass

    # Construct full z array
    z_full = np.zeros(N)
    z_full[1:] = best_z
    
    # Construct 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = z_full[i_flat]
    base_y = z_full[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
