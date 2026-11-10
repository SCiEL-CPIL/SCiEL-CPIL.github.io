import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    frac = 1.0 / N
    start_time = time.time()
    timeout = 600  # Evaluator kills at 630s, leave margin
    
    # Pre-compute indices for each of the 64 tubes
    # Tube k corresponds to i = k // 8, j = k % 8
    indices = np.array([(k // N, k % N) for k in range(N * N)], dtype=np.int64)
    i_vals = indices[:, 0].astype(np.float64)
    j_vals = indices[:, 1].astype(np.float64)
    
    # Exact Union Length Calculation for 1D intervals
    def exact_union_length(starts, lengths):
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = s_sorted + lengths[order]
        
        cur_start = s_sorted[0]
        cur_end = e_sorted[0]
        total_len = 0.0
        
        for k in range(1, len(s_sorted)):
            if s_sorted[k] > cur_end:
                total_len += cur_end - cur_start
                cur_start = s_sorted[k]
                cur_end = e_sorted[k]
            else:
                if e_sorted[k] > cur_end:
                    cur_end = e_sorted[k]
        
        total_len += cur_end - cur_start
        return total_len

    def compute_volume_exact(params):
        x_pos = params[:64]
        y_pos = params[64:]
        
        c_pts, w_pts = np.polynomial.legendre.leggauss(32)
        c_vals = 0.5 * (c_pts + 1.0)
        
        vol = 0.0
        for c, w in zip(c_vals, w_pts):
            x_starts = x_pos + c * (i_vals / N)
            y_starts = y_pos + c * (j_vals / N)
            
            lx = exact_union_length(x_starts, np.full(64, frac))
            ly = exact_union_length(y_starts, np.full(64, frac))
            
            vol += w * lx * ly
            
        return 0.5 * vol

    # Smooth Surrogate for Union Length
    # Uses a soft-max approximation to make the union length differentiable.
    # The union of intervals [s_k, s_k + L] has length approximated by
    # max_k(s_k + L) - min_k(s_k) if they are disjoint, but generally
    # we want a smooth function that approximates the measure of the union.
    # A common trick is to use the log-sum-exp of endpoints, but for intervals,
    # we can approximate the union length by integrating the indicator function
    # or using a smooth maximum of the "coverage" function.
    
    # Alternative Smooth Strategy:
    # The volume is the integral of Lx(c) * Ly(c).
    # Lx(c) is the length of the union of intervals.
    # We can approximate the length of the union of intervals [s_k, e_k]
    # by summing lengths minus overlaps. This is hard to smooth directly.
    
    # Better Smooth Strategy:
    # Use the fact that for small intervals, the union length is close to
    # the sum of lengths if they are well-separated.
    # A robust smooth approximation for the union of intervals with same length L:
    # Sort starts. The union length is sum of (e_k - s_k) - overlaps.
    # Overlaps are max(0, min(e_i, e_j) - max(s_i, s_j)).
    # This is still non-smooth.
    
    # Let's use a differentiable proxy:
    # The "soft" union length can be approximated by:
    # L_soft = (1/2) * (sum(e_k) - sum(s_k)) + (1/2) * (max_soft(e) - min_soft(s))
    # This isn't quite right for overlapping intervals.
    
    # Let's stick to the known effective strategy from inspirations:
    # Use a smooth maximum function to approximate the union boundaries.
    # However, the most effective smooth surrogate for union of intervals
    # in these Kakeya problems often involves approximating the measure
    # via a smooth "max" of the indicator or using a Gaussian smoothing
    # of the step functions.
    
    # Given time constraints and the specific success of L-BFGS with smooth surrogates,
    # we will define a smooth volume objective.
    # A simple smooth approximation for the union length of intervals [s_k, s_k+L]
    # is not trivial. But we can use the following property:
    # If we assume the intervals are "mostly" disjoint or use a very smooth
    # approximation, we can optimize a proxy.
    
    # Actually, a very effective smooth surrogate for the union length of intervals
    # is based on the log-sum-exp of the endpoints, but adjusted.
    # Let's use a simpler differentiable proxy that correlates well with volume:
    # Minimize the sum of squared distances between interval centers? No.
    
    # Let's use the "Soft Union" defined by:
    # U_soft(s) = sum_k L_k - sum_{i<j} overlap_soft(i,j)
    # Where overlap_soft(i,j) = soft_max(0, min(e_i, e_j) - max(s_i, s_j))
    # This is O(N^2) but N=64 is small.
    
    def soft_max(x, t=0.01):
        return t * np.log(np.sum(np.exp(x / t)))
    
    def soft_min(x, t=0.01):
        return -soft_max(-x, t)
        
    def soft_overlap(s1, e1, s2, e2, t=0.01):
        # Overlap is max(0, min(e1, e2) - max(s1, s2))
        # min(e1, e2) approx soft_min(e1, e2)
        # max(s1, s2) approx soft_max(s1, s2)
        # diff = min_e - max_s
        # overlap = soft_max(0, diff)
        min_e = soft_min(np.array([e1, e2]), t)
        max_s = soft_max(np.array([s1, s2]), t)
        diff = min_e - max_s
        return soft_max(np.array([0.0, diff]), t)

    def smooth_union_length(starts, lengths, t=0.01):
        # N intervals
        n = len(starts)
        s = starts
        e = starts + lengths
        
        # Total length is sum of lengths minus pairwise overlaps
        # This is an over-approximation if triple overlaps exist, but for small t
        # and well-behaved configurations, it's a good smooth proxy.
        # Actually, inclusion-exclusion up to 2nd order is a common smooth approximation.
        total_len = np.sum(lengths)
        
        overlaps = 0.0
        for i in range(n):
            for j in range(i+1, n):
                overlaps += soft_overlap(s[i], e[i], s[j], e[j], t)
                
        return total_len - overlaps

    def compute_volume_smooth(params, t=0.01):
        x_pos = params[:64]
        y_pos = params[64:]
        
        c_pts, w_pts = np.polynomial.legendre.leggauss(16) # Fewer points for speed in smooth phase
        c_vals = 0.5 * (c_pts + 1.0)
        
        vol = 0.0
        for c, w in zip(c_vals, w_pts):
            x_starts = x_pos + c * (i_vals / N)
            y_starts = y_pos + c * (j_vals / N)
            
            lx = smooth_union_length(x_starts, np.full(64, frac), t)
            ly = smooth_union_length(y_starts, np.full(64, frac), t)
            
            vol += w * lx * ly
            
        return 0.5 * vol

    # Initial Guess: Keich Construction
    def keich_positions_1d(N):
        pos = int(np.log2(N))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        return keich

    u_keich = keich_positions_1d(N)
    v_keich = keich_positions_1d(N)
    
    x_init = np.zeros(64)
    y_init = np.zeros(64)
    for k in range(64):
        i = k // N
        j = k % N
        x_init[k] = u_keich[i]
        y_init[k] = v_keich[j]
        
    params_init = np.concatenate([x_init, y_init])
    
    best_params = params_init.copy()
    best_vol_exact = compute_volume_exact(best_params)
    best_vol_smooth = compute_volume_smooth(best_params)
    
    # Phase 1: Optimize Smooth Objective with L-BFGS-B
    # L-BFGS-B is efficient for large dimensions and handles bounds well.
    # We don't have explicit bounds, but the problem is translationally invariant
    # in a sense, so we can let it move. To prevent drift, we might add
    # a small penalty for large coordinates, or just rely on the volume minimization.
    
    def objective_smooth(params):
        return compute_volume_smooth(params)
        
    def gradient_smooth(params):
        # Numerical gradient for L-BFGS-B if analytical is too complex
        # Or use finite differences.
        # For 128D, central differences are 256 evals.
        # Smooth eval is fast (16 c-points * 64*64/2 overlaps ~ 16*2000 = 32k ops).
        # 256 * 32k = 8M ops per gradient step. Very fast.
        eps = 1e-6
        grad = np.zeros_like(params)
        for i in range(len(params)):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            grad[i] = (objective_smooth(p_plus) - objective_smooth(p_minus)) / (2 * eps)
        return grad

    # Use L-BFGS-B
    try:
        res1 = minimize(
            objective_smooth, 
            best_params, 
            jac=gradient_smooth,
            method='L-BFGS-B', 
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        if res1.fun < best_vol_smooth:
            # Verify with exact volume
            vol_exact_check = compute_volume_exact(res1.x)
            if vol_exact_check < best_vol_exact:
                best_vol_exact = vol_exact_check
                best_params = res1.x
                best_vol_smooth = res1.fun
    except Exception as e:
        print(f"Smooth phase error: {e}")

    # Phase 2: Polish with Exact Objective using Nelder-Mead or L-BFGS-B (numerical grad)
    # Since the smooth solution is close to a local min of the exact objective,
    # a few steps of Nelder-Mead or BFGS on the exact objective should refine it.
    
    def objective_exact(params):
        return compute_volume_exact(params)
        
    def gradient_exact(params):
        eps = 1e-5
        grad = np.zeros_like(params)
        for i in range(len(params)):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            grad[i] = (objective_exact(p_plus) - objective_exact(p_minus)) / (2 * eps)
        return grad

    # Check time
    if time.time() - start_time < timeout - 60:
        try:
            # Use L-BFGS-B on exact objective starting from smooth solution
            res2 = minimize(
                objective_exact, 
                best_params, 
                jac=gradient_exact,
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res2.fun < best_vol_exact:
                best_vol_exact = res2.fun
                best_params = res2.x
        except Exception as e:
            print(f"Exact phase error: {e}")

    # Phase 3: Random Restarts with Smooth -> Exact Pipeline
    # If time remains, try restarting from slightly perturbed Keich or random points.
    rng = np.random.default_rng(12345)
    
    max_restarts = 5
    for r in range(max_restarts):
        if time.time() - start_time > timeout - 90:
            break
            
        # Perturb the best solution or start from Keich with noise
        if r == 0:
            p_start = best_params + rng.normal(0, 0.01, size=128)
        else:
            # Start from Keich with larger noise to explore
            p_start = params_init + rng.normal(0, 0.05, size=128)
            
        try:
            # Quick smooth optimization
            res_sm = minimize(
                objective_smooth, 
                p_start, 
                jac=gradient_smooth,
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            # Quick exact polish
            res_ex = minimize(
                objective_exact, 
                res_sm.x, 
                jac=gradient_exact,
                method='L-BFGS-B', 
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            
            if res_ex.fun < best_vol_exact:
                best_vol_exact = res_ex.fun
                best_params = res_ex.x
        except Exception:
            pass

    x_final = best_params[:64]
    y_final = best_params[64:]
    
    positions = np.column_stack((x_final, y_final))
    
    return positions

if __name__ == "__main__":
    pos = kakeya_needle_3d8()
    print(pos.shape)
