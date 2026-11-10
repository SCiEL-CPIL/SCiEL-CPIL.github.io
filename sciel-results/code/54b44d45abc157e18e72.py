import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp
import sys

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    frac = 1.0 / N
    start_time = time.time()
    timeout = 580  # Leave margin for the 600s limit

    # Keich construction for initial guess
    def keich_positions(N):
        pos = int(np.log2(N))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        return keich

    u_init = keich_positions(N)
    v_init = keich_positions(N)
    
    # We optimize 14 variables: u_1..u_7, v_1..v_7 (u_0, v_0 fixed at 0)
    # Total 64 tubes defined by (u_i, v_j)
    
    # Smooth union length approximation
    # We want to approximate L = len(union of [l_i, r_i])
    # L = max(r) - min(l) + sum of gaps
    # Gaps are max(0, r_j - l_{j+1}) where intervals are sorted by l.
    # Since N is small (8), we can enumerate all permutations? No, that's 40320.
    # Instead, we use the property that for fixed order, it's linear.
    # But order changes.
    # Alternative smooth approximation:
    # L ≈ max_i(r_i) - min_i(l_i) + sum_{i<j} max(0, r_i - l_j) * (some weight?)
    # Actually, a better smooth approximation for union length of intervals:
    # If we sort by left endpoints l_(1) <= l_(2) <= ... <= l_(N)
    # L = max(l_(1), r_(1)) - l_(1) + ... 
    # Standard formula: L = sum_{k=1}^N max(0, r_{(k)} - max(l_{(k)}, r_{(k-1)}))
    
    # Let's use a differentiable approximation based on the fact that
    # the union length is the integral of the indicator function.
    # Or simpler: Since N is very small, we can use a "soft" merge.
    # However, the most robust differentiable approach for small N is to
    # use the log-sum-exp trick to approximate max, and handle the sorting
    # via a "soft" ordering or just rely on the fact that the objective is
    # piecewise linear and L-BFGS-B can navigate it if we start close enough
    # or use a smooth relaxation.
    
    # Let's try the "Soft Union Length" using a high-temperature limit of a partition function?
    # Or simpler: Just use the exact piecewise linear function but provide subgradients?
    # SciPy's L-BFGS-B requires a smooth function.
    
    # Let's use the approximation:
    # L(u, v) = max_i (u_i + c*i/N + 1/N) - min_i (u_i + c*i/N) + sum_{i,j: i!=j} max(0, ...)
    # This is getting complicated.
    
    # Let's stick to the discrete union length but make it "smooth" enough for L-BFGS-B
    # by using a soft-max approximation for the max operations in the merge process.
    # Actually, for N=8, we can just evaluate the exact union length.
    # The function is continuous but not differentiable at boundaries.
    # L-BFGS-B uses finite differences if no gradient is provided, which is slow.
    # If we provide a gradient, it must be a subgradient or an approximation.
    
    # Let's implement the exact union length calculation efficiently in NumPy.
    def exact_union_length_1d(shifts, base, c):
        # shifts: array of shape (N,) containing i/N
        # base: array of shape (N,) containing u_i
        # c: scalar
        starts = base + c * shifts
        ends = starts + frac
        # Sort by start
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        
        # Merge
        current_end = e_sorted[0]
        total_len = 0.0
        for k in range(1, N):
            if s_sorted[k] > current_end:
                total_len += current_end - s_sorted[k-1] # Wait, this logic is slightly off for accumulation
                # Correct merge logic:
                # total_len += current_end - prev_start? No.
                # Let's just do standard merge.
                pass
        
        # Standard merge implementation
        total_len = 0.0
        cur_start = s_sorted[0]
        cur_end = e_sorted[0]
        for k in range(1, N):
            if s_sorted[k] > cur_end:
                total_len += cur_end - cur_start
                cur_start = s_sorted[k]
                cur_end = e_sorted[k]
            else:
                cur_end = max(cur_end, e_sorted[k])
        total_len += cur_end - cur_start
        return total_len

    # To make this differentiable, we can use a soft approximation.
    # But maybe a better strategy: Use the "Keich" structure as a strong prior.
    # The Keich construction is known to be optimal for Kakeya sets in 1D?
    # Actually, the Keich construction minimizes the measure of the union in 1D for N directions.
    # If we use the exact Keich positions, we get a specific volume.
    # Can we improve it?
    # The 3D volume is the integral over c of L_x(c) * L_y(c).
    # L_x(c) is the union length of intervals [u_i + c*i/N, +1/N].
    # If u_i are the Keich positions, L_x(c) is constant? No, it varies.
    
    # Let's try to optimize the Keich parameters or the positions directly using L-BFGS-B
    # with a smoothed objective.
    
    def smooth_max(x, k=100.0):
        return np.logsumexp(k * x) / k
    
    def smooth_union_length(base, c, k=50.0):
        # base: (N,)
        # Approximate union length using soft operations
        # This is tricky. Let's use the exact function and rely on L-BFGS-B's
        # finite difference gradient estimation, but accelerate it by using
        # a very fast exact evaluation.
        starts = base + c * (np.arange(N) / N)
        ends = starts + frac
        order = np.argsort(starts)
        s = starts[order]
        e = ends[order]
        
        # Vectorized merge
        # If s[i] > e[i-1], gap
        # L = e[-1] - s[0] - sum(gaps) ? No.
        # L = sum(e[i] - s[i]) - sum(overlaps)
        # Overlaps are hard to compute vectorized without loops.
        
        # For N=8, a python loop is fine if we call it many times?
        # L-BFGS-B might call it 1000s of times.
        # 1000 * 14 * 2 (finite diff) * 8 (loop) = 224k ops. Very fast.
        
        total = 0.0
        cs = s[0]
        ce = e[0]
        for i in range(1, N):
            if s[i] > ce:
                total += ce - cs
                cs = s[i]
                ce = e[i]
            else:
                ce = max(ce, e[i])
        total += ce - cs
        return total

    def volume_objective(params):
        u = np.zeros(N)
        v = np.zeros(N)
        u[1:] = params[:7]
        v[1:] = params[7:]
        
        # Integrate over c
        # Use Gauss-Legendre quadrature for high accuracy and speed
        # 50 points is usually enough for smooth-ish functions
        c_points, weights = np.polynomial.legendre.leggauss(50)
        c_vals = 0.5 * (c_points + 1.0) # map to [0, 1]
        
        vol = 0.0
        for c, w in zip(c_vals, weights):
            lx = smooth_union_length(u, c)
            ly = smooth_union_length(v, c)
            vol += w * lx * ly
        return 0.5 * vol # leggauss weights integrate over [-1,1], so factor 0.5 for [0,1]

    # Initial guess: Keich
    u0 = np.zeros(N)
    v0 = np.zeros(N)
    u0[1:] = u_init[1:]
    v0[1:] = v_init[1:]
    x0 = np.concatenate([u0[1:], v0[1:]])
    
    # Bounds
    # Keep positions within a reasonable range to avoid huge boxes
    bounds = [(-2.0, 2.0)] * 14
    
    best_x = x0
    best_f = volume_objective(x0)
    
    # Run L-BFGS-B
    try:
        res = minimize(
            volume_objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
    except Exception as e:
        pass
    
    # If time remains, try a second phase with different initializations
    # or Nelder-Mead polish
    if time.time() - start_time < timeout:
        # Try Nelder-Mead on the result
        try:
            res_nm = minimize(
                volume_objective, 
                best_x, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res_nm.fun < best_f:
                best_f = res_nm.fun
                best_x = res_nm.x
        except Exception:
            pass

    # Try a few random restarts if time permits
    rng = np.random.default_rng(42)
    for _ in range(5):
        if time.time() - start_time > timeout - 60:
            break
        # Perturb initial guess
        x_rand = x0 + rng.normal(0, 0.1, size=14)
        x_rand = np.clip(x_rand, -2.0, 2.0)
        
        try:
            res_r = minimize(
                volume_objective, 
                x_rand, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res_r.fun < best_f:
                best_f = res_r.fun
                best_x = res_r.x
        except Exception:
            pass

    # Construct final positions
    u_final = np.zeros(N)
    v_final = np.zeros(N)
    u_final[1:] = best_x[:7]
    v_final[1:] = best_x[7:]
    
    positions = np.zeros((64, 2))
    for k in range(64):
        i = k // 8
        j = k % 8
        positions[k, 0] = u_final[i]
        positions[k, 1] = v_final[j]
        
    return positions

if __name__ == "__main__":
    # Quick test
    pos = kakeya_needle_3d8()
    print(pos.shape)
