# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.

    We build N*N = 64 "generalized tubes" indexed by (i, j), i, j in {0, ..., N-1}.
    Tube (i, j) has a base position (x, y) and, at height c in [0, 1], occupies the
    axis-aligned square
        [x + c*i/N, x + 1/N + c*i/N] x [y + c*j/N, y + 1/N + c*j/N].

    We exploit the product structure: the 3D union volume is a function of the
    1D union lengths in x (depending only on i) and y (depending only on j).
    We use a smooth, differentiable approximation of the 1D interval union length
    and optimize the 16 base positions (8 in x, 8 in y) using L-BFGS-B.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # 10s safety margin

    N = 8
    w = 1.0 / N  # tube width

    # Precompute shear offsets for each i (and j)
    # At height c, x-offset = c * i / N, so the x-interval for tube i is:
    #   [x_i + c*i/N, x_i + w + c*i/N] for c in [0, 1]
    # The union of these intervals over c is equivalent to the union over c of
    # intervals [x_i, x_i + w] shifted by c*i/N.
    # Actually, the x-projection of tube (i,j) onto the x-axis at height c is
    # [x_i + c*i/N, x_i + w + c*i/N]. The union over c in [0,1] of these intervals
    # is [x_i, x_i + w + i/N] (since the shift goes from 0 to i/N).
    # Wait, that's not right. The union over c of [a + c*s, a + w + c*s] for c in [0,1]
    # is [a, a + w + s] if s >= 0, which is just the convex hull.
    # But the VOLUME is not just the product of the x-range and y-range.
    # The volume is the measure of the set of (x, y, c) such that x is in the x-interval
    # of some tube AND y is in the y-interval of some tube AND the tube index matches.
    #
    # Actually, the product structure works as follows:
    # A point (a, b, c) is in tube (i, j) iff:
    #   a in [x_i + c*i/N, x_i + w + c*i/N] AND b in [y_j + c*j/N, y_j + w + c*j/N]
    # The union over all (i, j) is the set of (a, b, c) such that there exists i with
    # a in [x_i + c*i/N, x_i + w + c*i/N] AND there exists j with b in [y_j + c*j/N, y_j + w + c*j/N].
    # Since the i-condition and j-condition are independent, the volume is:
    #   ∫_0^1 L_x(c) * L_y(c) dc
    # where L_x(c) = length of union over i of [x_i + c*i/N, x_i + w + c*i/N]
    # and L_y(c) = length of union over j of [y_j + c*j/N, y_j + w + c*j/N].
    #
    # So we need to minimize ∫_0^1 L_x(c) * L_y(c) dc.

    def union_length_1d_smooth(base, shear_indices, c, eps=1e-4):
        """
        Smooth approximation of the length of the union of intervals
        [base[i] + c * shear_indices[i] / N, base[i] + w + c * shear_indices[i] / N]
        for i in range(N).

        We use a soft-max based approach: the union length can be approximated by
        summing the lengths of intervals minus overlaps, but that's complex.
        
        Alternative: discretize the 1D domain and compute coverage.
        But that's not smooth.

        Better approach: The union of intervals [a_i, b_i] has length
        = max(b_i) - min(a_i) - sum of gaps.
        
        For smoothness, we can use the fact that for intervals that are "close",
        the union length is approximately the length of the convex hull minus
        the "gaps" between sorted intervals.

        Let's use a different smooth approximation:
        For a set of intervals, the union length is:
        sum_i (b_i - a_i) - sum_{i<j} overlap(i, j) + sum_{i<j<k} triple_overlap - ...
        This is the inclusion-exclusion principle, which is exact but computationally
        expensive and not smooth.

        Practical approach: Use a fine discretization of the 1D line and compute
        the fraction of the line covered, then multiply by the range.
        But this is piecewise constant, not smooth.

        Let's use the following smooth approximation:
        The union length of intervals [a_i, b_i] can be approximated by:
        (max b_i - min a_i) * sigmoid-based coverage factor
        
        Actually, let's just use numerical differentiation-friendly computation:
        We'll compute the union length exactly (piecewise) and rely on L-BFGS-B's
        numerical gradients, or we can compute analytical gradients.

        For now, let's compute the exact union length (it's piecewise linear in the
        base positions, so gradients exist almost everywhere).
        """
        n = len(base)
        starts = base + c * shear_indices / N
        ends = starts + w
        
        # Sort intervals by start
        order = np.argsort(starts)
        starts_sorted = starts[order]
        ends_sorted = ends[order]
        
        # Compute union length by merging
        union_len = 0.0
        current_start = starts_sorted[0]
        current_end = ends_sorted[0]
        for k in range(1, n):
            s = starts_sorted[k]
            e = ends_sorted[k]
            if s <= current_end:
                current_end = max(current_end, e)
            else:
                union_len += current_end - current_start
                current_start = s
                current_end = e
        union_len += current_end - current_start
        
        return union_len

    def volume_1d_integral(base_x, base_y, num_c_samples=200):
        """
        Compute ∫_0^1 L_x(c) * L_y(c) dc using numerical quadrature.
        """
        c_vals = np.linspace(0, 1, num_c_samples)
        shear = np.arange(N)
        
        lx_vals = np.array([union_length_1d_smooth(base_x, shear, c) for c in c_vals])
        ly_vals = np.array([union_length_1d_smooth(base_y, shear, c) for c in c_vals])
        
        # Trapezoidal rule
        volume = np.trapz(lx_vals * ly_vals, c_vals)
        return volume

    def objective(params):
        """
        Objective function: volume of the 3D union.
        params: 16 values, first 8 are base_x, last 8 are base_y.
        """
        base_x = params[:N]
        base_y = params[N:]
        return volume_1d_integral(base_x, base_y, num_c_samples=100)

    # Initial guess: Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    x0 = np.concatenate([keich, keich])

    # First, try L-BFGS-B with the Keich initialization
    best_params = x0.copy()
    best_vol = objective(x0)

    if time.time() < deadline - 30:
        res = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.success or res.fun < best_vol:
            best_params = res.x
            best_vol = res.fun

    # Try multiple random restarts with L-BFGS-B
    rng = np.random.default_rng(42)
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Random initialization within a reasonable range
        # The base positions should be in a range that allows good overlap
        # Typical range is [-1, 2] or so
        x0_r = np.concatenate([
            rng.uniform(-0.5, 1.5, N),
            rng.uniform(-0.5, 1.5, N)
        ])
        
        res = minimize(objective, x0_r, method='L-BFGS-B',
                      options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.fun < best_vol:
            best_vol = res.fun
            best_params = res.x

    # Also try differential evolution for global search if time permits
    if time.time() < deadline - 60:
        bounds = [(-1.0, 2.0)] * (2 * N)
        de_res = differential_evolution(objective, bounds, seed=42, maxiter=50,
                                        tol=1e-6, popsize=15, polish=False,
                                        init='latinhypercube')
        if de_res.fun < best_vol:
            best_vol = de_res.fun
            best_params = de_res.x
    
    # Final polish with L-BFGS-B from the best point found
    if time.time() < deadline - 10:
        res = minimize(objective, best_params, method='L-BFGS-B',
                      options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.fun < best_vol:
            best_vol = res.fun
            best_params = res.x

    base_x = best_params[:N]
    base_y = best_params[N:]

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    positions = np.stack([base_x[i_flat], base_y[j_flat]], axis=1)
    return positions


# EVOLVE-BLOCK-END
