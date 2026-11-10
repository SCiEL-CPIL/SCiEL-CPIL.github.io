import numpy as np
import time
import sys
from scipy.optimize import minimize

def compute_2d_cost_and_grad(positions):
    """
    Computes the integral from 0 to 1 of (union length)^2 for the 1D problem,
    and its gradient with respect to positions.
    The union length at height c is the total length of the union of intervals
    [x_i + c*i/8, x_i + c*i/8 + 1/8] for i=0..7.
    
    We use the fact that the union length is piecewise linear in c (breakpoints
    where interval endpoints coincide), so (union length)^2 is piecewise quadratic.
    We integrate exactly by splitting at breakpoints.
    """
    N = 8
    w = 1.0 / 8.0
    # Collect breakpoints in c where any two interval endpoints meet
    breakpoints = set([0.0, 1.0])
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            denom = (i - j) / 8.0
            if abs(denom) < 1e-12:
                continue
            # left endpoint of i meets left endpoint of j
            c_val = (positions[j] - positions[i]) / denom
            if 0.0 < c_val < 1.0:
                breakpoints.add(c_val)
            # left endpoint of i meets right endpoint of j
            c_val = (positions[j] + w - positions[i]) / denom
            if 0.0 < c_val < 1.0:
                breakpoints.add(c_val)
            # right endpoint of i meets left endpoint of j
            c_val = (positions[j] - positions[i] - w) / denom
            if 0.0 < c_val < 1.0:
                breakpoints.add(c_val)
            # right endpoint of i meets right endpoint of j
            c_val = (positions[j] + w - positions[i] - w) / denom
            if 0.0 < c_val < 1.0:
                breakpoints.add(c_val)
    
    breakpoints = sorted(breakpoints)
    
    total_cost = 0.0
    total_grad = np.zeros(N)
    
    for k in range(len(breakpoints) - 1):
        c_start = breakpoints[k]
        c_end = breakpoints[k + 1]
        c_mid = (c_start + c_end) / 2.0
        dc = c_end - c_start
        
        # At c_mid, compute the union of intervals and its length
        # Intervals: [x_i + c*i/8, x_i + c*i/8 + w]
        lefts = positions + (np.arange(N) / 8.0) * c_mid
        rights = lefts + w
        
        # Sort by left endpoint
        order = np.argsort(lefts)
        sorted_lefts = lefts[order]
        sorted_rights = rights[order]
        sorted_indices = order
        
        # Merge intervals to find union segments
        merged = []
        cur_l = sorted_lefts[0]
        cur_r = sorted_rights[0]
        for idx in range(1, N):
            l = sorted_lefts[idx]
            r = sorted_rights[idx]
            if l <= cur_r:
                if r > cur_r:
                    cur_r = r
            else:
                merged.append((cur_l, cur_r))
                cur_l = l
                cur_r = r
        merged.append((cur_l, cur_r))
        
        # Union length at c_mid
        union_len = sum(r - l for l, r in merged)
        
        # The union length is piecewise linear in c within each breakpoint interval.
        # Within a sub-interval between consecutive breakpoints, the set of overlapping
        # intervals is constant, so the union length is linear: L(c) = a*c + b.
        # We can determine a and b from two points, but since we're in a linear region,
        # L(c) = L(c_mid) + slope * (c - c_mid).
        # Actually, since it's linear, the integral of L(c)^2 over [c_start, c_end]
        # can be computed exactly if we know L at two points or the slope.
        # 
        # Better approach: Since L(c) is linear on [c_start, c_end], we have
        # L(c) = L(c_mid) + m*(c - c_mid) where m is the slope.
        # The slope m is the sum of contributions: each interval contributes +1/8 to
        # the "density" but the union structure is fixed, so the slope of the union
        # length equals the sum over the rightmost and leftmost active boundaries.
        # 
        # Actually, the simplest exact method: sample L at c_start and c_end (they are
        # breakpoints, but the formula is continuous), then L(c) is linear so:
        # integral of L^2 = (dc/3) * (L_start^2 + L_start*L_end + L_end^2)
        
        # Compute L at c_start and c_end
        def union_length_at(c_val):
            ls = positions + (np.arange(N) / 8.0) * c_val
            rs = ls + w
            o = np.argsort(ls)
            sl = ls[o]
            sr = rs[o]
            merged_l = []
            cl = sl[0]
            cr = sr[0]
            for idx in range(1, N):
                if sl[idx] <= cr:
                    if sr[idx] > cr:
                        cr = sr[idx]
                else:
                    merged_l.append(cr - cl)
                    cl = sl[idx]
                    cr = sr[idx]
            merged_l.append(cr - cl)
            return sum(merged_l)
        
        L_start = union_length_at(c_start)
        L_end = union_length_at(c_end)
        
        # Integral of (a*c + b)^2 from c_start to c_end where L is linear
        # = (dc/3) * (L_start^2 + L_start*L_end + L_end^2)
        total_cost += (dc / 3.0) * (L_start ** 2 + L_start * L_end + L_end ** 2)
        
        # Gradient: d/dc of L(c) is the slope m.
        # m = (L_end - L_start) / dc
        m = (L_end - L_start) / dc if dc > 1e-15 else 0.0
        
        # The gradient of the cost w.r.t. positions[i] involves the derivative of
        # L(c) w.r.t. positions[i]. Since L(c) is the union length, and each interval
        # [x_i + c*i/8, x_i + c*i/8 + w] has its endpoints depending on x_i,
        # the derivative of L w.r.t. x_i is the sum of contributions from the
        # left and right endpoints of interval i to the union boundary.
        #
        # For a union of intervals, the derivative of the total length w.r.t. a
        # parameter that shifts an interval is +1 if the left endpoint is a
        # "free" left boundary of the union, -1 if the right endpoint is a
        # "free" right boundary of the union, and 0 otherwise.
        #
        # More precisely: if interval i's left endpoint is the leftmost point of
        # a union segment, shifting x_i right by dx increases the union length by dx
        # (if the left endpoint moves right, the union segment shrinks... wait).
        #
        # Let me think carefully. If interval i = [a_i, b_i] where a_i = x_i + c*i/8,
        # b_i = a_i + w. The union length is sum of (right - left) for merged segments.
        # d/dx_i of union length:
        # - If a_i is the left endpoint of a merged segment: moving x_i right by dx
        #   moves a_i right by dx, so the segment's left moves right, length decreases by dx.
        #   Contribution: -1
        # - If b_i is the right endpoint of a merged segment: moving x_i right by dx
        #   moves b_i right by dx, so the segment's right moves right, length increases by dx.
        #   Contribution: +1
        # - If both: contribution 0 (the segment just translates)
        # - If neither: contribution 0
        #
        # So dL/dx_i = (1 if b_i is a free right boundary) - (1 if a_i is a free left boundary)
        #
        # We need to determine for each interval i at a given c whether its left or right
        # endpoint is a free boundary of the union.
        
        def compute_boundary_flags(c_val):
            ls = positions + (np.arange(N) / 8.0) * c_val
            rs = ls + w
            o = np.argsort(ls)
            sl = ls[o]
            sr = rs[o]
            
            # Determine which endpoints are free boundaries of the union
            # A left endpoint sl[k] is a free left boundary if it is not covered by
            # any previous interval's right endpoint (i.e., sl[k] > max previous sr)
            # A right endpoint sr[k] is a free right boundary if it is the maximum
            # right endpoint among all intervals that form the current merged segment.
            
            left_free = np.zeros(N, dtype=bool)
            right_free = np.zeros(N, dtype=bool)
            
            # Process in order of left endpoints
            max_r_so_far = -np.inf
            for k in range(N):
                orig_idx = o[k]
                if sl[k] > max_r_so_far + 1e-15:
                    # This interval starts a new merged segment
                    left_free[orig_idx] = True
                max_r_so_far = max(max_r_so_far, sr[k])
            
            # For right boundaries: the right endpoint of interval i is a free right
            # boundary if it is the maximum right endpoint in its merged segment.
            # We need to identify merged segments.
            # Re-do the merge and track which interval provides the max right in each segment.
            cur_seg_max_r = -np.inf
            cur_seg_max_r_idx = -1
            segments = []  # list of (start_idx_in_order, end_idx_in_order)
            seg_start = 0
            for k in range(1, N):
                if sl[k] > sr[k-1] + 1e-15:
                    # New segment starts
                    segments.append((seg_start, k-1))
                    seg_start = k
            segments.append((seg_start, N-1))
            
            for seg in segments:
                a, b = seg
                # Find the index in [a, b] with max right endpoint
                max_r = -np.inf
                max_idx = -1
                for k in range(a, b+1):
                    if sr[k] > max_r:
                        max_r = sr[k]
                        max_idx = k
                orig_idx = o[max_idx]
                right_free[orig_idx] = True
            
            return left_free, right_free
        
        # The gradient of the cost integral w.r.t. positions[i] is:
        # integral_0^1 2 * L(c) * dL/dc * dL/dx_i dc
        # where dL/dx_i is the boundary flag contribution at height c.
        #
        # This is complex to compute exactly. Instead, since L(c) is piecewise linear
        # and dL/dx_i is piecewise constant (changing only at breakpoints), we can
        # compute the gradient numerically or use a finite difference approximation.
        #
        # For simplicity and robustness, let's use a central finite difference for
        # the gradient, which is expensive but reliable. However, this defeats the
        # purpose of L-BFGS-B. Let's use a simpler approach: compute the gradient
        # analytically for each sub-interval.
        #
        # Actually, let's just use a numerical gradient with a small step, but only
        # evaluate the cost function. This is slower but correct.
        #
        # Alternative: since we're in a linear region, and the boundary flags are
        # constant within the region, we can compute:
        # dL/dx_i = right_free[i] - left_free[i]  (at any c in the interval)
        # Then the contribution to the gradient from this sub-interval is:
        # integral_{c_start}^{c_end} 2 * L(c) * m * (right_free[i] - left_free[i]) dc
        # = 2 * (right_free[i] - left_free[i]) * m * integral_{c_start}^{c_end} L(c) dc
        # = 2 * (right_free[i] - left_free[i]) * m * (dc/2) * (L_start + L_end)
        # = (right_free[i] - left_free[i]) * m * dc * (L_start + L_end)
        
        lf, rf = compute_boundary_flags(c_mid)
        dL_dx = rf.astype(float) - lf.astype(float)
        
        total_grad += dL_dx * m * dc * (L_start + L_end)
    
    return total_cost, total_grad

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = 620.0  # safety margin
    
    N = 8
    
    # Initial guess: Keich configuration
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    best_positions = keich.copy()
    best_cost, _ = compute_2d_cost_and_grad(best_positions)
    
    rng = np.random.default_rng(12345)
    
    # Multi-start L-BFGS-B optimization
    num_starts = 50
    max_iter_per_start = 500
    
    for start_idx in range(num_starts):
        if time.time() - start_time > deadline:
            break
        
        # Generate initial point
        if start_idx == 0:
            x0 = keich.copy()
        else:
            x0 = keich + rng.normal(0, 0.1, N)
        
        def objective(x):
            cost, _ = compute_2d_cost_and_grad(x)
            return cost
        
        def gradient(x):
            _, grad = compute_2d_cost_and_grad(x)
            return grad
        
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': max_iter_per_start, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_cost:
                best_cost = res.fun
                best_positions = res.x.copy()
        except Exception:
            pass
    
    # Construct the 2D positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_positions[i_flat]
    base_y = best_positions[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

if __name__ == "__main__":
    res = kakeya_needle_3d8()
    print(res.shape)
