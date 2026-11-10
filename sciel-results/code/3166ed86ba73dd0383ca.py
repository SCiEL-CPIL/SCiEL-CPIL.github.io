# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def compute_1d_union_length_integral(x_vals, N):
    """
    Compute the integral over c in [0,1] of the union length of intervals
    [x_i + c*i/N, x_i + c*i/N + 1/N] for i=0,...,N-1.
    
    The union length at height c is a piecewise linear function of c.
    We find all critical c values where the ordering of interval endpoints changes,
    then integrate the union length over each linear segment.
    """
    # Each interval i has left endpoint L_i(c) = x_i + c*i/N and right endpoint R_i(c) = L_i(c) + 1/N
    # Critical c values occur when L_i(c) = L_j(c), R_i(c) = R_j(c), L_i(c) = R_j(c), etc.
    # L_i(c) = L_j(c) => x_i + c*i/N = x_j + c*j/N => c = (x_j - x_i) / ((i-j)/N) = N*(x_j - x_i)/(i-j)
    # Similarly for other pairs.
    
    # Collect all critical c values in [0,1]
    critical = [0.0, 1.0]
    for i in range(N):
        for j in range(i+1, N):
            # L_i = L_j
            if i != j:
                c = N * (x_vals[j] - x_vals[i]) / (i - j)
                if 0 <= c <= 1:
                    critical.append(c)
            # R_i = R_j
            if i != j:
                c = N * (x_vals[j] - x_vals[i]) / (i - j)
                if 0 <= c <= 1:
                    critical.append(c)
            # L_i = R_j
            # x_i + c*i/N = x_j + c*j/N + 1/N
            # c*(i-j)/N = x_j - x_i + 1/N
            # c = N*(x_j - x_i + 1/N)/(i-j)
            if i != j:
                c = N * (x_vals[j] - x_vals[i] + 1.0/N) / (i - j)
                if 0 <= c <= 1:
                    critical.append(c)
            # R_i = L_j
            # x_i + c*i/N + 1/N = x_j + c*j/N
            # c*(i-j)/N = x_j - x_i - 1/N
            # c = N*(x_j - x_i - 1/N)/(i-j)
            if i != j:
                c = N * (x_vals[j] - x_vals[i] - 1.0/N) / (i - j)
                if 0 <= c <= 1:
                    critical.append(c)
    
    critical = sorted(set([c for c in critical if 0 <= c <= 1]))
    
    # Integrate union length over each segment
    total_integral = 0.0
    for k in range(len(critical) - 1):
        c_mid = (critical[k] + critical[k+1]) / 2
        # Compute union length at c_mid
        intervals = []
        for i in range(N):
            left = x_vals[i] + c_mid * i / N
            right = left + 1.0 / N
            intervals.append((left, right))
        intervals.sort()
        
        # Compute union length
        union_len = 0.0
        if intervals:
            cur_left, cur_right = intervals[0]
            for idx in range(1, len(intervals)):
                l, r = intervals[idx]
                if l > cur_right:
                    union_len += cur_right - cur_left
                    cur_left, cur_right = l, r
                else:
                    cur_right = max(cur_right, r)
            union_len += cur_right - cur_left
        
        # The union length is linear on this segment, so integral = length * union_len(c_mid)
        seg_len = critical[k+1] - critical[k]
        total_integral += seg_len * union_len
    
    return total_integral


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    
    # Optimize 1D positions for x and y independently
    # The 3D volume = (integral of 1D x-union) * (integral of 1D y-union) / 1
    # Wait, no: volume = ∫₀¹ L_x(c) L_y(c) dc, not the product of integrals.
    # But if L_x(c) and L_y(c) are similar, it's approximately (∫L_x)(∫L_y).
    # Actually, for the product structure, the 3D volume is exactly ∫₀¹ L_x(c) L_y(c) dc.
    
    # Let me just optimize the 1D positions to minimize ∫₀¹ L_x(c) dc, and do the same for y.
    # Then the 3D volume will be small.
    
    def obj_1d(x_vals):
        return compute_1d_union_length_integral(x_vals, N)
    
    # Initial guess: Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # Optimize x positions
    result_x = minimize(obj_1d, keich.copy(), method='Nelder-Mead', 
                        options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
    x_vals = result_x.x
    
    # Optimize y positions (same problem, so same solution)
    y_vals = x_vals.copy()
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    positions = np.stack([x_vals[i_flat], y_vals[j_flat]], axis=1)
    return positions


# EVOLVE-BLOCK-END
