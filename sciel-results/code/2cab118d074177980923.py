import numpy as np
import time
from scipy.optimize import minimize

def _compute_1d_volume(x, N=8):
    """
    Compute the 1D union volume for a set of N intervals that shear over c in [0,1].
    Interval i: [x_i + c*i/N, x_i + c*i/N + 1/N]
    Volume = integral_0^1 (length of union of intervals at c) dc
    
    The union length changes only at breakpoints where two interval endpoints coincide.
    Between breakpoints, the union length is linear in c, so we integrate exactly.
    """
    w = 1.0 / N
    slopes = np.arange(N, dtype=float) / N
    
    # Find all breakpoints in (0, 1)
    # Breakpoint: x_i + c*i/N = x_j + c*j/N + w  => c = N*(w + x_j - x_i)/(i-j)
    # Or: x_i + c*i/N + w = x_j + c*j/N  => c = N*(x_j - x_i - w)/(i-j)
    # In general: x_i + s_i*c + d1 = x_j + s_j*c + d2
    # c = (x_j - x_i + d2 - d1) / (s_i - s_j)
    # Endpoints are x_i + s_i*c and x_i + s_i*c + w
    # So d1, d2 are either 0 or w.
    
    bps = {0.0, 1.0}
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            for d1 in (0.0, w):
                for d2 in (0.0, w):
                    denom = slopes[i] - slopes[j]
                    if abs(denom) < 1e-15:
                        continue
                    c_val = (x[j] - x[i] + d2 - d1) / denom
                    if -1e-12 < c_val < 1.0 + 1e-12:
                        c_val = max(0.0, min(1.0, c_val))
                        bps.add(c_val)
    
    bps = sorted(bps)
    
    total_vol = 0.0
    for k in range(len(bps) - 1):
        c0 = bps[k]
        c1 = bps[k + 1]
        h = c1 - c0
        if h < 1e-12:
            continue
        
        # Evaluate union length at c0 and c1
        # At a given c, intervals are [x_i + c*s_i, x_i + c*s_i + w]
        def union_length(c_val):
            lefts = x + c_val * slopes
            rights = lefts + w
            # Sort by left endpoint
            order = np.argsort(lefts)
            lefts_s = lefts[order]
            rights_s = rights[order]
            
            total_len = 0.0
            cur_r = rights_s[0]
            for idx in range(1, N):
                if lefts_s[idx] > cur_r:
                    total_len += cur_r - (lefts_s[idx-1] if idx == 1 else lefts_s[idx-1])  # This is wrong
                    # Let me redo this properly
                    break
            # Proper merge
            total_len = 0.0
            cur_l = lefts_s[0]
            cur_r = rights_s[0]
            for idx in range(1, N):
                if lefts_s[idx] <= cur_r:
                    cur_r = max(cur_r, rights_s[idx])
                else:
                    total_len += cur_r - cur_l
                    cur_l = lefts_s[idx]
                    cur_r = rights_s[idx]
            total_len += cur_r - cur_l
            return total_len
        
        L0 = union_length(c0)
        L1 = union_length(c1)
        
        # L(c) is linear on [c0, c1], so integral = (L0 + L1) / 2 * h
        total_vol += 0.5 * (L0 + L1) * h
    
    return total_vol


def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 550
    
    N = 8
    
    # The volume of the 64-tube union is V = integral_0^1 A(c) dc
    # where A(c) is the area of the union of 64 squares at height c.
    # Each square at height c is [x_i + c*i/N, x_i + c*i/N + 1/N] x [y_j + c*j/N, y_j + c*j/N + 1/N]
    # for i,j in 0..N-1.
    
    # If the x-intervals at height c form a single connected interval of length Lx(c),
    # and the y-intervals form a single connected interval of length Ly(c),
    # then A(c) = Lx(c) * Ly(c).
    
    # In that case, V = integral_0^1 Lx(c) Ly(c) dc.
    # If x and y have the same structure, Lx(c) = Ly(c) = L(c), and V = integral_0^1 L(c)^2 dc.
    
    # Even if they don't have the same structure, if both are single intervals,
    # V = integral Lx(c) Ly(c) dc.
    
    # The key insight: we want to minimize V. The best case is when Lx(c) and Ly(c) 
    # are as small as possible. The minimum possible L(c) for N intervals of width 1/N
    # that all must be covered is... well, they could all overlap perfectly, giving L(c) = 1/N.
    # But they shear differently, so they can't all overlap at all c.
    
    # Let's optimize x and y to minimize the 1D union volume, and use that as a proxy.
    # Actually, let's directly optimize the 2D volume assuming single-rectangle structure.
    
    def _2d_volume(x, y, N=8):
        """
        Compute the 2D union volume assuming single-rectangle structure.
        If the structure breaks, return a penalty.
        """
        w = 1.0 / N
        slopes = np.arange(N, dtype=float) / N
        
        # Find breakpoints for x and y
        def get_bps(pos):
            bps = {0.0, 1.0}
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    for d1 in (0.0, w):
                        for d2 in (0.0, w):
                            denom = slopes[i] - slopes[j]
                            if abs(denom) < 1e-15:
                                continue
                            c_val = (pos[j] - pos[i] + d2 - d1) / denom
                            if -1e-12 < c_val < 1.0 + 1e-12:
                                c_val = max(0.0, min(1.0, c_val))
                                bps.add(c_val)
            return sorted(bps)
        
        bps_x = get_bps(x)
        bps_y = get_bps(y)
        all_bps = sorted(set(bps_x + bps_y))
        
        total_vol = 0.0
        
        for k in range(len(all_bps) - 1):
            c0 = all_bps[k]
            c1 = all_bps[k + 1]
            h = c1 - c0
            if h < 1e-12:
                continue
            
            def check_and_get_length(pos, c_val):
                lefts = pos + c_val * slopes
                rights = lefts + w
                order = np.argsort(lefts)
                lefts_s = lefts[order]
                rights_s = rights[order]
                
                # Check if single interval
                cur_r = rights_s[0]
                for idx in range(1, N):
                    if lefts_s[idx] > cur_r + 1e-14:
                        return None  # Not single interval
                    cur_r = max(cur_r, rights_s[idx])
                return cur_r - lefts_s[0]
            
            Lx0 = check_and_get_length(x, c0)
            if Lx0 is None:
                return 1e6
            Ly0 = check_and_get_length(y, c0)
            if Ly0 is None:
                return 1e6
            Lx1 = check_and_get_length(x, c1)
            if Lx1 is None:
                return 1e6
            Ly1 = check_and_get_length(y, c1)
            if Ly1 is None:
                return 1e6
            
            # Integrate Lx(c) Ly(c) from c0 to c1
            # Both are linear, so product is quadratic
            a = (Lx1 - Lx0) / h
            b = (Ly1 - Ly0) / h
            d = Lx0
            e = Ly0
            
            # Int (d + a*t)(e + b*t) dt from 0 to h
            # = de*h + (a*e + d*b)*h^2/2 + a*b*h^3/3
            vol_seg = d * e * h + (a * e + d * b) * h**2 / 2.0 + a * b * h**3 / 3.0
            total_vol += vol_seg
        
        return total_vol
    
    # Optimize x and y
    best_x = np.zeros(N)
    best_y = np.zeros(N)
    best_vol = _2d_volume(best_x, best_y, N)
    
    # Try multiple initializations
    rng_init = np.random.default_rng(42)
    
    inits = []
    # Symmetric: x_i = y_i
    inits.append((np.zeros(N), np.zeros(N)))
    inits.append((np.linspace(-0.5, 0.5, N), np.linspace(-0.5, 0.5, N)))
    inits.append((np.linspace(0.5, -0.5, N), np.linspace(0.5, -0.5, N)))
    inits.append((np.arange(N) * 0.1 - 0.35, np.arange(N) * 0.1 - 0.35))
    
    # Random symmetric
    for _ in range(20):
        xv = rng_init.uniform(-0.5, 0.5, N)
        inits.append((xv, xv.copy()))
    
    # Random asymmetric
    for _ in range(20):
        xv = rng_init.uniform(-0.5, 0.5, N)
        yv = rng_init.uniform(-0.5, 0.5, N)
        inits.append((xv, yv))
    
    for ix, iy in inits:
        if time.time() > deadline - 30:
            break
        
        def objective(params):
            x = params[:N]
            y = params[N:]
            return _2d_volume(x, y, N)
        
        try:
            res = minimize(objective, np.concatenate([ix, iy]), method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            if res.fun < best_vol:
                best_vol = res.fun
                best_x = res.x[:N].copy()
                best_y = res.x[N:].copy()
        except Exception:
            pass
    
    # Verify the result
    final_vol = _2d_volume(best_x, best_y, N)
    if final_vol >= 1e5:
        # Fallback to zeros
        best_x = np.zeros(N)
        best_y = np.zeros(N)
        final_vol = _2d_volume(best_x, best_y, N)
    
    # Construct the 64x2 array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_x[i_flat]
    base_y = best_y[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
