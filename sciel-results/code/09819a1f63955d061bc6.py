import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

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
        
        def union_length(c_val):
            lefts = x + c_val * slopes
            rights = lefts + w
            order = np.argsort(lefts)
            lefts_s = lefts[order]
            rights_s = rights[order]
            
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
        total_vol += 0.5 * (L0 + L1) * h
    
    return total_vol

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 550
    
    N = 8
    
    def _2d_volume(x, y, N=8):
        """
        Compute the 2D union volume assuming single-rectangle structure.
        If the structure breaks, return a penalty.
        """
        w = 1.0 / N
        slopes = np.arange(N, dtype=float) / N
        
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
                        return None
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
            
            a = (Lx1 - Lx0) / h
            b = (Ly1 - Ly0) / h
            d = Lx0
            e = Ly0
            
            vol_seg = d * e * h + (a * e + d * b) * h**2 / 2.0 + a * b * h**3 / 3.0
            total_vol += vol_seg
        
        return total_vol

    # We optimize 14 variables: x[1..7] and y[1..7]. x[0]=0, y[0]=0.
    # This reduces dimensionality and handles translation invariance.
    
    def objective(params):
        x = np.zeros(N)
        y = np.zeros(N)
        x[1:] = params[:N-1]
        y[1:] = params[N-1:]
        return _2d_volume(x, y, N)
    
    best_params = None
    best_vol = float('inf')
    
    # Strategy 1: Differential Evolution for global search
    # Bounds: reasonable range for relative positions.
    # The "width" of the set is roughly 1. Positions should be within [-1, 1].
    bounds = [(-1.0, 1.0)] * (2 * (N - 1))
    
    try:
        if time.time() < deadline - 100:
            de_res = differential_evolution(
                objective, 
                bounds, 
                maxiter=50, 
                popsize=15, 
                tol=0.01, 
                seed=42,
                polish=False # We will polish manually
            )
            if de_res.fun < best_vol:
                best_vol = de_res.fun
                best_params = de_res.x
    except Exception:
        pass
    
    # Strategy 2: Local polish with Nelder-Mead
    # Start from DE result, and a few symmetric guesses
    
    candidates = []
    if best_params is not None:
        candidates.append(best_params)
    
    # Symmetric guess: x_i = y_i = (i - (N-1)/2) * delta
    # Try a few delta values
    for delta in [0.1, 0.2, 0.5, 1.0]:
        x_guess = np.linspace(-delta, delta, N)
        x_guess[0] = 0.0 # Force x[0]=0 for param space
        # Adjust so x[0] is actually 0? No, x[0] is fixed to 0 in objective.
        # So we just take x_guess[1:]
        p = np.concatenate([x_guess[1:], x_guess[1:]])
        candidates.append(p)
        
    # Random guesses
    rng = np.random.default_rng(123)
    for _ in range(5):
        p = rng.uniform(-0.5, 0.5, 2*(N-1))
        candidates.append(p)
        
    for p0 in candidates:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(
                objective, 
                p0, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Construct final x, y
    if best_params is None:
        # Fallback
        best_params = np.zeros(2*(N-1))
        
    x = np.zeros(N)
    y = np.zeros(N)
    x[1:] = best_params[:N-1]
    y[1:] = best_params[N-1:]
    
    # Verify
    final_vol = _2d_volume(x, y, N)
    if final_vol >= 1e5:
        # Fallback to simple symmetric
        x = np.linspace(-0.5, 0.5, N)
        y = np.linspace(-0.5, 0.5, N)
        x[0] = 0.0
        y[0] = 0.0
        # Re-center? No, just use these.
        
    # Construct the 64x2 array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x[i_flat]
    base_y = y[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
