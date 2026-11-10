import numpy as np
import scipy.optimize as opt
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    time_limit = 630.0
    margin = 30.0
    
    def union_area_2d(rects):
        """Compute exact union area of axis-aligned rectangles.
        rects: (M, 4) array [x0, x1, y0, y1]
        """
        if len(rects) == 0:
            return 0.0
        
        x0s = rects[:, 0]
        x1s = rects[:, 1]
        y0s = rects[:, 2]
        y1s = rects[:, 3]
        
        # Get unique x boundaries
        x_coords = np.unique(np.concatenate([x0s, x1s]))
        
        if len(x_coords) < 2:
            return 0.0
        
        area = 0.0
        for i in range(len(x_coords) - 1):
            xl = x_coords[i]
            xr = x_coords[i + 1]
            w = xr - xl
            if w <= 0:
                continue
            
            # Find rects that cover this x-slab
            mask = (x0s <= xl + 1e-12) & (x1s >= xr - 1e-12)
            if not np.any(mask):
                continue
            
            ay0 = y0s[mask]
            ay1 = y1s[mask]
            
            # Union length of y-intervals
            idx = np.argsort(ay0)
            ay0 = ay0[idx]
            ay1 = ay1[idx]
            
            ylen = 0.0
            cs = ay0[0]
            ce = ay1[0]
            for j in range(1, len(ay0)):
                if ay0[j] <= ce + 1e-12:
                    if ay1[j] > ce:
                        ce = ay1[j]
                else:
                    ylen += ce - cs
                    cs = ay0[j]
                    ce = ay1[j]
            ylen += ce - cs
            
            area += w * ylen
        
        return area
    
    def volume(p, c_grid=None):
        """Compute union volume exactly by integrating area over c."""
        x = p[:N]
        y = p[N:]
        
        if c_grid is None:
            c_grid = np.linspace(0, 1, 200)
        
        total = 0.0
        dc = c_grid[1] - c_grid[0]
        
        for c in c_grid:
            rects = np.empty((N*N, 4))
            k = 0
            for i in range(N):
                rx0 = x[i] + c * i / N
                rx1 = rx0 + 1.0 / N
                for j in range(N):
                    ry0 = y[j] + c * j / N
                    ry1 = ry0 + 1.0 / N
                    rects[k] = [rx0, rx1, ry0, ry1]
                    k += 1
            total += union_area_2d(rects)
        
        # Trapezoidal rule
        # Actually, let's use simple Riemann sum since we have uniform grid
        # Better: use trapezoidal
        areas = np.zeros(len(c_grid))
        for idx, c in enumerate(c_grid):
            rects = np.empty((N*N, 4))
            k = 0
            for i in range(N):
                rx0 = x[i] + c * i / N
                rx1 = rx0 + 1.0 / N
                for j in range(N):
                    ry0 = y[j] + c * j / N
                    ry1 = ry0 + 1.0 / N
                    rects[k] = [rx0, rx1, ry0, ry1]
                    k += 1
            areas[idx] = union_area_2d(rects)
        
        return np.trapz(areas, c_grid)
    
    # Precompute c grid for speed
    c_fine = np.linspace(0, 1, 150)
    
    def vol_fast(p):
        x = p[:N]
        y = p[N:]
        areas = np.empty(len(c_fine))
        for idx, c in enumerate(c_fine):
            rects = np.empty((N*N, 4))
            k = 0
            for i in range(N):
                rx0 = x[i] + c * i / N
                rx1 = rx0 + 1.0 / N
                for j in range(N):
                    ry0 = y[j] + c * j / N
                    ry1 = ry0 + 1.0 / N
                    rects[k] = [rx0, rx1, ry0, ry1]
                    k += 1
            areas[idx] = union_area_2d(rects)
        return np.trapz(areas, c_fine)
    
    # Keich initial construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    p0 = np.concatenate([keich, keich])
    
    best_p = p0.copy()
    best_vol = vol_fast(p0)
    
    rng = np.random.default_rng(123)
    
    # Generate diverse initial points
    inits = [p0.copy()]
    # Perturbations
    for _ in range(15):
        inits.append(p0 + rng.uniform(-0.3, 0.3, size=2*N))
    # Random
    inits.append(rng.uniform(-0.5, 1.5, size=2*N))
    inits.append(rng.uniform(-0.5, 1.5, size=2*N))
    # Structured
    inits.append(np.concatenate([np.linspace(-0.2, 1.2, N), np.linspace(-0.2, 1.2, N)]))
    inits.append(np.concatenate([np.linspace(0, 1, N), np.linspace(0, 1, N)]))
    
    for p_init in inits:
        if time.time() - start_time > time_limit - margin:
            break
        try:
            res = opt.minimize(vol_fast, p_init, method='L-BFGS-B',
                               options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12})
            v = vol_fast(res.x)
            if v < best_vol:
                best_vol = v
                best_p = res.x.copy()
        except Exception:
            continue
    
    # Polish with Nelder-Mead for non-smooth regions
    if time.time() - start_time < time_limit - margin:
        try:
            res2 = opt.minimize(vol_fast, best_p, method='Nelder-Mead',
                                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
            v2 = vol_fast(res2.x)
            if v2 < best_vol:
                best_vol = v2
                best_p = res2.x.copy()
        except Exception:
            pass
    
    # Final L-BFGS polish
    if time.time() - start_time < time_limit - margin:
        try:
            res3 = opt.minimize(vol_fast, best_p, method='L-BFGS-B',
                                options={'maxiter': 1000, 'gtol': 1e-10, 'ftol': 1e-14})
            v3 = vol_fast(res3.x)
            if v3 < best_vol:
                best_vol = v3
                best_p = res3.x.copy()
        except Exception:
            pass
    
    x_opt = best_p[:N]
    y_opt = best_p[N:]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    if not np.all(np.isfinite(positions)):
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
    
    return positions
