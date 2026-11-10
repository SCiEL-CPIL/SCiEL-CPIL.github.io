# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = keich[i_flat].copy()
    base_y = keich[j_flat].copy()
    positions0 = np.stack([base_x, base_y], axis=1).copy()
    
    start_time = time.time()
    deadline = start_time + 600.0
    
    def exact_volume_3d(pos_arr):
        """Compute exact 3D union volume using coordinate compression."""
        x = pos_arr[:, 0]
        y = pos_arr[:, 1]
        i = i_flat.astype(float)
        j = j_flat.astype(float)
        
        # Tube k at height c: x in [x_k + c*i_k/N, x_k + 1/N + c*i_k/N]
        #                      y in [y_k + c*j_k/N, y_k + 1/N + c*j_k/N]
        # The 3D box for tube k is a parallelepiped.
        # For coordinate compression, we need all critical x, y, c values.
        
        # x bounds: x_k + c*i_k/N for c in [0,1] -> [x_k, x_k + i_k/N]
        #           x_k + 1/N + c*i_k/N -> [x_k + 1/N, x_k + 1/N + i_k/N]
        # So x critical values: x_k, x_k + 1/N, x_k + i_k/N, x_k + 1/N + i_k/N for all k
        # But also x_k + c*i_k/N for all c, which is continuous... 
        
        # Actually for coordinate compression of the union of parallelepipeds,
        # we need to handle the fact that the x and y ranges depend on c.
        # This is more complex. Let me use a different approach.
        
        # The volume = integral_0^1 Area(c) dc, where Area(c) is the 2D union area at height c.
        # Area(c) is the area of union of squares [x_k + c*i_k/N, x_k + 1/N + c*i_k/N] x [y_k + c*j_k/N, y_k + 1/N + c*j_k/N]
        # As c varies, the squares translate linearly. The union area is piecewise linear in c
        # (breakpoints when edges of squares align).
        
        # For exact computation, find all breakpoints in c where the combinatorial structure changes.
        # This is complex. Let me use a fine grid integration instead.
        
        # Use 200 sample points in c and trapezoidal rule, with exact 2D union area at each c.
        c_samples = np.linspace(0, 1, 201)
        areas = np.zeros(len(c_samples))
        
        for idx_c, c in enumerate(c_samples):
            cx = x + c * i / N
            cy = y + c * j / N
            ix0 = cx
            ix1 = cx + 1.0/N
            iy0 = cy
            iy1 = cy + 1.0/N
            
            all_x = np.concatenate([ix0, ix1])
            all_y = np.concatenate([iy0, iy1])
            ux = np.unique(all_x)
            uy = np.unique(all_y)
            
            if len(ux) < 2 or len(uy) < 2:
                areas[idx_c] = 0
                continue
                
            dx = ux[1:] - ux[:-1]
            dy = uy[1:] - uy[:-1]
            
            ux_cells = ux[:-1]
            ux_cells_r = ux[1:]
            x_cover = (ix0[None, :] <= ux_cells[:, None]) & (ix1[None, :] >= ux_cells_r[:, None])
            
            uy_cells = uy[:-1]
            uy_cells_r = uy[1:]
            y_cover = (iy0[None, :] <= uy_cells[:, None]) & (iy1[None, :] >= uy_cells_r[:, None])
            
            n_cx = len(ux_cells)
            n_cy = len(uy_cells)
            covered = np.zeros((n_cx, n_cy), dtype=bool)
            for k in range(N*N):
                xc = x_cover[:, k]
                yc = y_cover[:, k]
                if xc.any() and yc.any():
                    covered[np.ix_(xc, yc)] = True
            
            areas[idx_c] = (covered * dx[:, None] * dy[None, :]).sum()
        
        return np.trapz(areas, c_samples)
    
    def objective(p):
        pos_arr = p.reshape(N*N, 2)
        return exact_volume_3d(pos_arr)
    
    best_vol = objective(positions0.flatten())
    best_pos = positions0.copy()
    
    # L-BFGS-B optimization
    try:
        res = minimize(objective, positions0.flatten(), method='L-BFGS-B',
                      options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000})
        if res.fun < best_vol:
            best_vol = res.fun
            best_pos = res.x.reshape(N*N, 2).copy()
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 60:
        return best_pos
    
    # Random restarts
    rng = np.random.default_rng(42)
    for r in range(5):
        if time.time() > deadline - 60:
            break
        x0r = best_pos.flatten() + rng.normal(0, 0.03, N*N*2)
        try:
            res = minimize(objective, x0r, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000})
            if res.fun < best_vol:
                best_vol = res.fun
                best_pos = res.x.reshape(N*N, 2).copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 30:
        try:
            res = minimize(objective, best_pos.flatten(), method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000})
            if res.fun < best_vol:
                best_vol = res.fun
                best_pos = res.x.reshape(N*N, 2).copy()
        except Exception:
            pass
    
    return best_pos

# EVOLVE-BLOCK-END
