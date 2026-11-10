# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

_T0 = time.time()
_DL = 600.0
_MARGIN = 30.0


def _deadline_ok():
    return (time.time() - _T0) < (_DL - _MARGIN)


def _keich(N):
    pos = int(round(np.log2(N)))
    j_terms = np.arange(1, pos + 1, dtype=float)
    out = np.empty(N, dtype=np.float64)
    for idx in range(N):
        bits = [int(b) for b in bin(idx)[2:]]
        bits = [0] * (pos - len(bits)) + bits
        out[idx] = np.sum((1.0 - j_terms) * np.array(bits) * (2.0 ** -j_terms) / pos)
    return out


def _keich_positions(N):
    keich = _keich(N)
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    base_x = keich[i_grid.flatten()]
    base_y = keich[j_grid.flatten()]
    return np.stack([base_x, base_y], axis=1)


def _vol1d(p, N):
    c = np.linspace(0.0, 1.0, 1025)
    cc = c[:, None]
    L = p[None, :] + cc * (np.arange(N) / N)
    R = L + 1.0 / N
    a = np.min(L, axis=1)
    b = np.max(R, axis=1)
    inside = (c[:, None] >= L) & (c[:, None] < R)
    cov = np.count_nonzero(inside, axis=1)
    length = np.where(cov > 0, b - a, 0.0)
    return 0.5 * (length[0] + length[-1] + 2.0 * np.sum(length[1:-1])) / 1024.0


def _vol3d_fast(pos, N, M=200):
    """Fast analytical 3D volume estimate using product structure.
    pos is (64, 2) array. Returns estimated union volume."""
    N2 = N * N
    if pos.shape != (N2, 2):
        return np.inf
    
    # Extract 1D positions
    # Row k: i = k // N, j = k % N
    # x_k depends on i, y_k depends on j in product structure
    # But for general pos, we can't assume separability
    # Use Monte Carlo with fewer samples for speed
    
    x = pos[:, 0]
    y = pos[:, 1]
    i_idx = np.arange(N2) // N
    j_idx = np.arange(N2) % N
    
    # Compute bounding box
    X0 = np.min(x)
    X1 = np.max(x + (i_idx + 1) / N)
    Y0 = np.min(y)
    Y1 = np.max(y + (j_idx + 1) / N)
    
    area_box = (X1 - X0) * (Y1 - Y0)
    if area_box <= 1e-9:
        return 1e10
    
    # Use grid-based Monte Carlo for speed
    # Sample c values
    c_samples = np.linspace(0.05, 0.95, M)  # avoid endpoints for stability
    
    # For each c, compute 2D coverage
    # At height c, tube k occupies:
    # [x_k + c*i_k/N, x_k + c*i_k/N + 1/N] x [y_k + c*j_k/N, y_k + c*j_k/N + 1/N]
    
    # Use a 2D grid for each c
    nx_grid = 40
    ny_grid = 40
    dx = (X1 - X0) / nx_grid
    dy = (Y1 - Y0) / ny_grid
    
    # Grid centers
    gx = X0 + (np.arange(nx_grid) + 0.5) * dx
    gy = Y0 + (np.arange(ny_grid) + 0.5) * dy
    GX, GY = np.meshgrid(gx, gy, indexing='xy')
    GX = GX.ravel()
    GY = GY.ravel()
    
    total_area = 0.0
    for c in c_samples:
        # Check which tubes cover each grid point
        # Tube k covers point (px, py) if:
        # x_k + c*i_k/N <= px < x_k + c*i_k/N + 1/N
        # y_k + c*j_k/N <= py < y_k + c*j_k/N + 1/N
        
        # Vectorized check
        # Shifted positions
        xs = x + c * i_idx / N
        ys = y + c * j_idx / N
        
        # For each grid point, check if inside any tube
        # (len(GX), N2) boolean matrix
        in_x = (GX[:, None] >= xs[None, :]) & (GX[:, None] < xs[None, :] + 1.0/N)
        in_y = (GY[:, None] >= ys[None, :]) & (GY[:, None] < ys[None, :] + 1.0/N)
        
        covered = (in_x & in_y).any(axis=1)
        area_c = covered.sum() * dx * dy
        total_area += area_c
    
    # Integrate over c using trapezoidal rule
    V = total_area / M * 1.0  # height is 1
    return V


def _vol3d_exact(pos, N, n_c=50, n_grid=25):
    """More accurate 3D volume estimate."""
    N2 = N * N
    x = pos[:, 0]
    y = pos[:, 1]
    i_idx = np.arange(N2) // N
    j_idx = np.arange(N2) % N
    
    X0 = np.min(x)
    X1 = np.max(x + (i_idx + 1) / N)
    Y0 = np.min(y)
    Y1 = np.max(y + (j_idx + 1) / N)
    
    area_box = (X1 - X0) * (Y1 - Y0)
    if area_box <= 1e-9:
        return 1e10
    
    c_samples = np.linspace(0.0, 1.0, n_c)
    nx_grid = n_grid
    ny_grid = n_grid
    dx = (X1 - X0) / nx_grid
    dy = (Y1 - Y0) / ny_grid
    
    gx = X0 + (np.arange(nx_grid) + 0.5) * dx
    gy = Y0 + (np.arange(ny_grid) + 0.5) * dy
    GX, GY = np.meshgrid(gx, gy, indexing='xy')
    GX = GX.ravel()
    GY = GY.ravel()
    
    total_area = 0.0
    for c in c_samples:
        xs = x + c * i_idx / N
        ys = y + c * j_idx / N
        
        in_x = (GX[:, None] >= xs[None, :]) & (GX[:, None] < xs[None, :] + 1.0/N)
        in_y = (GY[:, None] >= ys[None, :]) & (GY[:, None] < ys[None, :] + 1.0/N)
        
        covered = (in_x & in_y).any(axis=1)
        area_c = covered.sum() * dx * dy
        total_area += area_c
    
    # Trapezoidal integration over c
    if n_c > 1:
        dc = 1.0 / (n_c - 1)
        V = (0.5 * total_area[0] + total_area[1:-1].sum() + 0.5 * total_area[-1]) * dc if n_c > 2 else total_area
    else:
        V = total_area
    
    # Recompute properly
    areas = []
    for c in c_samples:
        xs = x + c * i_idx / N
        ys = y + c * j_idx / N
        in_x = (GX[:, None] >= xs[None, :]) & (GX[:, None] < xs[None, :] + 1.0/N)
        in_y = (GY[:, None] >= ys[None, :]) & (GY[:, None] < ys[None, :] + 1.0/N)
        covered = (in_x & in_y).any(axis=1)
        areas.append(covered.sum() * dx * dy)
    
    areas = np.array(areas)
    if len(areas) > 1:
        V = np.trapz(areas, c_samples)
    else:
        V = areas[0] if len(areas) > 0 else 0.0
    
    return V


def _make_1d(N, seed):
    rng = np.random.default_rng(seed)
    c = np.linspace(0.0, 1.0, 1025)
    cc = c[:, None]
    slopes = np.arange(N) / N
    Lm = (1.0 / N) * (1.0 - cc) * slopes
    Rm = (1.0 / N) * (1.0 - cc) * slopes
    lo = Lm.min(axis=0)
    hi = Rm.max(axis=0)
    p0 = np.linspace(lo, hi, N) + rng.uniform(-0.02, 0.02, N)
    return p0, lo, hi


def _opt_1d(N, seed):
    from scipy.optimize import minimize
    p0, lo, hi = _make_1d(N, seed)
    res = minimize(_vol1d, p0, args=(N,), method="L-BFGS-B",
                   bounds=list(zip(lo, hi)),
                   options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-10, "maxls": 40})
    p = res.x
    if _deadline_ok():
        r2 = minimize(_vol1d, p, args=(N,), method="L-BFGS-B",
                      bounds=list(zip(lo, hi)),
                      options={"maxiter": 200, "ftol": 1e-13, "gtol": 1e-12, "maxls": 40})
        if r2.success:
            p = r2.x
    return p


def _separable_volume(pos, N):
    """Compute 3D volume assuming separable structure (x depends only on i, y only on j)."""
    N2 = N * N
    x = pos[:, 0]
    y = pos[:, 1]
    i_idx = np.arange(N2) // N
    j_idx = np.arange(N2) % N
    
    # Extract 1D positions
    px = np.array([x[i*N] for i in range(N)])
    py = np.array([y[j] for j in range(N)])
    
    ax = _vol1d(px, N)
    ay = _vol1d(py, N)
    
    # Bounding box area factor
    X0 = np.min(x)
    X1 = np.max(x + (i_idx + 1) / N)
    Y0 = np.min(y)
    Y1 = np.max(y + (j_idx + 1) / N)
    box_area = (X1 - X0) * (Y1 - Y0)
    
    # The actual volume from MC is: fraction_in * box_area
    # For separable case, fraction_in = (A_x / (X1-X0)) * (A_y / (Y1-Y0))
    # So V = A_x * A_y
    return ax * ay


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    try:
        keich = _keich(N)
        
        # First, get good separable 1D solutions as initial guess
        p_x_best = None
        v_x_best = np.inf
        for seed in (0, 1, 2):
            if not _deadline_ok():
                break
            try:
                p = _opt_1d(N, seed)
                v = _vol1d(p, N)
                if np.isfinite(v) and v < v_x_best:
                    v_x_best = v
                    p_x_best = p.copy()
            except Exception:
                pass
        if p_x_best is None:
            p_x_best = keich.copy()

        p_y_best = None
        v_y_best = np.inf
        for seed in (3, 4, 5):
            if not _deadline_ok():
                break
            try:
                p = _opt_1d(N, seed)
                v = _vol1d(p, N)
                if np.isfinite(v) and v < v_y_best:
                    v_y_best = v
                    p_y_best = p.copy()
            except Exception:
                pass
        if p_y_best is None:
            p_y_best = keich.copy()

        # Build initial separable position
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        base_x = p_x_best[i_grid.flatten()]
        base_y = p_y_best[j_grid.flatten()]
        pos_init = np.stack([base_x, base_y], axis=1)
        
        # Try non-separable optimization using differential evolution
        # But this is 128D - DE will be very slow
        # Instead, try a local perturbation approach: add small random perturbations
        # to break separability and see if it helps
        
        best_pos = pos_init.copy()
        best_vol = _separable_volume(pos_init, N)
        
        # Try several perturbations
        rng = np.random.default_rng(42)
        for trial in range(5):
            if not _deadline_ok():
                break
            # Create perturbed version
            pos_trial = pos_init.copy()
            # Add small perturbations to break separability
            noise = rng.normal(0, 0.01, (64, 2))
            pos_trial = pos_init + noise
            
            # Evaluate using fast method
            vol_trial = _vol3d_fast(pos_trial, N, M=50)
            
            if np.isfinite(vol_trial) and vol_trial < best_vol:
                best_vol = vol_trial
                best_pos = pos_trial.copy()
        
        # If DE is available and we have time, try a coarse DE search
        if _deadline_ok() and (time.time() - _T0) < (_DL - _MARGIN - 120):
            try:
                # Use differential evolution on a reduced parameterization
                # Represent as: base separable + low-rank perturbation
                # pos[i,j] = (px[i], py[j]) + delta[i,j]
                # Use a small number of perturbation parameters
                
                def _obj_perturb(params):
                    if not _deadline_ok():
                        return 1e10
                    # params: first 8 are x perturbations per i, next 8 are y perturbations per j
                    # plus 4 more for cross terms
                    px_pert = params[:8]
                    py_pert = params[8:16]
                    cross = params[16:20] if len(params) > 20 else np.zeros(4)
                    
                    pos = pos_init.copy()
                    for i in range(N):
                        pos[i*N:(i+1)*N, 0] += px_pert[i]
                    for j in range(N):
                        pos[j::N, 1] += py_pert[j]
                    # Add small cross terms
                    for k in range(4):
                        idx = k * 16
                        pos[idx, 0] += cross[k]
                        pos[idx+1, 1] += cross[k]
                    
                    vol = _vol3d_fast(pos, N, M=30)
                    return vol if np.isfinite(vol) else 1e10
                
                bounds = [(-0.05, 0.05)] * 16 + [(-0.02, 0.02)] * 4
                res_de = differential_evolution(_obj_perturb, bounds, seed=42,
                                                 maxiter=50, popsize=15,
                                                 tol=1e-6, mutation=(0.5, 1.5),
                                                 recombination=0.8, polish=False,
                                                 update_interval=5)
                if res_de.success or res_de.fun < best_vol:
                    params = res_de.x
                    px_pert = params[:8]
                    py_pert = params[8:16]
                    cross = params[16:20] if len(params) > 20 else np.zeros(4)
                    
                    pos_de = pos_init.copy()
                    for i in range(N):
                        pos_de[i*N:(i+1)*N, 0] += px_pert[i]
                    for j in range(N):
                        pos_de[j::N, 1] += py_pert[j]
                    for k in range(4):
                        idx = k * 16
                        pos_de[idx, 0] += cross[k]
                        pos_de[idx+1, 1] += cross[k]
                    
                    vol_de = _vol3d_fast(pos_de, N, M=50)
                    if np.isfinite(vol_de) and vol_de < best_vol:
                        best_vol = vol_de
                        best_pos = pos_de.copy()
                        
            except Exception:
                pass
        
        # Final validation and return
        positions = best_pos
        if not (np.all(np.isfinite(positions)) and positions.shape == (N * N, 2)):
            positions = _keich_positions(N)
        return positions
    except Exception:
        return _keich_positions(N)


# EVOLVE-BLOCK-END
