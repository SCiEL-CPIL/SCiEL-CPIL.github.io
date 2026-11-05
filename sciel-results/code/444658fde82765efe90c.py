"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _mean_reflectance_vectorized(mats, thicknesses, n0, ns, band_nm):
    """
    Compute mean reflectance percent over the band using vectorized operations.
    mats: list of refractive indices
    thicknesses: list of thicknesses
    n0: ambient index
    ns: substrate index
    band_nm: [min_nm, max_nm]
    """
    lam_min, lam_max = band_nm
    wls = np.linspace(lam_min, lam_max, 401)
    
    # Initialize M components as arrays of ones/zeros
    # M = [[M11, M12], [M21, M22]]
    M11 = np.ones_like(wls, dtype=complex)
    M12 = np.zeros_like(wls, dtype=complex)
    M21 = np.zeros_like(wls, dtype=complex)
    M22 = np.ones_like(wls, dtype=complex)
    
    for i in range(len(mats)):
        n = mats[i]
        d = thicknesses[i]
        
        # beta = 2 * pi * n * d / lam
        beta = 2.0 * np.pi * n * d / wls
        cb = np.cos(beta)
        sb = np.sin(beta)
        
        # M_i components
        Mi11 = cb
        Mi12 = 1.0j * sb / n
        Mi21 = 1.0j * n * sb
        Mi22 = cb
        
        # M_new = M_old @ M_i
        # M11_new = M11*Mi11 + M12*Mi21
        # M12_new = M11*Mi12 + M12*Mi22
        # M21_new = M21*Mi11 + M22*Mi21
        # M22_new = M21*Mi12 + M22*Mi22
        
        new_M11 = M11 * Mi11 + M12 * Mi21
        new_M12 = M11 * Mi12 + M12 * Mi22
        new_M21 = M21 * Mi11 + M22 * Mi21
        new_M22 = M21 * Mi12 + M22 * Mi22
        
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
        
    # Calculate Y
    Y = (M11 + M12 * ns) / (M21 + M22 * ns)
    r = (n0 - Y) / (n0 + Y)
    R = np.abs(r)**2
    
    return np.mean(R) * 100.0

def _project_to_feasible(x, max_layer_nm, max_total_nm):
    """Project point x to satisfy box and sum constraints."""
    x = np.array(x, dtype=float)
    x = np.clip(x, 0.0, max_layer_nm)
    total = np.sum(x)
    if total > max_total_nm:
        x = x * (max_total_nm / total)
        x = np.clip(x, 0.0, max_layer_nm)
    return x

def _nelder_mead_fast(objective, x0, max_iter=200, timeout=10.0):
    """
    Optimized Nelder-Mead with timeout.
    """
    start_time = time.time()
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    simplex = [x0.copy()]
    fvals = [objective(x0)]
    
    for i in range(n):
        xi = x0.copy()
        step = 0.1 if x0[i] == 0 else 0.1 * x0[i]
        # Ensure step is not too large initially to stay in bounds roughly
        if step > 10: step = 10
        xi[i] += step
        simplex.append(xi)
        fvals.append(objective(xi))
        
    best_idx = 0
    for i in range(1, n+1):
        if fvals[i] < fvals[best_idx]:
            best_idx = i
            
    for k in range(max_iter):
        if time.time() - start_time > timeout:
            break
            
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < 1e-5:
            break
            
        centroid = np.mean(simplex[:-1], axis=0)
        
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            if fr < fvals[-1]:
                xc = centroid + rho * (xr - centroid)
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
            fc = objective(xc)
            
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
                
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def _design_quarterwave_stack(n_layers, nL, nH, band_nm):
    """Design a quarter-wave stack for the center wavelength."""
    center = 0.5 * (band_nm[0] + band_nm[1])
    labels = []
    thicknesses = []
    for i in range(n_layers):
        if i % 2 == 0:
            labels.append('L')
            thicknesses.append(center / (4.0 * nL))
        else:
            labels.append('H')
            thicknesses.append(center / (4.0 * nH))
    return labels, thicknesses

def design(problem):
    """Design an antireflection coating."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    best_labels = None
    best_thicknesses = None
    best_score = float('inf')
    
    # Determine layer counts to try
    # Start with smaller counts, go up
    layer_counts = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    layer_counts = [n for n in layer_counts if n <= max_layers]
    
    # Allocate time roughly proportional to complexity? 
    # Actually, more layers = more variables = slower NM.
    # But more layers = potentially better solution.
    # Let's give smaller stacks more restarts and larger stacks fewer.
    
    for n_layers in layer_counts:
        if time.time() > deadline - 10:
            break
            
        # Create labels
        labels = []
        for i in range(n_layers):
            if i % 2 == 0:
                labels.append('L')
            else:
                labels.append('H')
        
        mats = [nL if l == 'L' else nH for l in labels]
        
        def objective(x):
            t = _project_to_feasible(x, max_layer_nm, max_total_nm)
            return _mean_reflectance_vectorized(mats, t, n0, ns, band_nm)
        
        # Generate initial points
        initial_points = []
        
        # 1. Quarter-wave
        _, qw_thick = _design_quarterwave_stack(n_layers, nL, nH, band_nm)
        qw_thick = _project_to_feasible(qw_thick, max_layer_nm, max_total_nm)
        initial_points.append(np.array(qw_thick))
        
        # 2. Random starts
        # Number of restarts depends on remaining time and n_layers
        # Estimate cost: n_layers * 401 * ops.
        # Let's do 5 restarts for small, 2 for large.
        n_restarts = 5 if n_layers <= 10 else 2
        
        for _ in range(n_restarts):
            if time.time() > deadline - 5:
                break
            # Random thicknesses
            x_rand = np.random.uniform(0, max_layer_nm, n_layers)
            x_rand = _project_to_feasible(x_rand, max_layer_nm, max_total_nm)
            initial_points.append(x_rand)
        
        # Optimize each initial point
        for x0 in initial_points:
            if time.time() > deadline - 5:
                break
                
            # Allocate time per optimization
            # Total time left / remaining iterations
            time_left = deadline - time.time()
            # Heuristic: 5s per run
            t_budget = min(5.0, time_left * 0.8)
            if t_budget < 1.0:
                continue
                
            try:
                x_opt, f_opt = _nelder_mead_fast(objective, x0, max_iter=100, timeout=t_budget)
                x_opt = _project_to_feasible(x_opt, max_layer_nm, max_total_nm)
                f_opt = objective(x_opt)
                
                if f_opt < best_score:
                    best_score = f_opt
                    best_labels = labels[:]
                    best_thicknesses = x_opt.tolist()
            except Exception:
                pass
                
    # Fallback
    if best_labels is None:
        labels, thicknesses = _design_quarterwave_stack(4, nL, nH, band_nm)
        thicknesses = _project_to_feasible(thicknesses, max_layer_nm, max_total_nm)
        best_labels = labels
        best_thicknesses = thicknesses.tolist()
        best_score = _mean_reflectance_vectorized([nL if l=='L' else nH for l in best_labels], best_thicknesses, n0, ns, band_nm)
        
    # Final constraint check
    best_thicknesses = [max(0, min(t, max_layer_nm)) for t in best_thicknesses]
    total = sum(best_thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        best_thicknesses = [t * scale for t in best_thicknesses]
        
    return best_labels, best_thicknesses
# EVOLVE-BLOCK-END
