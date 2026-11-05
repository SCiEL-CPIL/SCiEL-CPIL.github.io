"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
import sys
import numpy as np

def _char_matrix(n, d, lam):
    """Compute the 2x2 characteristic matrix for a single layer."""
    beta = 2.0 * np.pi * n * d / lam
    cb = np.cos(beta)
    sb = np.sin(beta)
    return np.array([[cb, 1.0j * sb / n],
                     [1.0j * n * sb, cb]])

def _reflectance_vec(n0, ns, mats, thicknesses, lam):
    """Compute reflectance at a single wavelength using vectorized matrix ops."""
    # Initialize M as identity
    M11, M12, M21, M22 = 1.0+0.0j, 0.0+0.0j, 0.0+0.0j, 1.0+0.0j
    
    for i in range(len(mats)):
        n = mats[i]
        d = thicknesses[i]
        beta = 2.0 * np.pi * n * d / lam
        cb = np.cos(beta)
        sb = np.sin(beta)
        
        # Matrix M_i = [[cos(beta), j*sin(beta)/n], [j*n*sin(beta), cos(beta)]]
        # New M = M @ M_i
        # M_new[0,0] = M11*M_i11 + M12*M_i21
        # M_new[0,1] = M11*M_i12 + M12*M_i22
        # M_new[1,0] = M21*M_i11 + M22*M_i21
        # M_new[1,1] = M21*M_i12 + M22*M_i22
        
        Mi11 = cb
        Mi12 = 1.0j * sb / n
        Mi21 = 1.0j * n * sb
        Mi22 = cb
        
        new_M11 = M11 * Mi11 + M12 * Mi21
        new_M12 = M11 * Mi12 + M12 * Mi22
        new_M21 = M21 * Mi11 + M22 * Mi21
        new_M22 = M21 * Mi12 + M22 * Mi22
        
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
        
    # Calculate Y
    Y = (M11 + M12 * ns) / (M21 + M22 * ns)
    r = (n0 - Y) / (n0 + Y)
    return np.abs(r)**2

def _mean_reflectance(mats, thicknesses, n0, ns, band_nm):
    """Compute mean reflectance percent over the band."""
    lam_min, lam_max = band_nm
    wls = np.linspace(lam_min, lam_max, 401)
    total = 0.0
    for lam in wls:
        total += _reflectance_vec(n0, ns, mats, thicknesses, lam)
    return 100.0 * total / len(wls)

def _project_to_feasible(x, max_layer_nm, max_total_nm):
    """Project point x to satisfy box and sum constraints."""
    x = np.clip(x, 0.0, max_layer_nm)
    total = np.sum(x)
    if total > max_total_nm:
        x = x * (max_total_nm / total)
        x = np.clip(x, 0.0, max_layer_nm)
    return x

def _nelder_mead(objective, x0, max_iter=200, timeout=30.0):
    """
    Simple Nelder-Mead implementation.
    x0: initial point
    objective: function to minimize
    """
    start_time = time.time()
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    # Initialize simplex
    simplex = [x0.copy()]
    fvals = [objective(x0)]
    
    # Create other vertices
    for i in range(n):
        xi = x0.copy()
        # Step size: 5% or 0.05 if zero
        step = 0.05 if x0[i] == 0 else 0.05 * x0[i]
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
            
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[-1] - fvals[0]) < 1e-6:
            break
            
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            # Accept reflection
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                # Outside contraction
                xc = centroid + rho * (xr - centroid)
            else:
                # Inside contraction
                xc = centroid + rho * (simplex[-1] - centroid)
            fc = objective(xc)
            
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
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
    """Design an antireflection coating using systematic initialization and Nelder-Mead."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
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
    
    # Try different numbers of layers
    # Focus on smaller stacks first as they are easier to optimize and often sufficient for broad AR
    for n_layers in [4, 6, 8, 10, 12, 14, 16]:
        if n_layers > max_layers:
            break
        if time.time() > deadline - 20:
            break
        
        # Initialize with quarter-wave stack
        labels, thicknesses = _design_quarterwave_stack(n_layers, nL, nH, band_nm)
        
        # Check initial constraints
        if sum(thicknesses) > max_total_nm:
            scale = max_total_nm / sum(thicknesses)
            thicknesses = [t * scale for t in thicknesses]
        
        mats = [nL if l == 'L' else nH for l in labels]
        
        # Initial evaluation
        score = _mean_reflectance(mats, thicknesses, n0, ns, band_nm)
        
        if score < best_score:
            best_score = score
            best_labels = labels[:]
            best_thicknesses = thicknesses[:]
        
        # Optimize thicknesses using custom Nelder-Mead
        def objective(x):
            t = _project_to_feasible(x, max_layer_nm, max_total_nm)
            return _mean_reflectance(mats, t, n0, ns, band_nm)
        
        x0 = np.array(thicknesses, dtype=float)
        
        try:
            # Allocate more time for larger stacks if possible, but keep total under control
            # Simple heuristic: 10s per stack size, min 5s
            time_budget = 10.0
            if n_layers > 10:
                time_budget = 5.0
                
            x_opt, f_opt = _nelder_mead(objective, x0, max_iter=300, timeout=time_budget)
            
            # Final check and update
            x_opt = _project_to_feasible(x_opt, max_layer_nm, max_total_nm)
            opt_score = objective(x_opt)
            
            if opt_score < best_score:
                best_score = opt_score
                best_labels = labels[:]
                best_thicknesses = x_opt.tolist()
                
        except Exception:
            pass
        
    # If we haven't found a good solution, fall back to simple quarter-wave
    if best_labels is None:
        labels, thicknesses = _design_quarterwave_stack(4, nL, nH, band_nm)
        if sum(thicknesses) > max_total_nm:
            scale = max_total_nm / sum(thicknesses)
            thicknesses = [t * scale for t in thicknesses]
        best_labels = labels
        best_thicknesses = thicknesses
    
    # Ensure constraints are satisfied
    best_thicknesses = [max(0, min(t, max_layer_nm)) for t in best_thicknesses]
    total = sum(best_thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_labels, best_thicknesses
# EVOLVE-BLOCK-END
