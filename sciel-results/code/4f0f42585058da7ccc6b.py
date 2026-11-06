"""Broadband antireflection coating design via gradient-based optimization."""
import numpy as np
import time
import math

def _char_matrix(n, d, w, k=0.0):
    """Characteristic matrix for a single layer at wavelength w (nm)."""
    beta = 2.0 * np.pi * n * d / w
    # For normal incidence, the matrix is [[cos(beta), i*sin(beta)/n], [i*n*sin(beta), cos(beta)]]
    # We use the form: M = [[cos, sin/(n*beta0)], [n*beta0*sin, cos]] where beta0 = 2*pi/w
    # Actually standard form: M = [[cos(beta), i*sin(beta)/n], [i*n*sin(beta), cos(beta)]]
    c = np.cos(beta)
    s = np.sin(beta)
    return np.array([[c, 1j * s / n], [1j * n * s, c]], dtype=complex)


def _reflectance(materials, thicknesses, w, n_ambient, n_substrate):
    """Compute reflectance (0 to 1) at wavelength w using characteristic matrix method."""
    if len(materials) == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return r * r
    
    # Start with substrate matrix
    M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    
    # Multiply layer matrices from top (air side) to bottom
    for i in range(len(materials)):
        n = materials[i]
        d = thicknesses[i]
        Mi = _char_matrix(n, d, w)
        M = Mi @ M
    
    # Total matrix M = [[M11, M12], [M21, M22]]
    M11, M12 = M[0, 0], M[0, 1]
    M21, M22 = M[1, 0], M[1, 1]
    
    # Reflectance formula:
    # Y = (M21 + M22 * n_sub) / (M11 + M12 * n_sub)
    # r = (n_amb * Y - n_sub) / (n_amb * Y + n_sub)
    # R = |r|^2
    Y = (M21 + M22 * n_substrate) / (M11 + M12 * n_substrate)
    r = (n_ambient * Y - n_substrate) / (n_ambient * Y + n_substrate)
    R = np.abs(r) ** 2
    return float(np.real(R))


def _mean_reflectance(materials, thicknesses, wavelengths, n_ambient, n_substrate):
    """Compute mean reflectance in percent over the given wavelengths."""
    total = 0.0
    for w in wavelengths:
        total += _reflectance(materials, thicknesses, w, n_ambient, n_substrate)
    return 100.0 * total / len(wavelengths)


def _gradient(materials, thicknesses, wavelengths, n_ambient, n_substrate, eps=1e-6):
    """Compute gradient of mean reflectance w.r.t. each thickness using central differences."""
    n = len(thicknesses)
    grad = np.zeros(n)
    base = _mean_reflectance(materials, thicknesses, wavelengths, n_ambient, n_substrate)
    
    for i in range(n):
        # Central difference
        t_plus = list(thicknesses)
        t_minus = list(thicknesses)
        t_plus[i] += eps
        t_minus[i] -= eps
        f_plus = _mean_reflectance(materials, t_plus, wavelengths, n_ambient, n_substrate)
        f_minus = _mean_reflectance(materials, t_minus, wavelengths, n_ambient, n_substrate)
        grad[i] = (f_plus - f_minus) / (2.0 * eps)
    
    return grad


def _l_bfgs_b(materials, thicknesses, wavelengths, n_ambient, n_substrate, max_iter=200, tol=1e-8, max_layer_nm=400.0, max_total_nm=1000.0):
    """Simple L-BFGS-B implementation with box constraints."""
    n = len(thicknesses)
    x = np.array(thicknesses, dtype=float)
    
    # L-BFGS-B parameters
    m = 10  # memory size
    s = []  # previous steps
    y = []  # previous gradient differences
    rho = []  # 1/(y^T s)
    
    g = _gradient(materials, list(x), wavelengths, n_ambient, n_substrate)
    f = _mean_reflectance(materials, list(x), wavelengths, n_ambient, n_substrate)
    
    # Simple 2-point L-BFGS
    for it in range(max_iter):
        # Check convergence
        if np.max(np.abs(g)) < tol:
            break
        
        # Compute search direction using L-BFGS two-loop recursion
        # Simplified: use steepest descent with line search for robustness
        # Actually let's do a simple projected gradient method with line search
        
        # Projected gradient descent with backtracking line search
        alpha = 1.0
        x_new = x - alpha * g
        
        # Project to feasible region
        for i in range(n):
            x_new[i] = max(0.0, min(max_layer_nm, x_new[i]))
        
        # Check total thickness constraint
        total = np.sum(x_new)
        if total > max_total_nm:
            scale = max_total_nm / total
            x_new *= scale
        
        # Backtracking line search
        f_new = _mean_reflectance(materials, list(x_new), wavelengths, n_ambient, n_substrate)
        c1 = 1e-4
        while f_new > f + c1 * alpha * np.dot(g, x_new - x):
            alpha *= 0.5
            x_new = x - alpha * g
            for i in range(n):
                x_new[i] = max(0.0, min(max_layer_nm, x_new[i]))
            total = np.sum(x_new)
            if total > max_total_nm:
                scale = max_total_nm / total
                x_new *= scale
            f_new = _mean_reflectance(materials, list(x_new), wavelengths, n_ambient, n_substrate)
            if alpha < 1e-10:
                break
        
        # Update history for L-BFGS
        s.append(x_new - x)
        y.append(_gradient(materials, list(x_new), wavelengths, n_ambient, n_substrate) - g)
        rho.append(1.0 / max(1e-10, np.dot(y[-1], s[-1])))
        
        if len(s) > m:
            s.pop(0)
            y.pop(0)
            rho.pop(0)
        
        x = x_new
        g = _gradient(materials, list(x), wavelengths, n_ambient, n_substrate)
        f = f_new
    
    return x, f


def _optimize_stack(n_layers, materials, wavelengths, n_ambient, n_substrate, max_layer_nm, max_total_nm, start_time, deadline):
    """Optimize a stack with fixed number of layers using multiple restarts."""
    best_x = None
    best_f = float('inf')
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random initial guess
        if restart == 0:
            # Start with quarter-wave stack
            centre_w = 0.5 * (wavelengths[0] + wavelengths[-1])
            x = np.array([(centre_w / (4.0 * materials[i])) for i in range(n_layers)])
        else:
            # Random perturbation
            x = np.random.uniform(0.1, max_layer_nm * 0.5, size=n_layers)
        
        # Optimize
        x_opt, f_opt = _l_bfgs_b(materials, list(x), wavelengths, n_ambient, n_substrate, 
                                   max_iter=100, max_layer_nm=max_layer_nm, max_total_nm=max_total_nm)
        
        if f_opt < best_f:
            best_f = f_opt
            best_x = x_opt
    
    return best_x, best_f


def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    materials_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Create wavelength grid (401 points)
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Material indices
    n_L = materials_dict['L']
    n_H = materials_dict['H']
    
    # Try different numbers of layers and material sequences
    # Common AR coating structures: L-H, H-L, L-H-L, H-L-H, etc.
    
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    # Strategy: try various layer counts and material patterns
    # For each, optimize the thicknesses
    
    # Pattern 1: Alternating L and H starting with L
    for n_layers in range(2, min(max_layers + 1, 11)):
        if time.time() > deadline - 10:
            break
        
        # Generate a few material patterns
        patterns = []
        
        # Pattern: L, H, L, H, ...
        pattern1 = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        patterns.append(pattern1)
        
        # Pattern: H, L, H, L, ...
        pattern2 = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        patterns.append(pattern2)
        
        # For odd n_layers, also try symmetric patterns
        if n_layers % 2 == 1:
            # L, H, L, H, ..., L
            pattern3 = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
            if pattern3 not in patterns:
                patterns.append(pattern3)
        
        for pattern in patterns:
            if time.time() > deadline - 5:
                break
            
            # Get material indices for this pattern
            mat_indices = [materials_dict[m] for m in pattern]
            
            # Optimize
            x_opt, f_opt = _optimize_stack(n_layers, mat_indices, wavelengths, n_ambient, n_substrate,
                                            max_layer_nm, max_total_nm, start_time, deadline)
            
            if x_opt is not None and f_opt < best_score:
                best_score = f_opt
                best_materials = pattern
                best_thicknesses = list(x_opt)
    
    # If no valid solution found, return a simple quarter-wave layer
    if best_materials is None:
        centre = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [centre / (4.0 * n_L)]
    
    # Round thicknesses to reasonable precision and ensure constraints
    best_thicknesses = [max(0.0, min(max_layer_nm, t)) for t in best_thicknesses]
    total = sum(best_thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_materials, best_thicknesses
