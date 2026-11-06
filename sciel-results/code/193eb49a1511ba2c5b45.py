# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.optimize import minimize

def _compute_reflectance(materials, thicknesses, problem):
    """
    Compute mean reflectance in percent over the band.
    Uses characteristic matrix method.
    """
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    n_dict = problem["materials"]
    lam0, lam1 = problem["band_nm"]
    n_waves = 401
    lambdas = np.linspace(lam0, lam1, n_waves)
    
    # Pre-compute material indices and thicknesses arrays for faster access
    n_layers = len(materials)
    if n_layers == 0:
        # Bare glass
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r

    # Validate constraints inside to be safe, though GA should maintain them
    total_thick = sum(thicknesses)
    if total_thick > problem["max_total_nm"] or any(t < 0 or t > problem["max_layer_nm"] for t in thicknesses):
        return float('inf')
        
    n_vals = np.array([n_dict[m] for m in materials])
    d_vals = np.array(thicknesses)
    
    total_r = 0.0
    for lam in lambdas:
        # Initialize total matrix M
        A = 1.0 + 0j
        B = 0.0 + 0j
        C = 0.0 + 0j
        D = 1.0 + 0j
        
        for i in range(n_layers):
            n_i = n_vals[i]
            d_i = d_vals[i]
            delta = 2.0 * np.pi * n_i * d_i / lam
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            
            # Layer matrix
            M11 = cos_d
            M12 = 1j * sin_d / n_i
            M21 = 1j * n_i * sin_d
            M22 = cos_d
            
            # M_new = M_old * M_layer
            nA = A * M11 + B * M21
            nB = A * M12 + B * M22
            nC = C * M11 + D * M21
            nD = C * M12 + D * M22
            
            A, B, C, D = nA, nB, nC, nD
        
        Y_in = (C + D * ns) / (A + B * ns)
        r = (n0 - Y_in) / (n0 + Y_in)
        R = np.abs(r) ** 2
        total_r += R
        
    return 100.0 * total_r / n_waves

def _compute_reflectance_fast(materials, thicknesses, problem, lambdas, n_vals, d_vals):
    """
    Faster version with precomputed arrays for optimization.
    """
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    n_waves = len(lambdas)
    
    if len(materials) == 0:
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r

    total_r = 0.0
    for lam in lambdas:
        A = 1.0 + 0j
        B = 0.0 + 0j
        C = 0.0 + 0j
        D = 1.0 + 0j
        
        for i in range(len(materials)):
            n_i = n_vals[i]
            d_i = d_vals[i]
            delta = 2.0 * np.pi * n_i * d_i / lam
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            
            M11 = cos_d
            M12 = 1j * sin_d / n_i
            M21 = 1j * n_i * sin_d
            M22 = cos_d
            
            nA = A * M11 + B * M21
            nB = A * M12 + B * M22
            nC = C * M11 + D * M21
            nD = C * M12 + D * M22
            
            A, B, C, D = nA, nB, nC, nD
        
        Y_in = (C + D * ns) / (A + B * ns)
        r = (n0 - Y_in) / (n0 + Y_in)
        R = np.abs(r) ** 2
        total_r += R
        
    return 100.0 * total_r / n_waves

def _generate_starting_points(problem):
    """
    Generate physically-informed starting points for optimization.
    """
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    starts = []
    
    # 1. Quarter-wave stack (classic antireflection)
    # For broadband, use multiple quarter-wave pairs
    for n_pairs in range(1, 5):
        # Quarter-wave thickness at center wavelength (600 nm)
        lam_center = 600.0
        dL = lam_center / (4 * nL)
        dH = lam_center / (4 * nH)
        
        # Start with H-L pairs (H next to air, L next to substrate for nH > nL)
        mats = []
        thicks = []
        for _ in range(n_pairs):
            mats.append('H')
            thicks.append(dH)
            mats.append('L')
            thicks.append(dL)
        
        # Ensure constraints
        total = sum(thicks)
        if total > max_total_nm:
            factor = max_total_nm / total
            thicks = [t * factor for t in thicks]
        
        starts.append((mats, thicks))
    
    # 2. Random structures with reasonable thicknesses
    random.seed(42)
    for _ in range(20):
        n_layers = random.randint(2, min(10, max_layers))
        mats = [random.choice(['L', 'H']) for _ in range(n_layers)]
        
        # Random thicknesses in [30, 150] nm
        thicks = [random.uniform(30, 150) for _ in range(n_layers)]
        
        total = sum(thicks)
        if total > max_total_nm:
            factor = max_total_nm / total
            thicks = [t * factor for t in thicks]
        
        starts.append((mats, thicks))
    
    # 3. Specific known good structures
    # 2-layer: H-L
    starts.append((['H', 'L'], [120.0, 80.0]))
    # 4-layer: H-L-H-L
    starts.append((['H', 'L', 'H', 'L'], [100.0, 70.0, 90.0, 60.0]))
    # 6-layer
    starts.append((['H', 'L', 'H', 'L', 'H', 'L'], [80.0, 60.0, 75.0, 55.0, 70.0, 50.0]))
    
    return starts

def design(problem):
    """
    Design a stack using multi-start Nelder-Mead local optimization.
    """
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    lam0, lam1 = problem["band_nm"]
    n_waves = 401
    lambdas = np.linspace(lam0, lam1, n_waves)
    
    # Precompute for fast evaluation
    def make_objective(mats, n_vals):
        def objective(d_vals):
            # Clip to valid range
            d_vals = np.clip(d_vals, 0, max_layer_nm)
            # Check total thickness constraint
            if np.sum(d_vals) > max_total_nm:
                return 1e6
            return _compute_reflectance_fast(mats, d_vals, problem, lambdas, n_vals, d_vals)
        return objective
    
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    starts = _generate_starting_points(problem)
    
    # Limit number of starts to fit within time budget
    # Each Nelder-Mead run takes ~1-5 seconds
    max_starts = 15
    
    for mats, thicks in starts[:max_starts]:
        # Skip if already worse than best
        n_vals = np.array([problem["materials"][m] for m in mats])
        d_vals = np.array(thicks, dtype=float)
        
        # Check constraints
        if sum(d_vals) > max_total_nm or any(t < 0 or t > max_layer_nm for t in d_vals):
            continue
        
        # Quick evaluation
        current_score = _compute_reflectance(mats, d_vals, problem)
        if current_score < best_score:
            best_score = current_score
            best_materials = list(mats)
            best_thicknesses = list(d_vals)
        
        # Run Nelder-Mead optimization
        objective = make_objective(mats, n_vals)
        
        # Bounds for Nelder-Mead (it doesn't support bounds directly, but we clip)
        bounds = [(0, max_layer_nm)] * len(mats)
        
        # Initial simplex
        x0 = d_vals.copy()
        # Ensure x0 is valid
        x0 = np.clip(x0, 0, max_layer_nm)
        if np.sum(x0) > max_total_nm:
            factor = max_total_nm / np.sum(x0)
            x0 = x0 * factor
        
        # Run optimization with limited iterations
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'maxfev': 500,
                    'xatol': 0.5,
                    'fatol': 0.01,
                    'adaptive': True
                }
            )
            
            if result.success or result.fun < best_score:
                opt_d = np.clip(result.x, 0, max_layer_nm)
                if np.sum(opt_d) > max_total_nm:
                    factor = max_total_nm / np.sum(opt_d)
                    opt_d = opt_d * factor
                
                opt_score = _compute_reflectance(mats, opt_d, problem)
                if opt_score < best_score:
                    best_score = opt_score
                    best_materials = list(mats)
                    best_thicknesses = list(opt_d)
        except Exception:
            pass
    
    # If no valid solution found, return a simple 2-layer stack
    if best_materials is None:
        best_materials = ['H', 'L']
        best_thicknesses = [120.0, 80.0]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
