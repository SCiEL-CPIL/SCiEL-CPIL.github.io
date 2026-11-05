"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def _char_matrix(theta, n, d, lam):
    """Characteristic matrix for a single layer."""
    beta = 2.0 * np.pi * n * d / lam
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    return np.array([[cos_b, 1.0j * sin_b / theta],
                     [1.0j * theta * sin_b, cos_b]], dtype=complex)

def _reflectance_stack(materials, thicknesses, lambdas, n_amb, n_sub):
    """
    Compute reflectance for a stack of layers at given wavelengths.
    materials: list of 'L'/'H'
    thicknesses: list of thicknesses in nm
    lambdas: array of wavelengths in nm
    n_amb: ambient index
    n_sub: substrate index
    """
    n_map = {'L': 1.38, 'H': 2.35}
    ns = [n_map[m] for m in materials]
    ds = thicknesses
    n_layers = len(ns)
    
    # Build total characteristic matrix
    # M_total = M1 * M2 * ... * Mn
    # At each wavelength, compute the product
    
    # For efficiency, we can vectorize over wavelengths
    # But let's start with a simple loop and optimize if needed
    
    # Actually, let's vectorize over wavelengths using numpy broadcasting
    # This might be memory intensive for many wavelengths, but 401 is fine
    
    # Let's do it wavelength by wavelength to keep memory low
    reflectances = np.zeros(len(lambdas))
    
    for i, lam in enumerate(lambdas):
        # Start with identity
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for j in range(n_layers):
            theta = ns[j]  # normal incidence, so theta = n_j
            d = ds[j]
            Mj = _char_matrix(theta, ns[j], d, lam)
            M = M @ Mj
        
        # Boundary condition
        # [B; C] = M * [1; Y_sub] where Y_sub = n_sub (at normal incidence)
        Y_sub = n_sub
        B = M[0, 0] + M[0, 1] * Y_sub
        C = M[1, 0] + M[1, 1] * Y_sub
        
        Y_in = B / C
        # Reflectance
        r = (n_amb - Y_in) / (n_amb + Y_in)
        R = np.abs(r) ** 2
        reflectances[i] = R
    
    return reflectances

def _objective(x, problem, lambdas):
    """
    Objective function: mean reflectance in percent.
    x: vector of thicknesses (length = num_layers)
    """
    n_layers = len(x)
    materials = []
    # We need to determine materials based on some encoding
    # For now, let's assume the materials pattern is fixed and we only optimize thicknesses
    # But we also need to optimize the number of layers and material sequence
    
    # This is a hybrid problem. Let's use a different approach:
    # We'll have a fixed number of layers, and the material sequence is determined by x encoding
    
    # Actually, let's restructure: we'll optimize both thicknesses and material choices
    # For a given number of layers N, we have 2N variables: N thicknesses and N material choices (0 or 1)
    # But material choices are discrete, so we need a different approach
    
    # Let's use a simpler approach: fix the number of layers and material pattern, 
    # and only optimize thicknesses. Then try different patterns.
    
    # For this objective, x is just the thicknesses for a fixed pattern
    # The pattern is stored in a closure or passed separately
    
    # Let's restructure the whole thing
    pass

def design(problem):
    """
    Design an antireflection coating.
    Returns (materials, thicknesses) where materials is a list of 'L'/'H' 
    and thicknesses is a list of physical thicknesses in nm.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    # Wavelengths for evaluation
    lambdas = np.linspace(band[0], band[1], 401)
    
    best_score = np.inf
    best_materials = None
    best_thicknesses = None
    
    def _evaluate(materials, thicknesses):
        """Evaluate a design and return mean reflectance in percent."""
        if len(materials) == 0:
            # Bare glass
            r = (n_amb - n_sub) / (n_amb + n_sub)
            return 100.0 * r ** 2
        R = _reflectance_stack(materials, thicknesses, lambdas, n_amb, n_sub)
        return 100.0 * np.mean(R)
    
    def _is_valid(materials, thicknesses):
        """Check if the design is valid."""
        if len(materials) != len(thicknesses):
            return False
        if len(materials) > max_layers:
            return False
        if any(t < 0 or t > max_layer for t in thicknesses):
            return False
        if sum(thicknesses) > max_total:
            return False
        return True
    
    # Strategy: Try various numbers of layers and material patterns
    # For each, optimize the thicknesses using scipy.optimize
    
    # First, try some known good starting points
    # Quarter-wave stack
    centre = 0.5 * (band[0] + band[1])
    
    # Try 1 layer
    for mat in ['L', 'H']:
        n = n_L if mat == 'L' else n_H
        t = centre / (4.0 * n)
        if t <= max_layer:
            materials = [mat]
            thicknesses = [t]
            score = _evaluate(materials, thicknesses)
            if _is_valid(materials, thicknesses) and score < best_score:
                best_score = score
                best_materials = materials
                best_thicknesses = thicknesses
    
    # Try 2 layers: L-H or H-L
    for pattern in [['L', 'H'], ['H', 'L']]:
        ns = [n_L if m == 'L' else n_H for m in pattern]
        # Initial guess: quarter-wave for each
        t0 = [centre / (4.0 * n) for n in ns]
        
        def obj(x):
            if time.time() > deadline - 5:
                return 1e10
            if any(t < 0 or t > max_layer for t in x):
                return 1e10
            if sum(x) > max_total:
                return 1e10
            return _evaluate(pattern, x)
        
        try:
            res = minimize(obj, t0, method='L-BFGS-B', 
                          bounds=[(0, max_layer)] * len(pattern),
                          options={'maxiter': 100, 'ftol': 1e-8})
            if res.success and _is_valid(pattern, res.x):
                score = _evaluate(pattern, res.x)
                if score < best_score:
                    best_score = score
                    best_materials = list(pattern)
                    best_thicknesses = list(res.x)
        except Exception:
            pass
    
    # Try 3 layers: various patterns
    patterns_3 = [
        ['L', 'H', 'L'],
        ['H', 'L', 'H'],
        ['L', 'L', 'H'],
        ['H', 'H', 'L'],
        ['L', 'H', 'H'],
        ['H', 'L', 'L'],
    ]
    
    for pattern in patterns_3:
        if time.time() > deadline - 10:
            break
        ns = [n_L if m == 'L' else n_H for m in pattern]
        t0 = [centre / (4.0 * n) for n in ns]
        
        def obj(x):
            if time.time() > deadline - 5:
                return 1e10
            if any(t < 0 or t > max_layer for t in x):
                return 1e10
            if sum(x) > max_total:
                return 1e10
            return _evaluate(pattern, x)
        
        try:
            res = minimize(obj, t0, method='L-BFGS-B',
                          bounds=[(0, max_layer)] * len(pattern),
                          options={'maxiter': 200, 'ftol': 1e-8})
            if res.success and _is_valid(pattern, res.x):
                score = _evaluate(pattern, res.x)
                if score < best_score:
                    best_score = score
                    best_materials = list(pattern)
                    best_thicknesses = list(res.x)
        except Exception:
            pass
    
    # Try 4 layers
    patterns_4 = [
        ['L', 'H', 'L', 'H'],
        ['H', 'L', 'H', 'L'],
        ['L', 'L', 'H', 'H'],
        ['H', 'H', 'L', 'L'],
    ]
    
    for pattern in patterns_4:
        if time.time() > deadline - 10:
            break
        ns = [n_L if m == 'L' else n_H for m in pattern]
        t0 = [centre / (4.0 * n) for n in ns]
        
        def obj(x):
            if time.time() > deadline - 5:
                return 1e10
            if any(t < 0 or t > max_layer for t in x):
                return 1e10
            if sum(x) > max_total:
                return 1e10
            return _evaluate(pattern, x)
        
        try:
            res = minimize(obj, t0, method='L-BFGS-B',
                          bounds=[(0, max_layer)] * len(pattern),
                          options={'maxiter': 300, 'ftol': 1e-8})
            if res.success and _is_valid(pattern, res.x):
                score = _evaluate(pattern, res.x)
                if score < best_score:
                    best_score = score
                    best_materials = list(pattern)
                    best_thicknesses = list(res.x)
        except Exception:
            pass
    
    # Try 5 layers
    patterns_5 = [
        ['L', 'H', 'L', 'H', 'L'],
        ['H', 'L', 'H', 'L', 'H'],
        ['L', 'H', 'H', 'L', 'H'],
        ['H', 'L', 'L', 'H', 'L'],
    ]
    
    for pattern in patterns_5:
        if time.time() > deadline - 10:
            break
        ns = [n_L if m == 'L' else n_H for m in pattern]
        t0 = [centre / (4.0 * n) for n in ns]
        
        def obj(x):
            if time.time() > deadline - 5:
                return 1e10
            if any(t < 0 or t > max_layer for t in x):
                return 1e10
            if sum(x) > max_total:
                return 1e10
            return _evaluate(pattern, x)
        
        try:
            res = minimize(obj, t0, method='L-BFGS-B',
                          bounds=[(0, max_layer)] * len(pattern),
                          options={'maxiter': 400, 'ftol': 1e-8})
            if res.success and _is_valid(pattern, res.x):
                score = _evaluate(pattern, res.x)
                if score < best_score:
                    best_score = score
                    best_materials = list(pattern)
                    best_thicknesses = list(res.x)
        except Exception:
            pass
    
    # Try 6 layers
    patterns_6 = [
        ['L', 'H', 'L', 'H', 'L', 'H'],
        ['H', 'L', 'H', 'L', 'H', 'L'],
    ]
    
    for pattern in patterns_6:
        if time.time() > deadline - 10:
            break
        ns = [n_L if m == 'L' else n_H for m in pattern]
        t0 = [centre / (4.0 * n) for n in ns]
        
        def obj(x):
            if time.time() > deadline - 5:
                return 1e10
            if any(t < 0 or t > max_layer for t in x):
                return 1e10
            if sum(x) > max_total:
                return 1e10
            return _evaluate(pattern, x)
        
        try:
            res = minimize(obj, t0, method='L-BFGS-B',
                          bounds=[(0, max_layer)] * len(pattern),
                          options={'maxiter': 500, 'ftol': 1e-8})
            if res.success and _is_valid(pattern, res.x):
                score = _evaluate(pattern, res.x)
                if score < best_score:
                    best_score = score
                    best_materials = list(pattern)
                    best_thicknesses = list(res.x)
        except Exception:
            pass
    
    # If we haven't found a good solution, try a random search
    if time.time() < deadline - 20:
        np.random.seed(42)
        for _ in range(1000):
            if time.time() > deadline - 10:
                break
            
            # Random number of layers
            n_layers = np.random.randint(1, min(max_layers, 10) + 1)
            
            # Random material pattern
            materials = list(np.random.choice(['L', 'H'], size=n_layers))
            
            # Random thicknesses
            thicknesses = list(np.random.uniform(0, max_layer, size=n_layers))
            
            # Check total thickness
            if sum(thicknesses) > max_total:
                # Scale down
                scale = max_total / sum(thicknesses)
                thicknesses = [t * scale for t in thicknesses]
            
            if _is_valid(materials, thicknesses):
                score = _evaluate(materials, thicknesses)
                if score < best_score:
                    best_score = score
                    best_materials = materials
                    best_thicknesses = thicknesses
    
    # Final polish: try to improve the best solution found
    if best_materials is not None and time.time() < deadline - 5:
        pattern = list(best_materials)
        t0 = list(best_thicknesses)
        
        def obj(x):
            if time.time() > deadline - 2:
                return 1e10
            if any(t < 0 or t > max_layer for t in x):
                return 1e10
            if sum(x) > max_total:
                return 1e10
            return _evaluate(pattern, x)
        
        try:
            res = minimize(obj, t0, method='L-BFGS-B',
                          bounds=[(0, max_layer)] * len(pattern),
                          options={'maxiter': 500, 'ftol': 1e-10})
            if res.success and _is_valid(pattern, res.x):
                score = _evaluate(pattern, res.x)
                if score < best_score:
                    best_score = score
                    best_materials = list(pattern)
                    best_thicknesses = list(res.x)
        except Exception:
            pass
    
    # Return the best result
    if best_materials is None:
        # Fallback: single quarter-wave L layer
        centre = 0.5 * (band[0] + band[1])
        t = centre / (4.0 * n_L)
        return ["L"], [t]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
