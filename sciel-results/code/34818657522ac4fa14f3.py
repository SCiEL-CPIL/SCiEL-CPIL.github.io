"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass."""
import time
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize

# EVOLVE-BLOCK-START

def _reflectance_spectrum(materials, thicknesses, n_amb, n_sub, band_nm, n_points=401):
    """
    Compute mean normal-incidence reflectance (%) over the band using the
    characteristic-matrix method.
    
    materials: list of 'L'/'H' labels (air to substrate)
    thicknesses: list of physical thicknesses in nm
    n_amb: ambient refractive index
    n_sub: substrate refractive index
    band_nm: [lambda_min, lambda_max]
    n_points: number of equally spaced wavelengths
    """
    n_layers = len(materials)
    if n_layers == 0:
        # Bare interface
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return 100.0 * r * r
    
    # Refractive indices of each layer
    n_idx = [problem_materials[m] for m in materials]
    
    # Optical thicknesses (n * d)
    # d is in nm, lambda in nm, so phase delta = 2*pi*n*d/lambda
    
    lambdas = np.linspace(band_nm[0], band_nm[1], n_points)
    
    total_R = 0.0
    for lam in lambdas:
        # Characteristic matrix for each layer
        # M_i = [[cos(delta_i), j*sin(delta_i)/n_i], [j*n_i*sin(delta_i), cos(delta_i)]]
        # where delta_i = 2*pi*n_i*d_i/lambda
        
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for i in range(n_layers):
            n_i = n_idx[i]
            d_i = thicknesses[i]
            delta_i = 2.0 * math.pi * n_i * d_i / lam
            c = math.cos(delta_i)
            s = math.sin(delta_i)
            
            M_i = np.array([[c, 1j * s / n_i],
                            [1j * n_i * s, c]], dtype=complex)
            M = M @ M_i
        
        # Input admittance
        Y = M[1, 1] / M[0, 1]
        
        # Reflectance
        r = (n_amb - Y) / (n_amb + Y)
        R = np.abs(r) ** 2
        total_R += R * 100.0
    
    return total_R / n_points


def _make_objective(materials, n_amb, n_sub, band_nm):
    """Create an objective function that takes thicknesses and returns mean reflectance."""
    def objective(thicknesses):
        try:
            # Clip to valid range
            t = np.clip(thicknesses, 0, 400.0)
            return _reflectance_spectrum(materials, list(t), n_amb, n_sub, band_nm)
        except Exception:
            return 1e6
    return objective


def design(problem):
    """
    Design a stack of at most twenty low- and high-index layers (1.38 and 2.35) on glass
    that minimizes the mean reflectance between 400 and 800 nm at normal incidence.
    """
    global problem_materials
    
    t_start = time.time()
    deadline = t_start + 140.0  # 10s safety margin before 150s kill
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    problem_materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    n_L = problem_materials["L"]
    n_H = problem_materials["H"]
    
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    # Try different numbers of layers and material patterns
    # For each number of layers, try several structural templates
    
    def time_left():
        return deadline - time.time()
    
    # Templates to try: (number of layers, pattern generator)
    # Pattern generators return a list of 'L'/'H' labels
    
    def make_alternating(n):
        """Alternating L, H, L, H, ... starting with L"""
        return ['L' if i % 2 == 0 else 'H' for i in range(n)]
    
    def make_alternating_H(n):
        """Alternating H, L, H, L, ... starting with H"""
        return ['H' if i % 2 == 0 else 'L' for i in range(n)]
    
    def make_symmetric_L(n):
        """Symmetric pattern starting and ending with L"""
        half = n // 2
        left = ['L' if i % 2 == 0 else 'H' for i in range(half)]
        if n % 2 == 0:
            return left + list(reversed(left))
        else:
            return left + ['L'] + list(reversed(left))
    
    def make_symmetric_H(n):
        """Symmetric pattern starting and ending with H"""
        half = n // 2
        left = ['H' if i % 2 == 0 else 'L' for i in range(half)]
        if n % 2 == 0:
            return left + list(reversed(left))
        else:
            return left + ['H'] + list(reversed(left))
    
    def make_graded(n):
        """Graded index: start with L, end with H (or vice versa)"""
        # Simple graded: more L at air side, more H at substrate side
        if n <= 1:
            return ['L']
        # Create a pattern that transitions from L to H
        pattern = []
        for i in range(n):
            # Fraction of the way through
            frac = i / (n - 1)
            if frac < 0.5:
                pattern.append('L')
            else:
                pattern.append('H')
        return pattern
    
    def make_graded_rev(n):
        """Graded index: start with H, end with L"""
        if n <= 1:
            return ['H']
        pattern = []
        for i in range(n):
            frac = i / (n - 1)
            if frac < 0.5:
                pattern.append('H')
            else:
                pattern.append('L')
        return pattern
    
    # List of (n_layers, pattern) combinations to try
    candidates = []
    
    # Small stacks: 1 to 6 layers
    for n in range(1, 7):
        candidates.append((n, make_alternating(n)))
        candidates.append((n, make_alternating_H(n)))
        if n >= 2:
            candidates.append((n, make_symmetric_L(n)))
            candidates.append((n, make_symmetric_H(n)))
            candidates.append((n, make_graded(n)))
            candidates.append((n, make_graded_rev(n)))
    
    # Medium stacks: 7 to 12 layers
    for n in range(7, 13):
        candidates.append((n, make_alternating(n)))
        candidates.append((n, make_alternating_H(n)))
        candidates.append((n, make_symmetric_L(n)))
        candidates.append((n, make_symmetric_H(n)))
    
    # Larger stacks: 13 to 20 layers
    for n in range(13, max_layers + 1):
        candidates.append((n, make_alternating(n)))
        candidates.append((n, make_alternating_H(n)))
    
    # Remove duplicates
    seen = set()
    unique_candidates = []
    for n, pattern in candidates:
        key = tuple(pattern)
        if key not in seen:
            seen.add(key)
            unique_candidates.append((n, pattern))
    
    # Sort by number of layers (smaller first, as they're often easier to optimize)
    unique_candidates.sort(key=lambda x: x[0])
    
    # For each candidate, run optimization
    for n, materials in unique_candidates:
        if time_left() < 5.0:
            break
        
        n_layers = len(materials)
        
        # Bounds for differential evolution: [0, max_layer_nm] for each layer
        bounds = [(0, max_layer_nm)] * n_layers
        
        # Objective function
        obj = _make_objective(materials, n_amb, n_sub, band_nm)
        
        # Initial guess: quarter-wave at center wavelength
        center_wl = 0.5 * (band_nm[0] + band_nm[1])
        init_guess = []
        for m in materials:
            n_m = problem_materials[m]
            init_guess.append(center_wl / (4.0 * n_m))
        
        # Run differential evolution with time limit
        try:
            de_result = differential_evolution(
                obj,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-4,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=True,
                init='latinhypercube'
            )
            
            if de_result.success or de_result.fun < best_score:
                # Polish with Nelder-Mead
                t_nm = time.time()
                nm_result = minimize(
                    obj,
                    de_result.x,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 0.1, 'fatol': 1e-4}
                )
                
                if nm_result.fun < de_result.fun:
                    best_thick = nm_result.x
                    best_val = nm_result.fun
                else:
                    best_thick = de_result.x
                    best_val = de_result.fun
                
                # Check total thickness constraint
                total_thick = sum(best_thick)
                if total_thick <= max_total_nm:
                    if best_val < best_score:
                        best_score = best_val
                        best_materials = list(materials)
                        best_thicknesses = list(best_thick)
                        
        except Exception:
            pass
        
        # Also try the initial quarter-wave guess
        init_val = obj(init_guess)
        total_init = sum(init_guess)
        if total_init <= max_total_nm and init_val < best_score:
            best_score = init_val
            best_materials = list(materials)
            best_thicknesses = list(init_guess)
    
    # If no valid solution found, return a simple quarter-wave layer
    if best_materials is None:
        center_wl = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [center_wl / (4.0 * n_L)]
    
    # Ensure thicknesses are within bounds
    best_thicknesses = [max(0.0, min(max_layer_nm, t)) for t in best_thicknesses]
    
    # Check total thickness constraint one more time
    while sum(best_thicknesses) > max_total_nm and len(best_thicknesses) > 1:
        # Scale down all thicknesses
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
