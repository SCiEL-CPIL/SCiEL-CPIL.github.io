"""Improved program: tapered alternating stack + differential evolution."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _compute_reflectance(wavelengths, materials_list, thicknesses, n_ambient, n_substrate, n_materials):
    """Compute reflectance spectrum using characteristic matrix method (vectorized)."""
    N = len(wavelengths)
    n_layers = len(materials_list)
    
    if n_layers == 0:
        # Bare substrate
        R = ((n_ambient - n_substrate) / (n_ambient + n_substrate)) ** 2
        return np.full(N, R)
    
    k = 2.0 * np.pi / wavelengths  # (N,)
    
    M00 = np.ones(N, dtype=np.complex128)
    M01 = np.zeros(N, dtype=np.complex128)
    M10 = np.zeros(N, dtype=np.complex128)
    M11 = np.ones(N, dtype=np.complex128)
    
    for j in range(n_layers):
        n_j = n_materials[materials_list[j]]
        d_j = thicknesses[j]
        delta_j = k * n_j * d_j  # (N,)
        
        cos_d = np.cos(delta_j)
        sin_d = np.sin(delta_j)
        
        L00 = cos_d
        L01 = 1j * sin_d / n_j
        L10 = 1j * n_j * sin_d
        L11 = cos_d
        
        new_M00 = M00 * L00 + M01 * L10
        new_M01 = M00 * L01 + M01 * L11
        new_M10 = M10 * L00 + M11 * L10
        new_M11 = M10 * L01 + M11 * L11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    Y = M01 / M00
    R = np.abs((n_ambient - Y) / (n_ambient + Y)) ** 2
    
    return R

def _mean_reflectance_pct(materials_list, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    band = problem["band_nm"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_materials = problem["materials"]
    
    wavelengths = np.linspace(band[0], band[1], 401)
    
    R = _compute_reflectance(wavelengths, materials_list, thicknesses, n_ambient, n_substrate, n_materials)
    
    return float(np.mean(R) * 100.0)

def _is_valid(materials_list, thicknesses, problem):
    """Check if the design is valid."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials_list) != len(thicknesses):
        return False
    if len(materials_list) > max_layers:
        return False
    if len(materials_list) == 0:
        return False
    
    for t in thicknesses:
        if t < -1e-9 or t > max_layer_nm + 1e-9:
            return False
    
    if sum(thicknesses) > max_total_nm + 1e-9:
        return False
    
    for m in materials_list:
        if m not in ('L', 'H'):
            return False
    
    return True

def _clamp_thicknesses(thicknesses, problem):
    """Clamp thicknesses to valid range."""
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    thicks = [max(0.0, min(t, max_layer_nm)) for t in thicknesses]
    
    # If total exceeds max, scale down
    total = sum(thicks)
    if total > max_total_nm and total > 0:
        scale = max_total_nm / total
        thicks = [t * scale for t in thicks]
    
    return thicks

def _tapered_stack(base_d, gradient, n_layers, problem):
    """
    Generate a tapered alternating stack.
    
    Parameters:
    - base_d: base thickness (nm) for quarter-wave at center
    - gradient: linear gradient factor (-1 to 1), positive means increasing thickness
    - n_layers: number of layers (alternating L/H starting with L)
    
    The thickness of layer i (0-indexed) is:
    d_i = base_d * (1 + gradient * (2*i/(n_layers-1) - 1)) for n_layers > 1
    d_0 = base_d for n_layers == 1
    
    This creates a smooth transition from low to high optical density.
    """
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    
    materials = []
    thicknesses = []
    
    for i in range(n_layers):
        if i % 2 == 0:
            materials.append("L")
            n_mat = n_L
        else:
            materials.append("H")
            n_mat = n_H
        
        if n_layers > 1:
            # Normalize position from -1 to 1
            pos = 2.0 * i / (n_layers - 1) - 1.0
            d = base_d * (1.0 + gradient * pos)
        else:
            d = base_d
        
        # Quarter-wave correction: actual physical thickness
        d_phys = d / (4.0 * n_mat) * centre
        
        thicknesses.append(d_phys)
    
    return materials, thicknesses

def _differential_evolution(objective, bounds, pop_size=40, max_iter=100, F=0.8, CR=0.9, tol=1e-6, time_limit=None, start_time=None):
    """
    Differential Evolution optimizer.
    
    Parameters:
    - objective: function to minimize
    - bounds: list of (low, high) tuples
    - pop_size: population size
    - max_iter: maximum iterations
    - F: mutation factor
    - CR: crossover rate
    - tol: convergence tolerance
    - time_limit: if provided, stop after this many seconds from start_time
    - start_time: start time for time_limit
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(bounds)
    rng = np.random.default_rng(42)
    
    # Initialize population
    pop = np.zeros((pop_size, n))
    for i in range(pop_size):
        for j in range(n):
            pop[i, j] = rng.uniform(bounds[j][0], bounds[j][1])
    
    # Evaluate initial population
    fvals = np.array([objective(x) for x in pop])
    
    best_idx = np.argmin(fvals)
    best_x = pop[best_idx].copy()
    best_f = fvals[best_idx]
    
    for it in range(max_iter):
        # Check time limit
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Check convergence
        if np.std(fvals) < tol:
            break
        
        # Create new population
        new_pop = np.zeros_like(pop)
        new_fvals = np.zeros_like(fvals)
        
        for i in range(pop_size):
            # Select three distinct random indices
            r1, r2, r3 = np.random.choice(pop_size, size=3, replace=False)
            
            # Mutation
            v = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Clamp to bounds
            for j in range(n):
                v[j] = max(bounds[j][0], min(v[j], bounds[j][1]))
            
            # Crossover
            j_rand = rng.integers(n)
            u = rng.random(n)
            trial = np.where(u < CR, v, pop[i])
            trial[j_rand] = v[j_rand]  # Ensure at least one gene from mutant
            
            # Selection
            f_trial = objective(trial)
            
            if f_trial <= fvals[i]:
                new_pop[i] = trial
                new_fvals[i] = f_trial
            else:
                new_pop[i] = pop[i]
                new_fvals[i] = fvals[i]
        
        pop = new_pop
        fvals = new_fvals
        
        # Update best
        idx = np.argmin(fvals)
        if fvals[idx] < best_f:
            best_f = fvals[idx]
            best_x = pop[idx].copy()
    
    return best_x, best_f

def _nelder_mead(objective, x0, bounds, max_iter=200, tol=1e-8):
    """Simple Nelder-Mead optimizer with bounds handling."""
    n = len(x0)
    x = np.array(x0, dtype=float)
    
    # Create simplex
    simplex = [x.copy()]
    for i in range(n):
        xi = x.copy()
        step = 0.1 * (bounds[i][1] - bounds[i][0])
        xi[i] += step
        simplex.append(xi)
    
    # Evaluate
    fvals = [objective(s) for s in simplex]
    
    for it in range(max_iter):
        # Sort by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + 2.0 * (centroid - simplex[-1])
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
            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
    
    return simplex[0], fvals[0]

def design(problem):
    """Design a broadband antireflection coating using tapered stack + DE."""
    deadline = time.time() + 140.0  # Safety margin
    start_time = time.time()
    
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Best solution tracking
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    # Try different numbers of layers with tapered stacks
    # For each n_layers, optimize (base_d, gradient) using DE
    
    # Number of layers to try (odd numbers work well for symmetric designs)
    layer_counts = [3, 5, 7, 9, 11, 13, 15]
    
    # DE parameters
    de_pop_size = 30
    de_max_iter = 80
    
    for n_layers in layer_counts:
        if time.time() > deadline - 15.0:
            break
        
        # Define objective for this n_layers
        def objective_tapered(params, n_layers=n_layers):
            base_d, gradient = params
            
            # Clamp gradient to reasonable range
            gradient = max(-1.0, min(1.0, gradient))
            base_d = max(10.0, min(base_d, 200.0))  # Reasonable range
            
            materials, thicknesses = _tapered_stack(base_d, gradient, n_layers, problem)
            thicknesses = _clamp_thicknesses(thicknesses, problem)
            
            if not _is_valid(materials, thicknesses, problem):
                return 100.0  # Penalty
            
            return _mean_reflectance_pct(materials, thicknesses, problem)
        
        # DE bounds for (base_d, gradient)
        bounds = [(50.0, 150.0), (-0.8, 0.8)]
        
        # Run DE
        x_opt, f_opt = _differential_evolution(
            objective_tapered,
            bounds,
            pop_size=de_pop_size,
            max_iter=de_max_iter,
            time_limit=15.0,
            start_time=start_time
        )
        
        # Get the resulting design
        materials, thicknesses = _tapered_stack(x_opt[0], x_opt[1], n_layers, problem)
        thicknesses = _clamp_thicknesses(thicknesses, problem)
        
        if _is_valid(materials, thicknesses, problem):
            score = _mean_reflectance_pct(materials, thicknesses, problem)
            if score < best_score:
                best_score = score
                best_materials = list(materials)
                best_thicknesses = list(thicknesses)
        
        # Local refinement with Nelder-Mead on the best found so far for this n_layers
        if time.time() < deadline - 10.0:
            # Optimize all thicknesses directly
            n_params = len(thicknesses)
            if n_params <= 15:
                x0 = np.array(thicknesses)
                bounds_nm = [(0.0, max_layer_nm)] * n_params
                
                def objective_full(x, materials=materials):
                    thicks = list(x)
                    thicks = _clamp_thicknesses(thicks, problem)
                    if not _is_valid(materials, thicks, problem):
                        return 100.0
                    return _mean_reflectance_pct(materials, thicks, problem)
                
                x_refined, f_refined = _nelder_mead(objective_full, x0, bounds_nm, max_iter=150)
                
                thicks_refined = _clamp_thicknesses(list(x_refined), problem)
                if _is_valid(materials, thicks_refined, problem):
                    score_refined = _mean_reflectance_pct(materials, thicks_refined, problem)
                    if score_refined < best_score:
                        best_score = score_refined
                        best_materials = list(materials)
                        best_thicknesses = list(thicks_refined)
    
    # If no valid solution found, use a simple fallback
    if best_materials is None:
        d_L = centre / (4.0 * n_L)
        best_materials = ["L"]
        best_thicknesses = [d_L]
        best_score = _mean_reflectance_pct(best_materials, best_thicknesses, problem)
    
    # Final validation
    if not _is_valid(best_materials, best_thicknesses, problem):
        d_L = centre / (4.0 * n_L)
        best_materials = ["L"]
        best_thicknesses = [d_L]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
