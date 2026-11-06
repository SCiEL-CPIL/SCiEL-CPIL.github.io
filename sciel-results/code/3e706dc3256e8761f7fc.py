"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def _compute_reflectance(labels, thicknesses, materials, ambient_index, substrate_index, band_nm):
    """Compute mean reflectance in percent over the band using characteristic matrix method."""
    if len(labels) == 0:
        r0 = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return r0 * r0 * 100.0
    
    n_L = materials['L']
    n_H = materials['H']
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    n_layers = len(labels)
    
    # Build index array
    n_indices = np.empty(n_layers + 2, dtype=float)
    n_indices[0] = ambient_index
    n_indices[-1] = substrate_index
    for i, lab in enumerate(labels):
        n_indices[i + 1] = n_L if lab == 'L' else n_H
    
    d = np.array(thicknesses, dtype=float)
    
    # Vectorized computation over all wavelengths
    delta = 2.0 * np.pi * np.outer(n_indices[1:-1] * d, 1.0 / wavelengths)
    cos_d = np.cos(delta)
    sin_d = np.sin(delta)
    
    m11 = np.ones(401, dtype=complex)
    m12 = np.zeros(401, dtype=complex)
    m21 = np.zeros(401, dtype=complex)
    m22 = np.ones(401, dtype=complex)
    
    for j in range(n_layers):
        n_j = n_indices[j + 1]
        lm11 = cos_d[j]
        lm12 = 1j * sin_d[j] / n_j
        lm21 = 1j * n_j * sin_d[j]
        lm22 = cos_d[j]
        
        new_m11 = m11 * lm11 + m12 * lm21
        new_m12 = m11 * lm12 + m12 * lm22
        new_m21 = m21 * lm11 + m22 * lm21
        new_m22 = m21 * lm12 + m22 * lm22
        
        m11 = new_m11
        m12 = new_m12
        m21 = new_m21
        m22 = new_m22
    
    Y = (m21 + m22 * substrate_index) / (m11 + m12 * substrate_index)
    r = (ambient_index - Y) / (ambient_index + Y)
    reflectances = np.abs(r) ** 2
    
    return np.mean(reflectances) * 100.0


def _nelder_mead(objective, x0, step=50.0, max_iter=200, tol=1e-6):
    """Simple Nelder-Mead optimizer for continuous variables."""
    n = len(x0)
    # Initialize simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        xi = np.array(x0, dtype=float)
        xi[i] += step
        simplex.append(xi)
    
    # Evaluate
    fvals = [objective(x) for x in simplex]
    
    for _ in range(max_iter):
        # Sort by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < fvals[0]:
            # Expansion
            gamma = 2.0
            xe = centroid + gamma * (centroid - simplex[-1])
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
            rho = 0.5
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
    
    # Return best
    order = np.argsort(fvals)
    return simplex[order[0]], fvals[order[0]]


def design(problem):
    """Design an antireflection coating stack using a hybrid discrete/continuous optimization."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    materials = problem["materials"]
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    n_L = materials['L']
    n_H = materials['H']
    
    def evaluate_design(labels, thicknesses):
        """Evaluate a design, returning infinity if invalid."""
        if len(labels) > max_layers:
            return float('inf')
        if len(labels) != len(thicknesses):
            return float('inf')
        if any(t < 0 or t > max_layer_nm for t in thicknesses):
            return float('inf')
        if sum(thicknesses) > max_total_nm:
            return float('inf')
        return _compute_reflectance(labels, thicknesses, materials, ambient_index, substrate_index, band_nm)
    
    # Start with a good known design: 5-layer quarter-wave stack
    # Center wavelength ~600nm
    t_L_qw = 600.0 / (4.0 * n_L)  # ~108.7 nm
    t_H_qw = 600.0 / (4.0 * n_H)  # ~63.8 nm
    
    best_labels = ['L', 'H', 'L', 'H', 'L']
    best_thicknesses = [t_L_qw, t_H_qw, t_L_qw, t_H_qw, t_L_qw]
    best_score = evaluate_design(best_labels, best_thicknesses)
    
    # Try different numbers of layers with quarter-wave designs
    for n_layers in range(1, min(11, max_layers + 1)):
        if time.time() > deadline:
            break
        # Alternate L and H starting with L
        labels = []
        thicknesses = []
        for i in range(n_layers):
            if i % 2 == 0:
                labels.append('L')
                thicknesses.append(t_L_qw)
            else:
                labels.append('H')
                thicknesses.append(t_H_qw)
        
        if sum(thicknesses) > max_total_nm:
            continue
        
        score = evaluate_design(labels, thicknesses)
        if score < best_score:
            best_score = score
            best_labels = labels[:]
            best_thicknesses = thicknesses[:]
    
    # Now use Nelder-Mead to optimize continuous thicknesses for the best structure
    # We'll optimize the thicknesses for a few candidate structures
    
    candidate_structures = [
        ['L', 'H', 'L', 'H', 'L'],
        ['L', 'H', 'L'],
        ['H', 'L', 'H'],
        ['L', 'H', 'L', 'H'],
        ['L', 'H', 'L', 'H', 'L', 'H'],
        ['L', 'H', 'L', 'H', 'L', 'H', 'L'],
    ]
    
    for labels in candidate_structures:
        if time.time() > deadline:
            break
        if len(labels) > max_layers:
            continue
        
        # Initial guess: quarter-wave at 600nm
        x0 = []
        for lab in labels:
            if lab == 'L':
                x0.append(t_L_qw)
            else:
                x0.append(t_H_qw)
        
        # Check if initial guess is valid
        if sum(x0) > max_total_nm:
            continue
        
        def objective(x):
            """Objective function for Nelder-Mead: mean reflectance with penalty for invalid."""
            # Check constraints
            if any(t < 0 or t > max_layer_nm for t in x):
                return 100.0 + sum(max(0, t - max_layer_nm) for t in x)
            if sum(x) > max_total_nm:
                return 100.0 + (sum(x) - max_total_nm)
            return _compute_reflectance(labels, list(x), materials, ambient_index, substrate_index, band_nm)
        
        # Run Nelder-Mead
        x_opt, f_opt = _nelder_mead(objective, x0, step=20.0, max_iter=100, tol=1e-4)
        
        # Validate the result
        if all(0 <= t <= max_layer_nm for t in x_opt) and sum(x_opt) <= max_total_nm:
            if f_opt < best_score:
                best_score = f_opt
                best_labels = labels[:]
                best_thicknesses = list(x_opt)
    
    # Try a few more random structures and optimize them
    for _ in range(5):
        if time.time() > deadline:
            break
        
        n_layers = random.randint(3, min(7, max_layers))
        labels = [random.choice(['L', 'H']) for _ in range(n_layers)]
        
        # Ensure not all same
        if len(set(labels)) == 1:
            continue
        
        x0 = []
        for lab in labels:
            if lab == 'L':
                x0.append(random.uniform(50, 150))
            else:
                x0.append(random.uniform(40, 100))
        
        # Scale to fit within total thickness
        total = sum(x0)
        if total > max_total_nm:
            scale = max_total_nm / total
            x0 = [t * scale for t in x0]
        
        if any(t > max_layer_nm for t in x0):
            continue
        
        def objective(x):
            if any(t < 0 or t > max_layer_nm for t in x):
                return 100.0 + sum(max(0, t - max_layer_nm) for t in x)
            if sum(x) > max_total_nm:
                return 100.0 + (sum(x) - max_total_nm)
            return _compute_reflectance(labels, list(x), materials, ambient_index, substrate_index, band_nm)
        
        x_opt, f_opt = _nelder_mead(objective, x0, step=10.0, max_iter=80, tol=1e-4)
        
        if all(0 <= t <= max_layer_nm for t in x_opt) and sum(x_opt) <= max_total_nm:
            if f_opt < best_score:
                best_score = f_opt
                best_labels = labels[:]
                best_thicknesses = list(x_opt)
    
    # Final validation
    if not best_labels or len(best_labels) > max_layers:
        best_labels = ["L"]
        best_thicknesses = [600 / (4.0 * n_L)]
    
    if any(t < 0 or t > max_layer_nm for t in best_thicknesses):
        best_labels = ["L"]
        best_thicknesses = [600 / (4.0 * n_L)]
    
    if sum(best_thicknesses) > max_total_nm:
        best_labels = ["L"]
        best_thicknesses = [600 / (4.0 * n_L)]
    
    # Round thicknesses to reasonable precision
    best_thicknesses = [round(t, 4) for t in best_thicknesses]
    
    return best_labels, best_thicknesses

# EVOLVE-BLOCK-END
