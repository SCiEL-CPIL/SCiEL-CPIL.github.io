"""Broadband antireflection coating design using Simulated Annealing with vectorized reflectance."""
import numpy as np
import time
import sys

# EVOLVE-BLOCK-START

def _reflectance_spectrum(materials, thicknesses, lambdas, n0, ns):
    """
    Compute reflectance spectrum using characteristic matrix method, vectorized over wavelengths.
    materials: list of refractive indices (one per layer)
    thicknesses: list of physical thicknesses in nm
    lambdas: array of wavelengths in nm
    n0: ambient index, ns: substrate index
    Returns: array of reflectance values (0 to 1)
    """
    N = len(thicknesses)
    M = len(lambdas)
    
    if N == 0:
        # Bare substrate
        r = (n0 - ns) / (n0 + ns)
        return np.full(M, r * r)
    
    # Pre-compute for all wavelengths
    # d_j = 2*pi*n_j*t_j / lambda
    # For each layer j: d_j[m] = 2*pi*materials[j]*thicknesses[j]/lambdas[m]
    
    # Build the total characteristic matrix by multiplying layer matrices
    # Each layer matrix is [[cos(d), i*sin(d)/n], [i*n*sin(d), cos(d)]]
    
    # Start with identity matrix for all wavelengths
    # M_total[m] = product over j of M_j[m]
    # Then Y = (B/A) where M_total = [[A,B],[C,D]]
    # R = |(n0*Y - ns)/(n0*Y + ns)|^2
    
    # Use complex numbers
    # M_total shape: (M, 2, 2)
    M_total = np.ones((M, 2, 2), dtype=complex)
    
    for j in range(N):
        n_j = materials[j]
        t_j = thicknesses[j]
        # d_j for all wavelengths
        d_j = 2.0 * np.pi * n_j * t_j / lambdas  # shape (M,)
        
        cos_d = np.cos(d_j)  # shape (M,)
        sin_d = np.sin(d_j)  # shape (M,)
        
        # Layer matrix elements
        # M_j = [[cos_d, 1j*sin_d/n_j], [1j*n_j*sin_d, cos_d]]
        
        # Multiply M_total = M_total @ M_j
        # M_total[:, 0, 0] = A, M_total[:, 0, 1] = B, etc.
        A = M_total[:, 0, 0]
        B = M_total[:, 0, 1]
        C = M_total[:, 1, 0]
        D = M_total[:, 1, 1]
        
        # New A = A*cos_d + B*(1j*sin_d/n_j)
        # New B = A*(1j*n_j*sin_d) + B*cos_d
        # New C = C*cos_d + D*(1j*sin_d/n_j)
        # New D = C*(1j*n_j*sin_d) + D*cos_d
        
        new_A = A * cos_d + B * (1j * sin_d / n_j)
        new_B = A * (1j * n_j * sin_d) + B * cos_d
        new_C = C * cos_d + D * (1j * sin_d / n_j)
        new_D = C * (1j * n_j * sin_d) + D * cos_d
        
        M_total[:, 0, 0] = new_A
        M_total[:, 0, 1] = new_B
        M_total[:, 1, 0] = new_C
        M_total[:, 1, 1] = new_D
    
    # Y = B / A
    A = M_total[:, 0, 0]
    B = M_total[:, 0, 1]
    
    # Avoid division by zero
    mask = np.abs(A) > 1e-15
    Y = np.zeros(M, dtype=complex)
    Y[mask] = B[mask] / A[mask]
    
    # R = |(n0*Y - ns)/(n0*Y + ns)|^2
    numerator = n0 * Y - ns
    denominator = n0 * Y + ns
    mask2 = np.abs(denominator) > 1e-15
    R = np.zeros(M, dtype=float)
    R[mask2] = np.abs(numerator[mask2] / denominator[mask2])**2
    
    return R

def _objective(materials, thicknesses, problem):
    """Compute mean reflectance in percent."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    lambdas = np.linspace(band[0], band[1], 401)
    
    R = _reflectance_spectrum(materials, thicknesses, lambdas, n0, ns)
    return np.mean(R) * 100.0

def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    if any(t < -1e-9 or t > max_layer_nm + 1e-9 for t in thicknesses):
        return False
    if sum(thicknesses) > max_total_nm + 1e-9:
        return False
    if any(m not in ['L', 'H'] for m in materials):
        return False
    return True

def design(problem):
    """Design a broadband antireflection coating using Simulated Annealing."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    materials_dict = problem["materials"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    n_L = materials_dict["L"]
    n_H = materials_dict["H"]
    
    # Initial design: quarter-wave stack
    centre = 0.5 * (band[0] + band[1])
    
    # Start with a reasonable initial design
    # Try a few-layer quarter-wave stack
    initial_n_layers = 4
    init_materials = []
    init_thicknesses = []
    for i in range(initial_n_layers):
        if i % 2 == 0:
            init_materials.append("L")
            init_thicknesses.append(centre / (4.0 * n_L))
        else:
            init_materials.append("H")
            init_thicknesses.append(centre / (4.0 * n_H))
    
    # Ensure validity
    if not _is_valid(init_materials, init_thicknesses, problem):
        init_materials = ["L"]
        init_thicknesses = [centre / (4.0 * n_L)]
    
    best_materials = init_materials[:]
    best_thicknesses = init_thicknesses[:]
    best_score = _objective(best_materials, best_thicknesses, problem)
    
    # Current state
    cur_materials = init_materials[:]
    cur_thicknesses = init_thicknesses[:]
    cur_score = best_score
    
    # Simulated Annealing parameters
    T = 5.0  # Initial temperature (in percent reflectance units)
    T_min = 0.01
    cooling_rate = 0.9995
    max_iterations = 50000
    
    np.random.seed(42)
    
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Generate a perturbation
        pert_type = np.random.random()
        
        new_materials = cur_materials[:]
        new_thicknesses = cur_thicknesses[:]
        
        if pert_type < 0.3:
            # Change a thickness
            if len(new_thicknesses) > 0:
                idx = np.random.randint(len(new_thicknesses))
                new_thicknesses[idx] = max(0.0, min(max_layer_nm, new_thicknesses[idx] + np.random.normal(0, 20.0)))
        elif pert_type < 0.5:
            # Change a material
            if len(new_materials) > 0:
                idx = np.random.randint(len(new_materials))
                new_materials[idx] = "H" if new_materials[idx] == "L" else "L"
        elif pert_type < 0.6:
            # Add a layer
            if len(new_materials) < max_layers:
                pos = np.random.randint(len(new_materials) + 1)
                mat = "L" if np.random.random() < 0.5 else "H"
                th = centre / (4.0 * (n_L if mat == "L" else n_H))
                new_materials.insert(pos, mat)
                new_thicknesses.insert(pos, th)
        elif pert_type < 0.7:
            # Remove a layer
            if len(new_materials) > 1:
                idx = np.random.randint(len(new_materials))
                new_materials.pop(idx)
                new_thicknesses.pop(idx)
        elif pert_type < 0.8:
            # Swap two layers
            if len(new_materials) >= 2:
                i, j = np.random.choice(len(new_materials), 2, replace=False)
                new_materials[i], new_materials[j] = new_materials[j], new_materials[i]
                new_thicknesses[i], new_thicknesses[j] = new_thicknesses[j], new_thicknesses[i]
        else:
            # Random restart with small stack
            n = np.random.randint(1, min(6, max_layers + 1))
            new_materials = []
            new_thicknesses = []
            for k in range(n):
                mat = "L" if np.random.random() < 0.5 else "H"
                th = centre / (4.0 * (n_L if mat == "L" else n_H))
                new_materials.append(mat)
                new_thicknesses.append(th)
        
        # Check validity
        if not _is_valid(new_materials, new_thicknesses, problem):
            continue
        
        # Compute new score
        new_score = _objective(new_materials, new_thicknesses, problem)
        
        # Acceptance criterion
        delta = new_score - cur_score
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, T_min)):
            cur_materials = new_materials
            cur_thicknesses = new_thicknesses
            cur_score = new_score
            
            if cur_score < best_score:
                best_score = cur_score
                best_materials = cur_materials[:]
                best_thicknesses = cur_thicknesses[:]
        
        # Cool down
        T = max(T * cooling_rate, T_min)
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
