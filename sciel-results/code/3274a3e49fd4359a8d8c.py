"""Antireflection coating design using Simulated Annealing with multi-temperature schedule."""
import numpy as np
import time

def _compute_reflectance(materials, thicknesses, n_ambient, n_substrate, wavelengths_nm):
    """
    Compute reflectance using the characteristic-matrix method.
    
    Parameters:
    - materials: list of 'L' or 'H' labels
    - thicknesses: list of physical thicknesses in nm
    - n_ambient: refractive index of ambient
    - n_substrate: refractive index of substrate
    - wavelengths_nm: array of wavelengths in nm
    
    Returns:
    - reflectance: array of reflectance values (0 to 1)
    """
    n_layers = len(materials)
    if n_layers == 0:
        # Bare interface
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full_like(wavelengths_nm, r * r, dtype=float)
    
    # Build the characteristic matrix
    # For normal incidence, the characteristic matrix for each layer is:
    # [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
    # where delta = 2*pi*n*d/lambda
    
    # Initialize the total matrix as identity
    total_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    
    # Process layers from air side to substrate
    for i, (mat, d) in enumerate(zip(materials, thicknesses)):
        n = problem_cache['materials'][mat]
        delta = 2.0 * np.pi * n * d / wavelengths_nm
        c = np.cos(delta)
        s = np.sin(delta)
        
        # Layer matrix
        layer_matrix = np.array([
            [c, 1j * s / n],
            [1j * n * s, c]
        ], dtype=complex)
        
        total_matrix = total_matrix @ layer_matrix
    
    # The effective admittance Y is related to the matrix elements:
    # Y = (M11 + M12 * n_substrate) / (M21 + M22 * n_substrate)
    # where M is the total matrix
    M11, M12 = total_matrix[0, 0], total_matrix[0, 1]
    M21, M22 = total_matrix[1, 0], total_matrix[1, 1]
    
    Y = (M11 + M12 * n_substrate) / (M21 + M22 * n_substrate)
    
    # Reflectance
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    
    return R

def design(problem):
    """
    Design an antireflection coating using Simulated Annealing.
    
    Returns:
    - materials: list of 'L'/'H' labels
    - thicknesses: list of physical thicknesses in nm
    """
    global problem_cache
    problem_cache = problem
    
    # Extract parameters
    n_ambient = problem['ambient_index']
    n_substrate = problem['substrate_index']
    materials_dict = problem['materials']
    band_nm = problem['band_nm']
    max_layers = problem['max_layers']
    max_total_nm = problem['max_total_nm']
    max_layer_nm = problem['max_layer_nm']
    
    # Create wavelength grid (401 points)
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin before 150s kill
    
    # Initial solution: start with a few quarter-wave layers
    def make_initial_solution():
        # Start with a simple 4-layer stack
        centre = 0.5 * (band_nm[0] + band_nm[1])
        nL = materials_dict['L']
        nH = materials_dict['H']
        
        # Quarter-wave thicknesses at centre wavelength
        dL = centre / (4.0 * nL)
        dH = centre / (4.0 * nH)
        
        # Start with L-H-L-H pattern (or similar)
        init_materials = ['L', 'H', 'L', 'H']
        init_thicknesses = [dL, dH, dL, dH]
        
        return init_materials, init_thicknesses
    
    def objective(materials, thicknesses):
        """Compute mean reflectance in percent."""
        if len(materials) == 0:
            return 100.0  # No coating, just use bare interface reflectance
        R = _compute_reflectance(materials, thicknesses, n_ambient, n_substrate, wavelengths)
        return np.mean(R) * 100.0
    
    def is_valid(materials, thicknesses):
        """Check if the design is valid."""
        if len(materials) > max_layers:
            return False
        if len(materials) != len(thicknesses):
            return False
        total = sum(thicknesses)
        if total > max_total_nm:
            return False
        for d in thicknesses:
            if d < 0 or d > max_layer_nm:
                return False
        return True
    
    # Generate a random neighbor
    def generate_neighbor(materials, thicknesses):
        """
        Generate a neighbor by one of several moves:
        1. Single-layer insertion
        2. Single-layer deletion
        3. Material swap
        4. Small continuous thickness perturbation
        """
        new_materials = list(materials)
        new_thicknesses = list(thicknesses)
        
        n = len(new_materials)
        
        # Choose a move type
        move = np.random.choice(['insert', 'delete', 'swap', 'perturb'], p=[0.25, 0.25, 0.25, 0.25])
        
        if move == 'insert' and n < max_layers:
            # Insert a new layer at a random position
            pos = np.random.randint(0, n + 1)
            mat = np.random.choice(['L', 'H'])
            # Thickness: random in [0, max_layer_nm], but try to keep total under limit
            current_total = sum(new_thicknesses)
            max_new_d = min(max_layer_nm, max_total_nm - current_total)
            if max_new_d <= 0:
                # Can't insert, try perturb instead
                move = 'perturb'
                if n == 0:
                    return materials, thicknesses
                idx = np.random.randint(0, n)
                new_thicknesses[idx] = max(0, min(max_layer_nm, new_thicknesses[idx] + np.random.normal(0, 10)))
                return new_materials, new_thicknesses
            
            d = np.random.uniform(0, max_new_d)
            new_materials.insert(pos, mat)
            new_thicknesses.insert(pos, d)
        
        elif move == 'delete' and n > 0:
            # Delete a random layer
            pos = np.random.randint(0, n)
            new_materials.pop(pos)
            new_thicknesses.pop(pos)
        
        elif move == 'swap' and n > 0:
            # Swap material of a random layer
            pos = np.random.randint(0, n)
            new_materials[pos] = 'H' if new_materials[pos] == 'L' else 'L'
        
        elif move == 'perturb' and n > 0:
            # Perturb thicknesses of one or more layers
            # Perturb 1-3 random layers
            num_perturb = np.random.randint(1, min(4, n + 1))
            indices = np.random.choice(n, size=num_perturb, replace=False)
            for idx in indices:
                # Add Gaussian perturbation
                sigma = 10.0  # nm
                new_d = new_thicknesses[idx] + np.random.normal(0, sigma)
                new_d = max(0, min(max_layer_nm, new_d))
                new_thicknesses[idx] = new_d
        
        return new_materials, new_thicknesses
    
    # Initialize
    best_materials, best_thicknesses = make_initial_solution()
    best_energy = objective(best_materials, best_thicknesses)
    
    current_materials, current_thicknesses = list(best_materials), list(best_thicknesses)
    current_energy = best_energy
    
    # Simulated Annealing parameters
    T_start = 5.0
    T_end = 0.01
    cooling_rate = 0.9995
    
    # Run SA
    T = T_start
    iteration = 0
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Generate neighbor
        new_materials, new_thicknesses = generate_neighbor(current_materials, current_thicknesses)
        
        if not is_valid(new_materials, new_thicknesses):
            continue
        
        new_energy = objective(new_materials, new_thicknesses)
        
        # Acceptance criterion
        delta = new_energy - current_energy
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_materials = new_materials
            current_thicknesses = new_thicknesses
            current_energy = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_materials = list(new_materials)
                best_thicknesses = list(new_thicknesses)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
        
        iteration += 1
        
        # If stuck for too long, restart with a new random solution
        if no_improve_count > max_no_improve:
            # Try a few random restarts
            for _ in range(5):
                rand_materials, rand_thicknesses = make_initial_solution()
                # Randomize a bit
                if len(rand_materials) > 0:
                    for i in range(len(rand_materials)):
                        rand_thicknesses[i] = max(0, min(max_layer_nm, rand_thicknesses[i] + np.random.normal(0, 20)))
                
                if is_valid(rand_materials, rand_thicknesses):
                    rand_energy = objective(rand_materials, rand_thicknesses)
                    if rand_energy < best_energy:
                        best_energy = rand_energy
                        best_materials = list(rand_materials)
                        best_thicknesses = list(rand_thicknesses)
                    if rand_energy < current_energy:
                        current_materials = list(rand_materials)
                        current_thicknesses = list(rand_thicknesses)
                        current_energy = rand_energy
            no_improve_count = 0
            T = max(T, T_start * 0.5)  # Reheat
    
    return best_materials, best_thicknesses
