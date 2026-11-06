# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _compute_reflectance(materials, thicknesses, problem):
    """
    Compute mean reflectance in percent over the band using vectorized characteristic matrix method.
    """
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    n_dict = problem["materials"]
    lam0, lam1 = problem["band_nm"]
    n_waves = 401
    lambdas = np.linspace(lam0, lam1, n_waves)
    
    n_layers = len(materials)
    if n_layers == 0:
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r

    total_thick = sum(thicknesses)
    if total_thick > problem["max_total_nm"] or any(t < 0 or t > problem["max_layer_nm"] for t in thicknesses):
        return float('inf')
        
    # Initialize total matrix M as identity for all wavelengths
    # M = [[A, B], [C, D]]
    A = np.ones(n_waves, dtype=np.complex128)
    B = np.zeros(n_waves, dtype=np.complex128)
    C = np.zeros(n_waves, dtype=np.complex128)
    D = np.ones(n_waves, dtype=np.complex128)
    
    for i in range(n_layers):
        n_i = n_dict[materials[i]]
        d_i = thicknesses[i]
        
        # Vectorized delta calculation
        delta = 2.0 * np.pi * n_i * d_i / lambdas
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # Layer matrix elements (vectorized)
        M11 = cos_d
        M12 = 1j * sin_d / n_i
        M21 = 1j * n_i * sin_d
        M22 = cos_d
        
        # Matrix multiplication M_new = M_old * M_layer
        # A_new = A*M11 + B*M21
        # B_new = A*M12 + B*M22
        # C_new = C*M11 + D*M21
        # D_new = C*M12 + D*M22
        
        nA = A * M11 + B * M21
        nB = A * M12 + B * M22
        nC = C * M11 + D * M21
        nD = C * M12 + D * M22
        
        A, B, C, D = nA, nB, nC, nD
        
    Y_in = (C + D * ns) / (A + B * ns)
    r = (n0 - Y_in) / (n0 + Y_in)
    R = np.abs(r) ** 2
    
    return 100.0 * np.mean(R)

def _initialize_quarter_wave(problem, n_layers):
    """
    Initialize a quarter-wave stack centered at 600nm.
    """
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    lambda_center = 600.0
    
    # Quarter wave thickness: n * d = lambda / 4 => d = lambda / (4 * n)
    dL = lambda_center / (4.0 * nL)
    dH = lambda_center / (4.0 * nH)
    
    # Standard AR stack: L H L H ... or H L H L ...
    # Usually starts with L (lower index) to match air (1.0) better? 
    # Air (1.0) -> L (1.38) -> H (2.35) -> Glass (1.52)
    # Actually, ideal single layer is sqrt(1.0*1.52) = 1.23.
    # L is 1.38, H is 2.35.
    # A common broadband design is L-H-L-H...
    
    materials = []
    thicknesses = []
    
    for i in range(n_layers):
        if i % 2 == 0:
            materials.append('L')
            # Add some randomness to thickness
            thicknesses.append(dL * random.uniform(0.8, 1.2))
        else:
            materials.append('H')
            thicknesses.append(dH * random.uniform(0.8, 1.2))
            
    # Ensure constraints
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    for i in range(len(thicknesses)):
        thicknesses[i] = max(0, min(thicknesses[i], max_layer_nm))
        
    total_t = sum(thicknesses)
    if total_t > max_total_nm:
        factor = max_total_nm / total_t
        thicknesses = [t * factor for t in thicknesses]
        
    return materials, thicknesses

def design(problem):
    """
    Design a stack using Simulated Annealing.
    """
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Start with a reasonable number of layers, e.g., 5 or 7
    # 5 layers is a good balance between complexity and constraint
    init_layers = 5
    current_mats, current_thicks = _initialize_quarter_wave(problem, init_layers)
    
    current_score = _compute_reflectance(current_mats, current_thicks, problem)
    best_mats = list(current_mats)
    best_thicks = list(current_thicks)
    best_score = current_score
    
    # SA Parameters
    T = 10.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.995
    max_iters = 30000  # Limit iterations to stay within time budget
    start_time = time.time()
    time_limit = 50.0  # Leave 10s buffer
    
    iter_count = 0
    while T > T_min and iter_count < max_iters:
        if time.time() - start_time > time_limit:
            break
            
        # Generate a neighbor
        new_mats = list(current_mats)
        new_thicks = list(current_thicks)
        
        move_type = random.random()
        
        if move_type < 0.7:
            # Perturb thickness of a random layer
            if new_thicks:
                idx = random.randint(0, len(new_thicks) - 1)
                new_thicks[idx] += random.gauss(0, 20)
                new_thicks[idx] = max(0, min(new_thicks[idx], max_layer_nm))
        elif move_type < 0.8:
            # Swap material of a random layer
            if new_mats:
                idx = random.randint(0, len(new_mats) - 1)
                new_mats[idx] = 'L' if new_mats[idx] == 'H' else 'H'
        elif move_type < 0.9:
            # Add a layer
            if len(new_mats) < max_layers:
                pos = random.randint(0, len(new_mats))
                new_mat = random.choice(['L', 'H'])
                new_thick = random.uniform(20, 100)
                new_mats.insert(pos, new_mat)
                new_thicks.insert(pos, new_thick)
        else:
            # Remove a layer
            if len(new_mats) > 1:
                pos = random.randint(0, len(new_mats) - 1)
                new_mats.pop(pos)
                new_thicks.pop(pos)
                
        # Constraint Repair: Total thickness
        total_t = sum(new_thicks)
        if total_t > max_total_nm:
            factor = max_total_nm / total_t
            new_thicks = [t * factor for t in new_thicks]
            
        new_score = _compute_reflectance(new_mats, new_thicks, problem)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_mats = new_mats
            current_thicks = new_thicks
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_mats = list(current_mats)
                best_thicks = list(current_thicks)
                
        T *= cooling_rate
        iter_count += 1
        
    return best_mats, best_thicks
# EVOLVE-BLOCK-END
