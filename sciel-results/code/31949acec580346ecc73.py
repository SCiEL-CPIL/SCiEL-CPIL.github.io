"""Broadband AR coating design using Differential Evolution with heuristic initialization."""
import numpy as np
import time
import sys

# EVOLVE-BLOCK-START

def _characteristic_matrix(n0, ns, materials, thicknesses, lambdas):
    """
    Compute reflectance using the characteristic matrix method.
    n0: ambient index, ns: substrate index
    materials: dict of layer indices (same order as thicknesses)
    thicknesses: list of physical thicknesses in nm
    lambdas: array of wavelengths in nm
    Returns: array of reflectance values (0 to 1)
    """
    N = len(thicknesses)
    # Pre-compute for all wavelengths at once
    # M_total starts as identity
    M11 = np.ones((N + 1, len(lambdas)), dtype=complex)
    M12 = np.zeros((N + 1, len(lambdas)), dtype=complex)
    M21 = np.zeros((N + 1, len(lambdas)), dtype=complex)
    M22 = np.ones((N + 1, len(lambdas)), dtype=complex)
    
    # We build the matrix from substrate to air? 
    # Standard convention: M_total = M1 * M2 * ... * MN
    # Where M1 is the first layer (air side).
    # Let's iterate from the last layer (substrate side) to the first?
    # Actually, the product M = M1 M2 ... MN.
    # Let's just iterate i from 0 to N-1.
    
    for i in range(N):
        n_i = materials[thicknesses[i]] # Wait, thicknesses are floats, we need indices.
        # The function signature in the prompt says: design(problem) -> (materials, thicknesses)
        # My internal helper should take the list of material indices and thicknesses.
        pass

    # Let's rewrite the helper to take explicit material indices and thicknesses.
    pass

def _reflectance_calc(material_indices, thicknesses, n0, ns, materials, lambdas):
    N = len(thicknesses)
    if N == 0:
        # Bare substrate
        r = (n0 - ns) / (n0 + ns)
        return np.full_like(lambdas, np.abs(r)**2)
        
    # M_total = M1 * M2 * ... * MN
    # Start with identity
    M11 = np.ones(len(lambdas), dtype=complex)
    M12 = np.zeros(len(lambdas), dtype=complex)
    M21 = np.zeros(len(lambdas), dtype=complex)
    M22 = np.ones(len(lambdas), dtype=complex)
    
    # Iterate from last layer to first? No, M = M1 * M2 ...
    # Let's compute M_i for each layer and multiply.
    # To be efficient, we can precompute cos and sin.
    
    for i in range(N):
        n_i = materials[material_indices[i]]
        d_i = thicknesses[i]
        
        delta = 2.0 * np.pi * n_i * d_i / lambdas
        c = np.cos(delta)
        s = np.sin(delta)
        
        # M_i = [[c, 1j*s/n_i], [1j*n_i*s, c]]
        m11 = c
        m12 = 1j * s / n_i
        m21 = 1j * n_i * s
        m22 = c
        
        # New M = M * M_i
        # M = [[M11, M12], [M21, M22]]
        # M_i = [[m11, m12], [m21, m22]]
        
        new_M11 = M11 * m11 + M12 * m21
        new_M12 = M11 * m12 + M12 * m22
        new_M21 = M21 * m11 + M22 * m21
        new_M22 = M21 * m12 + M22 * m22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
        
    # Input admittance Y_in = (M21 + M22 * ns) / (M11 + M12 * ns)
    # Wait, standard formula:
    # Y_in = (B + C * Y_s) / (A + D * Y_s)
    # Where M = [[A, B], [C, D]]
    # Y_s = ns (substrate index)
    
    Y_s = ns
    Y_in = (M21 + M22 * Y_s) / (M11 + M12 * Y_s)
    
    # Reflectance R = |(Y_in - n0) / (Y_in + n0)|^2
    r = (Y_in - n0) / (Y_in + n0)
    R = np.abs(r)**2
    
    return R

def design(problem):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    band = problem["band_nm"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    materials = problem["materials"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths: 401 equally spaced points from 400 to 800
    lambdas = np.linspace(band[0], band[1], 401)
    
    # Heuristic initialization: Quarter-wave stacks
    # We will try a few different depths and pick the best one to start DE, 
    # or just use DE to optimize the thicknesses for a fixed depth.
    # Given the non-convexity, we'll run DE for a specific depth, 
    # but we can also try to optimize the number of layers? 
    # DE is better fixed dimension. Let's pick a good depth.
    # 5-10 layers is usually good for broadband.
    
    # Let's try to be smart: run a quick local search on a few heuristic starts
    # to find a good depth, then run DE on that depth?
    # Or just run DE on a medium depth (e.g., 8 layers) with a large population.
    
    # Strategy: 
    # 1. Generate a set of candidate stacks (quarter-wave L/H alternating).
    # 2. Evaluate them to find the best initial depth and configuration.
    # 3. Run Differential Evolution on the thicknesses of that depth to refine.
    
    best_score = np.inf
    best_materials = []
    best_thicknesses = []
    
    # Candidate depths
    depths_to_try = [5, 8, 10, 12]
    
    # Pre-calculate quarter-wave thicknesses for center wavelength
    center_lambda = 0.5 * (band[0] + band[1])
    qw_L = center_lambda / (4.0 * materials['L'])
    qw_H = center_lambda / (4.0 * materials['H'])
    
    # 1. Quick heuristic scan
    for depth in depths_to_try:
        # Alternating L, H, L, H... starting with L (low index matches air better? 
        # Actually, starting with L reduces initial reflection from air (1.0) to L (1.38) 
        # compared to H (2.35). So L first is good.)
        mats = []
        thicks = []
        for i in range(depth):
            if i % 2 == 0:
                mats.append('L')
                thicks.append(qw_L)
            else:
                mats.append('H')
                thicks.append(qw_H)
                
        # Check constraints
        if sum(thicks) > max_total_nm:
            continue
            
        # Evaluate
        R = _reflectance_calc(mats, thicks, n0, ns, materials, lambdas)
        mean_R = np.mean(R) * 100.0
        
        if mean_R < best_score:
            best_score = mean_R
            best_materials = mats
            best_thicknesses = thicks
            
    if time.time() > deadline:
        return best_materials, best_thicknesses

    # 2. Differential Evolution on the best depth found
    # If no valid heuristic was found (unlikely), use depth 5
    if not best_materials:
        depth = 5
        best_materials = ['L' if i%2==0 else 'H' for i in range(depth)]
        best_thicknesses = [qw_L if i%2==0 else qw_H for i in range(depth)]
        
    depth = len(best_materials)
    
    # DE Parameters
    pop_size = 50
    # Variables: thicknesses (depth)
    # Bounds: [0, max_layer_nm]
    # We fix the material sequence to the best heuristic one to keep dimensionality low
    # and because the sequence is part of the combinatorial optimization which is harder.
    # However, we can also randomize the sequence in the population?
    # DE works on continuous variables. Let's stick to optimizing thicknesses for the best sequence.
    # To hedge against sequence issues, we could run DE on a few different sequences.
    
    sequences_to_optimize = []
    # 1. The best heuristic sequence
    sequences_to_optimize.append(best_materials)
    
    # 2. A few random valid sequences of the same depth
    for _ in range(3):
        rand_mats = []
        rand_thicks = []
        for i in range(depth):
            m = 'L' if np.random.rand() < 0.5 else 'H'
            rand_mats.append(m)
            rand_thicks.append(np.random.uniform(0, max_layer_nm))
        sequences_to_optimize.append(rand_mats)
        
    # Run DE for each sequence
    for seq in sequences_to_optimize:
        if time.time() > deadline:
            break
            
        # DE Implementation
        # x: array of thicknesses
        # Objective: mean reflectance
        
        # Initial population
        pop = np.random.uniform(0, max_layer_nm, (pop_size, depth))
        # Ensure at least one member is the heuristic start
        pop[0] = best_thicknesses if seq == best_materials else np.random.uniform(0, max_layer_nm, depth)
        
        # Evaluate initial population
        fitness = np.zeros(pop_size)
        for i in range(pop_size):
            R = _reflectance_calc(seq, pop[i], n0, ns, materials, lambdas)
            fitness[i] = np.mean(R)
            
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        # DE Loop
        max_iter = 50
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate
        
        for iter in range(max_iter):
            if time.time() > deadline:
                break
                
            for i in range(pop_size):
                # Select 3 distinct random individuals
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                j_rand = np.random.randint(0, depth)
                trial = pop[i].copy()
                for j in range(depth):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                    
                # Clip to bounds
                trial = np.clip(trial, 0, max_layer_nm)
                
                # Evaluate trial
                R = _reflectance_calc(seq, trial, n0, ns, materials, lambdas)
                trial_f = np.mean(R)
                
                # Selection
                if trial_f < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_f
                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial.copy()
                        
        # Check constraints for the best found in this sequence
        if sum(best_x) <= max_total_nm:
            if best_f < best_score:
                best_score = best_f
                best_materials = seq
                best_thicknesses = best_x.tolist()
                
    # Final check and return
    if sum(best_thicknesses) > max_total_nm:
        # Scale down if necessary
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
        
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
