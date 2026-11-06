"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_reflectance(materials, thicknesses, n_air, n_sub, wavelengths):
    """
    Compute reflectance for a stack across all wavelengths.
    materials: list of 'L' or 'H'
    thicknesses: list of physical thicknesses in nm
    n_air: ambient index
    n_sub: substrate index
    wavelengths: array of wavelengths in nm
    """
    n_layers = len(materials)
    if n_layers == 0:
        r = (n_air - n_sub) / (n_air + n_sub)
        return np.full_like(wavelengths, abs(r)**2)
    
    M_total = np.ones((len(wavelengths), 2, 2), dtype=complex)
    n_vals = [problem_global["materials"][m] for m in materials]
    
    for i in range(n_layers):
        n = n_vals[i]
        d = thicknesses[i]
        delta = 2.0 * np.pi * n * d / wavelengths
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        M_i = np.ones((len(wavelengths), 2, 2), dtype=complex)
        M_i[:, 0, 1] = 1j * sin_d / n
        M_i[:, 1, 0] = 1j * n * sin_d
        M_i[:, 0, 0] = cos_d
        M_i[:, 1, 1] = cos_d
        
        M_total = np.einsum('nij,njk->nik', M_total, M_i)
        
    m11 = M_total[:, 0, 0]
    m12 = M_total[:, 0, 1]
    m21 = M_total[:, 1, 0]
    m22 = M_total[:, 1, 1]
    
    Y_in = (m21 + m22 * n_sub) / (m11 + m12 * n_sub)
    r = (n_air - Y_in) / (n_air + Y_in)
    R = np.abs(r)**2
    return R

problem_global = None

def design(problem):
    global problem_global
    problem_global = problem
    
    n_air = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    def eval_stack(mats, thicks):
        if not mats:
            return 4.26 # Bare glass approx
        R = _compute_reflectance(mats, thicks, n_air, n_sub, wavelengths)
        return np.mean(R) * 100.0

    # Genetic Algorithm Setup
    pop_size = 50
    max_generations = 200
    start_time = time.time()
    time_budget = 55.0 # Leave some margin
    
    # Encoding: 
    # Each individual is an array of length max_layers * 2
    # [mat_idx_0, thick_0, mat_idx_1, thick_1, ...]
    # mat_idx: 0 for L, 1 for H
    # thick: 0 to max_layer_nm
    # If thick == 0, layer is ignored.
    
    def decode(ind):
        mats = []
        thicks = []
        for i in range(max_layers):
            m_idx = int(ind[2*i])
            t = ind[2*i + 1]
            if t > 1e-6:
                # Check total thickness constraint
                # We will handle total thickness penalty in fitness or repair
                mats.append('L' if m_idx == 0 else 'H')
                thicks.append(t)
        
        # Repair: if total thickness exceeds max, scale down
        if thicks:
            total = sum(thicks)
            if total > max_total_nm:
                scale = max_total_nm / total
                thicks = [t * scale for t in thicks]
        
        return mats, thicks

    def fitness(ind):
        mats, thicks = decode(ind)
        if not mats:
            return 100.0 # High penalty for empty stack
        score = eval_stack(mats, thicks)
        # Add penalty for exceeding max layers (handled by encoding, but just in case)
        # Add penalty for invalid thicknesses
        for t in thicks:
            if t < 0 or t > max_layer_nm:
                return 1000.0
        return score

    # Initialize Population
    # 1. Random individuals
    # 2. Quarter-wave stack (known good)
    
    def random_individual():
        ind = np.zeros(max_layers * 2)
        for i in range(max_layers):
            ind[2*i] = np.random.choice([0, 1])
            ind[2*i + 1] = np.random.uniform(0, max_layer_nm)
        return ind

    def quarter_wave_individual(n_layers):
        ind = np.zeros(max_layers * 2)
        lam_c = 0.5 * (band_nm[0] + band_nm[1])
        for i in range(n_layers):
            if i % 2 == 0:
                ind[2*i] = 0 # L
                ind[2*i + 1] = lam_c / (4.0 * n_L)
            else:
                ind[2*i] = 1 # H
                ind[2*i + 1] = lam_c / (4.0 * n_H)
        return ind

    population = []
    # Add some known good structures
    for nl in [1, 3, 5, 7, 9]:
        population.append(quarter_wave_individual(nl))
    
    # Fill rest with random
    while len(population) < pop_size:
        population.append(random_individual())

    best_score = float('inf')
    best_ind = None

    # Evaluate initial population
    for ind in population:
        score = fitness(ind)
        if score < best_score:
            best_score = score
            best_ind = ind.copy()

    # GA Loop
    for gen in range(max_generations):
        if time.time() - start_time > time_budget:
            break
            
        new_population = []
        
        # Elitism: keep best individual
        new_population.append(best_ind.copy())
        
        # Selection: Tournament selection
        def tournament():
            idxs = np.random.randint(0, len(population), 5)
            best_idx = idxs[0]
            best_fit = fitness(population[best_idx])
            for idx in idxs[1:]:
                f = fitness(population[idx])
                if f < best_fit:
                    best_fit = f
                    best_idx = idx
            return population[best_idx]

        # Generate new population
        while len(new_population) < pop_size:
            parent1 = tournament()
            parent2 = tournament()
            
            # Crossover: Uniform
            child = np.random.choice([parent1, parent2], size=len(parent1))
            
            # Mutation
            # Mutate material indices
            if np.random.random() < 0.3:
                mut_idx = np.random.randint(0, max_layers) * 2
                child[mut_idx] = 1 - child[mut_idx]
            
            # Mutate thicknesses
            if np.random.random() < 0.3:
                mut_idx = np.random.randint(0, max_layers) * 2 + 1
                # Gaussian mutation
                child[mut_idx] += np.random.normal(0, 20.0)
                child[mut_idx] = np.clip(child[mut_idx], 0, max_layer_nm)
            
            new_population.append(child)

        # Update population
        population = new_population
        
        # Evaluate new population and update best
        for ind in population:
            score = fitness(ind)
            if score < best_score:
                best_score = score
                best_ind = ind.copy()

    # Final decode
    final_mats, final_thicks = decode(best_ind)
    
    # Cleanup zero thickness
    clean_mats = []
    clean_thicks = []
    for m, t in zip(final_mats, final_thicks):
        if t > 1e-6:
            clean_mats.append(m)
            clean_thicks.append(float(t))
            
    if not clean_mats:
        # Fallback
        lam_c = 0.5 * (band_nm[0] + band_nm[1])
        clean_mats = ["L"]
        clean_thicks = [lam_c / (4.0 * n_L)]

    return clean_mats, clean_thicks
# EVOLVE-BLOCK-END
