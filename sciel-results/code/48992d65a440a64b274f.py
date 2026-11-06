"""Broadband antireflection coating design via Genetic Algorithm."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def _fast_compute_mean_reflectance(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm):
    """
    Compute mean reflectance in percent over 401 equally spaced wavelengths in band_nm.
    Uses characteristic matrix method.
    """
    if not labels:
        # Bare substrate
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return np.abs(r) ** 2 * 100.0
        
    lam_min, lam_max = band_nm
    wavelengths = np.linspace(lam_min, lam_max, 401)
    
    # Prepare layer parameters
    n_mat = np.array([n_L if l == 'L' else n_H for l in labels], dtype=float)
    d = np.array(thicknesses, dtype=float)
    beta = 2.0 * np.pi * n_mat * d  # 2*pi*n*d for each layer
    
    total_r = 0.0
    n_wl = len(wavelengths)
    
    # Process in chunks to manage memory, though 401 is small
    # Vectorize over wavelengths for speed
    # M is 2x2 matrix. We process wavelengths in a loop but vectorize layers? 
    # Actually, vectorizing over wavelengths is better.
    
    # Initialize M components for all wavelengths
    # M = [[M00, M01], [M10, M11]]
    M00 = np.ones(n_wl, dtype=complex)
    M01 = np.zeros(n_wl, dtype=complex)
    M10 = np.zeros(n_wl, dtype=complex)
    M11 = np.ones(n_wl, dtype=complex)
    
    for i in range(len(labels)):
        delta = beta[i] / wavelengths
        c = np.cos(delta)
        s = np.sin(delta)
        
        # Layer matrix components
        a = c
        b = 1j * s / n_mat[i]
        cc = 1j * n_mat[i] * s
        dd = c
        
        # Matrix multiplication M_new = M_old * M_layer
        # M00_new = M00*a + M01*cc
        # M01_new = M00*b + M01*dd
        # M10_new = M10*a + M11*cc
        # M11_new = M10*b + M11*dd
        
        new_M00 = M00 * a + M01 * cc
        new_M01 = M00 * b + M01 * dd
        new_M10 = M10 * a + M11 * cc
        new_M11 = M10 * b + M11 * dd
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
        
    # Admittance Y = M10 / M00
    Y = M10 / M00
    # Reflection coefficient
    r = (n_amb - Y) / (n_amb + Y)
    R = np.abs(r) ** 2 * 100.0
    
    return np.mean(R)

def design(problem):
    """Design a broadband antireflection coating stack using a Genetic Algorithm."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Discretize thicknesses
    # Step size 5 nm gives 80 steps for 400 nm. 
    # This allows for fine-grained search while keeping evaluation fast.
    thickness_step = 5.0
    num_thick_steps = int(max_layer_nm / thickness_step) + 1 # 0 to 80
    
    # Helper to decode a GA individual
    # Individual: [layer_type_1, ..., layer_type_N, thick_idx_1, ..., thick_idx_N]
    # layer_type: 0 for L, 1 for H
    # thick_idx: 0 to num_thick_steps-1
    # To handle variable length, we can use a fixed max length and ignore trailing "empty" layers.
    # Let's use a fixed population of length max_layers.
    # A layer is "active" if its thickness index > 0 OR if we enforce a minimum thickness?
    # Better: Use a specific encoding.
    # Let's stick to a fixed number of layers in the GA, say up to max_layers.
    # We will penalize solutions that exceed max_total_nm.
    
    # Encoding:
    # x[0:max_layers] are layer types (0=L, 1=H)
    # x[max_layers:2*max_layers] are thickness indices (0..num_thick_steps-1)
    # If thickness index is 0, the layer is effectively zero thickness (can be ignored).
    # However, zero thickness layers can cause numerical issues or be redundant.
    # We will filter out zero thickness layers during decoding.
    
    pop_size = 100
    max_generations = 2000 # Limited by time
    
    # Initialize population
    # Mix of random and some structured initial guesses
    population = np.zeros((pop_size, 2 * max_layers), dtype=int)
    
    # Seed with some known good structures
    # 1-layer L
    pop_idx = 0
    population[pop_idx, 0] = 0 # L
    population[pop_idx, max_layers] = int((600 / (4 * n_L)) / thickness_step) # ~217nm
    pop_idx += 1
    
    # 2-layer LH
    population[pop_idx, 0] = 0
    population[pop_idx, 1] = 1
    population[pop_idx, max_layers] = int((600 / (4 * n_L)) / thickness_step)
    population[pop_idx, max_layers + 1] = int((600 / (4 * n_H)) / thickness_step)
    pop_idx += 1
    
    # 3-layer LHL
    population[pop_idx, 0] = 0
    population[pop_idx, 1] = 1
    population[pop_idx, 2] = 0
    population[pop_idx, max_layers] = int((600 / (4 * n_L)) / thickness_step)
    population[pop_idx, max_layers + 1] = int((600 / (4 * n_H)) / thickness_step)
    population[pop_idx, max_layers + 2] = int((600 / (4 * n_L)) / thickness_step)
    pop_idx += 1
    
    # Fill rest with random
    np.random.seed(42)
    for i in range(pop_idx, pop_size):
        # Random layer types
        population[i, :max_layers] = np.random.randint(0, 2, max_layers)
        # Random thicknesses, biased towards lower values to satisfy total thickness constraint
        # Uniform 0 to max steps
        population[i, max_layers:] = np.random.randint(0, num_thick_steps, max_layers)
    
    def decode_ind(ind):
        """Decode a GA individual into labels and thicknesses."""
        labels = []
        thicknesses = []
        for i in range(max_layers):
            lt = ind[i]
            ti = ind[max_layers + i]
            if ti == 0:
                # Zero thickness, skip
                continue
            # Ensure it's within bounds
            thick = ti * thickness_step
            if thick > max_layer_nm:
                thick = max_layer_nm
            labels.append('L' if lt == 0 else 'H')
            thicknesses.append(thick)
        return labels, thicknesses

    def evaluate_population(pop):
        """Evaluate all individuals in population."""
        scores = np.full(len(pop), np.inf)
        for i, ind in enumerate(pop):
            if time.time() > deadline - 5.0:
                break
            labels, thicknesses = decode_ind(ind)
            
            # Constraint check
            total_thick = sum(thicknesses)
            if total_thick > max_total_nm:
                # Penalty
                scores[i] = _fast_compute_mean_reflectance(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm) + 10.0 * (total_thick - max_total_nm) / max_total_nm
                continue
                
            # Validity check
            if len(labels) > max_layers:
                 continue
                
            scores[i] = _fast_compute_mean_reflectance(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm)
        return scores

    # Initial evaluation
    scores = evaluate_population(population)
    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_ind = population[best_idx].copy()
    
    # GA Loop
    for gen in range(max_generations):
        if time.time() > deadline - 10.0:
            break
            
        # Selection: Tournament selection
        # Crossover and Mutation
        new_pop = np.zeros_like(population)
        
        # Elitism: Keep top 10%
        elite_size = max(1, pop_size // 10)
        elite_indices = np.argsort(scores)[:elite_size]
        new_pop[:elite_size] = population[elite_indices]
        
        # Fill rest
        for i in range(elite_size, pop_size):
            # Parents
            p1_idx = np.random.choice(pop_size, size=1, p=None)[0] # Uniform for simplicity, or tournament
            p2_idx = np.random.choice(pop_size, size=1, p=None)[0]
            
            # Better selection: Tournament
            def tournament_select(pop, scores, k=3):
                candidates = np.random.choice(len(pop), size=k, replace=False)
                best_cand = candidates[np.argmin(scores[candidates])]
                return best_cand
                
            p1_idx = tournament_select(population, scores, k=3)
            p2_idx = tournament_select(population, scores, k=3)
            
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            # Crossover: Uniform crossover
            mask = np.random.rand(len(p1)) < 0.5
            child = np.where(mask, p1, p2)
            
            # Mutation
            # 1. Flip layer type (5% chance per gene)
            for j in range(max_layers):
                if np.random.rand() < 0.05:
                    child[j] = 1 - child[j]
            # 2. Change thickness index (10% chance per gene)
            for j in range(max_layers):
                if np.random.rand() < 0.10:
                    # Random new thickness
                    child[max_layers + j] = np.random.randint(0, num_thick_steps)
                elif np.random.rand() < 0.10:
                    # Step change
                    delta = np.random.choice([-1, 1])
                    child[max_layers + j] = np.clip(child[max_layers + j] + delta, 0, num_thick_steps - 1)
            
            new_pop[i] = child
        
        population = new_pop
        scores = evaluate_population(population)
        
        # Update best
        gen_best_idx = np.argmin(scores)
        if scores[gen_best_idx] < best_score:
            best_score = scores[gen_best_idx]
            best_ind = population[gen_best_idx].copy()
            
        # Convergence check (optional)
        if np.std(scores[:elite_size]) < 1e-6:
            # Shuffle to restart exploration?
            # Just continue, mutation should handle it.
            pass

    # Final Local Refinement on Best Individual
    # Use a simple gradient-free local search (Nelder-Mead style but on discrete grid)
    # Or just try neighboring points in the discrete space.
    
    best_labels, best_thicknesses = decode_ind(best_ind)
    
    # If the best solution is empty or invalid, fallback
    if not best_labels or sum(best_thicknesses) > max_total_nm + 1e-9:
         # Fallback to a known good solution
         best_labels = ["L"]
         best_thicknesses = [600 / (4 * n_L)]
    
    # Ensure final constraints are met strictly
    best_thicknesses = [max(0.0, min(t, max_layer_nm)) for t in best_thicknesses]
    if sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
        
    # Final check: if score is worse than a simple 1-layer coating, use that
    simple_labels = ["L"]
    simple_thick = [600 / (4 * n_L)]
    simple_score = _fast_compute_mean_reflectance(simple_labels, simple_thick, n_amb, n_sub, n_L, n_H, band_nm)
    
    final_score = _fast_compute_mean_reflectance(best_labels, best_thicknesses, n_amb, n_sub, n_L, n_H, band_nm)
    
    if final_score > simple_score:
        best_labels = simple_labels
        best_thicknesses = simple_thick
        
    return best_labels, best_thicknesses
# EVOLVE-BLOCK-END
