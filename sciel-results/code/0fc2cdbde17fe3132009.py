"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

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
    lam_c = 0.5 * (band_nm[0] + band_nm[1])
    
    def eval_stack(mats, thicks):
        R = _compute_reflectance(mats, thicks, n_air, n_sub, wavelengths)
        return np.mean(R) * 100.0

    def valid_stack(mats, thicks):
        if len(mats) == 0:
            return False
        if len(mats) > max_layers:
            return False
        if any(t < 0 or t > max_layer_nm for t in thicks):
            return False
        if sum(thicks) > max_total_nm + 1e-6:
            return False
        return True

    def random_individual():
        n_layers = random.randint(3, max_layers)
        mats = ['L' if random.random() < 0.5 else 'H' for _ in range(n_layers)]
        # Initial thicknesses around quarter wave
        thicks = []
        for m in mats:
            n = n_L if m == 'L' else n_H
            t = lam_c / (4.0 * n) * random.uniform(0.5, 1.5)
            thicks.append(t)
        
        # Enforce total thickness constraint
        total = sum(thicks)
        if total > max_total_nm:
            scale = max_total_nm / total
            thicks = [t * scale for t in thicks]
        
        return mats, thicks

    def mutate(mats, thicks):
        mats = mats[:]
        thicks = thicks[:]
        mutation_type = random.random()
        
        if mutation_type < 0.4:
            # Perturb a random thickness
            if thicks:
                idx = random.randint(0, len(thicks)-1)
                thicks[idx] = thicks[idx] * random.uniform(0.8, 1.2)
                thicks[idx] = np.clip(thicks[idx], 0, max_layer_nm)
                
                total = sum(thicks)
                if total > max_total_nm:
                    scale = max_total_nm / total
                    thicks = [t * scale for t in thicks]
        
        elif mutation_type < 0.6:
            # Swap material of a random layer
            if mats:
                idx = random.randint(0, len(mats)-1)
                mats[idx] = 'H' if mats[idx] == 'L' else 'L'
        
        elif mutation_type < 0.75:
            # Add a layer
            if len(mats) < max_layers:
                pos = random.randint(0, len(mats))
                new_mat = 'L' if random.random() < 0.5 else 'H'
                new_thick = lam_c / (4.0 * (n_L if new_mat == 'L' else n_H)) * random.uniform(0.5, 1.0)
                mats.insert(pos, new_mat)
                thicks.insert(pos, new_thick)
                
                total = sum(thicks)
                if total > max_total_nm:
                    scale = max_total_nm / total
                    thicks = [t * scale for t in thicks]
        
        elif mutation_type < 0.85:
            # Remove a layer
            if len(mats) > 1:
                idx = random.randint(0, len(mats)-1)
                mats.pop(idx)
                thicks.pop(idx)
        
        else:
            # Shift thickness from one layer to another
            if len(thicks) >= 2:
                i, j = random.sample(range(len(thicks)), 2)
                delta = thicks[i] * random.uniform(0.1, 0.3)
                thicks[i] -= delta
                thicks[j] += delta
                thicks[i] = max(0, thicks[i])
                thicks[j] = min(max_layer_nm, thicks[j])
                
                total = sum(thicks)
                if total > max_total_nm:
                    scale = max_total_nm / total
                    thicks = [t * scale for t in thicks]
        
        return mats, thicks

    def crossover(mats1, thicks1, mats2, thicks2):
        # Uniform crossover
        mats = []
        thicks = []
        max_len = max(len(mats1), len(mats2))
        
        # Pad shorter with None to handle length differences simply, or just truncate
        # Let's just take up to the min length or max length?
        # Better: Pad with 'L' and 0, then filter out zeros later?
        # Actually, let's just do a simple splice.
        
        # Random cut point
        cut = random.randint(0, min(len(mats1), len(mats2)))
        
        # This simple crossover is tricky with different lengths.
        # Let's stick to a simpler "blend" or just random selection from parents for GA?
        # No, let's do:
        # Child takes first part from P1, second part from P2.
        # If lengths differ, we just take what we can.
        
        child_mats = mats1[:cut] + mats2[cut:]
        child_thicks = thicks1[:cut] + thicks2[cut:]
        
        # If lengths are very different, this might be messy.
        # Let's ensure we don't exceed max_layers
        if len(child_mats) > max_layers:
            # Randomly drop some layers
            indices_to_keep = random.sample(range(len(child_mats)), max_layers)
            indices_to_keep.sort()
            child_mats = [child_mats[i] for i in indices_to_keep]
            child_thicks = [child_thicks[i] for i in indices_to_keep]
            
        # Ensure at least 1 layer
        if len(child_mats) == 0:
             child_mats = ['L']
             child_thicks = [lam_c / (4.0 * n_L)]
             
        # Enforce total thickness
        total = sum(child_thicks)
        if total > max_total_nm:
            scale = max_total_nm / total
            child_thicks = [t * scale for t in child_thicks]
            
        return child_mats, child_thicks

    def local_refine(mats, thicks, time_limit=1.0):
        """Apply hill climbing to a single individual."""
        start_time = time.time()
        current_mats = mats[:]
        current_thicks = np.array(thicks, dtype=float)
        current_score = eval_stack(current_mats, current_thicks)
        
        while time.time() - start_time < time_limit:
            improved = False
            
            # Perturb thicknesses
            for i in range(len(current_mats)):
                if current_thicks[i] < 1e-9:
                    continue
                original = current_thicks[i]
                for factor in [0.9, 1.1, 0.8, 1.2]:
                    new_thicks = current_thicks.copy()
                    new_thicks[i] = original * factor
                    new_thicks[i] = np.clip(new_thicks[i], 0, max_layer_nm)
                    
                    if np.sum(new_thicks) > max_total_nm:
                        new_thicks = new_thicks * (max_total_nm / np.sum(new_thicks))
                    
                    score = eval_stack(current_mats, new_thicks)
                    if score < current_score:
                        current_thicks = new_thicks
                        current_score = score
                        improved = True
                        break
            
            # Swap materials
            for i in range(len(current_mats)):
                if current_thicks[i] < 1e-9:
                    continue
                original_mat = current_mats[i]
                new_mat = 'H' if original_mat == 'L' else 'L'
                current_mats[i] = new_mat
                score = eval_stack(current_mats, current_thicks)
                if score < current_score:
                    current_score = score
                    improved = True
                else:
                    current_mats[i] = original_mat
            
            if not improved:
                break
                
        # Cleanup
        final_mats = []
        final_thicks = []
        for m, t in zip(current_mats, current_thicks):
            if t > 1e-6:
                final_mats.append(m)
                final_thicks.append(float(t))
        
        if len(final_mats) == 0:
             final_mats = ['L']
             final_thicks = [lam_c / (4.0 * n_L)]
             
        return final_mats, final_thicks, current_score

    # Initialize Population
    pop_size = 50
    population = []
    scores = []
    
    # Add some structured seeds
    for n_layers in [3, 5, 7, 9]:
        mats = []
        thicks = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats.append('L')
                thicks.append(lam_c / (4.0 * n_L))
            else:
                mats.append('H')
                thicks.append(lam_c / (4.0 * n_H))
        population.append((mats, thicks))
        
    # Fill rest with random
    while len(population) < pop_size:
        population.append(random_individual())
        
    for mats, thicks in population:
        scores.append(eval_stack(mats, thicks))
        
    best_mats, best_thicks = population[0]
    best_score = scores[0]
    
    start_time = time.time()
    time_budget = 55.0
    
    generation = 0
    while time.time() - start_time < time_budget:
        generation += 1
        
        # Sort by score
        sorted_indices = np.argsort(scores)
        population = [population[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        # Update best
        if scores[0] < best_score:
            best_score = scores[0]
            best_mats, best_thicks = population[0]
            
        # Elitism: keep top 10%
        elite_count = max(5, pop_size // 10)
        new_population = [population[i] for i in range(elite_count)]
        new_scores = [scores[i] for i in range(elite_count)]
        
        # Create offspring
        while len(new_population) < pop_size:
            # Selection (Tournament)
            p1_idx = np.random.choice(pop_size, size=3, replace=False)
            p1_idx = p1_idx[np.argmin([scores[i] for i in p1_idx])]
            
            p2_idx = np.random.choice(pop_size, size=3, replace=False)
            p2_idx = p2_idx[np.argmin([scores[i] for i in p2_idx])]
            
            child_mats, child_thicks = crossover(population[p1_idx][0], population[p1_idx][1], 
                                                 population[p2_idx][0], population[p2_idx][1])
            
            # Mutation
            child_mats, child_thicks = mutate(child_mats, child_thicks)
            
            # Validate
            if not valid_stack(child_mats, child_thicks):
                # Repair or skip
                # Simple repair: clip and scale
                child_thicks = np.array(child_thicks)
                child_thicks = np.clip(child_thicks, 0, max_layer_nm)
                total = sum(child_thicks)
                if total > max_total_nm:
                    child_thicks = child_thicks * (max_total_nm / total)
                child_thicks = list(child_thicks)
                
                if len(child_mats) == 0:
                    child_mats = ['L']
                    child_thicks = [lam_c / (4.0 * n_L)]
                
                if not valid_stack(child_mats, child_thicks):
                    continue
            
            child_score = eval_stack(child_mats, child_thicks)
            new_population.append((child_mats, child_thicks))
            new_scores.append(child_score)
            
        population = new_population
        scores = new_scores
        
        # Periodic local refinement on the best individual
        if generation % 5 == 0 and time.time() - start_time < time_budget - 5:
            # Take the current best
            curr_best_mats, curr_best_thicks = population[0]
            ref_mats, ref_thicks, ref_score = local_refine(curr_best_mats, curr_best_thicks, time_limit=1.0)
            
            # Replace the worst individual in population with the refined one
            worst_idx = np.argmax(scores)
            population[worst_idx] = (ref_mats, ref_thicks)
            scores[worst_idx] = ref_score
            
            if ref_score < best_score:
                best_score = ref_score
                best_mats, best_thicks = ref_mats, ref_thicks

    # Final polish on the best found
    final_mats, final_thicks, final_score = local_refine(best_mats, best_thicks, time_limit=2.0)
    
    # Cleanup final result
    final_mats_clean = []
    final_thicks_clean = []
    for m, t in zip(final_mats, final_thicks):
        if t > 1e-6:
            final_mats_clean.append(m)
            final_thicks_clean.append(float(t))
            
    if len(final_mats_clean) == 0:
        final_mats_clean = ['L']
        final_thicks_clean = [lam_c / (4.0 * n_L)]
        
    return final_mats_clean, final_thicks_clean
# EVOLVE-BLOCK-END
