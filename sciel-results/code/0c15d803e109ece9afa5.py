"""Broadband antireflection coating design via discrete genetic algorithm over quarter-wave thickness grid."""
import numpy as np
import time

def _compute_reflectance(wavelengths_nm, indices, thicknesses_nm, ambient_idx, substrate_idx):
    """
    Compute reflectance (fraction, 0-1) at each wavelength using the characteristic matrix method.
    """
    n_layers = len(indices)
    if n_layers == 0:
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return np.full(len(wavelengths_nm), r0 * r0)
    
    k = 2.0 * np.pi / wavelengths_nm
    
    M00 = np.ones(len(wavelengths_nm), dtype=complex)
    M01 = np.zeros(len(wavelengths_nm), dtype=complex)
    M10 = np.zeros(len(wavelengths_nm), dtype=complex)
    M11 = np.ones(len(wavelengths_nm), dtype=complex)
    
    for i in range(n_layers):
        n_i = indices[i]
        d_i = thicknesses_nm[i]
        delta = k * n_i * d_i
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        B00 = cos_d
        B01 = 1j * sin_d / n_i
        B10 = 1j * n_i * sin_d
        B11 = cos_d
        
        new_M00 = M00 * B00 + M01 * B10
        new_M01 = M00 * B01 + M01 * B11
        new_M10 = M10 * B00 + M11 * B10
        new_M11 = M10 * B01 + M11 * B11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    Y_out = substrate_idx
    Y_in = (M10 + M11 * Y_out) / (M00 + M01 * Y_out)
    
    r = (ambient_idx - Y_in) / (ambient_idx + Y_in)
    R = np.abs(r) ** 2
    
    return R

def _eval_stack(labels, thicknesses_nm, problem, wavelengths_nm):
    """Evaluate a stack: returns mean reflectance in percent."""
    indices = [problem["materials"][label] for label in labels]
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses_nm,
                             problem["ambient_index"], problem["substrate_index"])
    return np.mean(R) * 100.0

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    # Discrete thickness grid: quarter-wave multiples at center wavelength
    # For L (1.38): qw = 600/4/1.38 ≈ 108.7 nm
    # For H (2.35): qw = 600/4/2.35 ≈ 63.8 nm
    # Use a grid of thicknesses: 0, qw/2, qw, 3qw/2, 2qw, ... up to max_layer_nm
    # But to keep it simpler, use a fixed set of candidate thicknesses per material
    
    # Candidate thicknesses: multiples of ~50 nm from 0 to 400
    # Actually, let's use quarter-wave and half-wave steps
    # L: qw ≈ 108.7, so candidates: 0, 54, 109, 163, 217, 272, 326, 380
    # H: qw ≈ 63.8, so candidates: 0, 32, 64, 95, 127, 159, 191, 223, 255, 287, 319, 351, 383
    
    # Simpler: use a uniform grid of 0, 25, 50, ..., 400 (17 values)
    # This gives more flexibility while keeping the search space manageable
    
    thickness_grid = np.arange(0, 401, 25)  # 0, 25, 50, ..., 400 (17 values)
    n_thick_opts = len(thickness_grid)
    
    # GA representation: each individual is (n_layers, 2) array
    # n_layers: number of layers (1 to max_layers)
    # For each layer: [type (0=L, 1=H), thickness_index]
    
    # We'll encode as: for a fixed max_layers, use a "padding" approach
    # Or better: variable-length encoding with a fixed max
    
    # Let's use a fixed encoding: up to max_layers positions
    # Each position: (active, type, thickness_idx)
    # active=0 means this position is unused (layer not present)
    # But this complicates things. Instead, let's try specific layer counts.
    
    # Strategy: GA over (type sequence, thickness sequence) for each layer count
    # Then take the best across all layer counts
    
    best_labels = None
    best_thicknesses = None
    best_score = np.inf
    
    # Try different layer counts
    layer_counts_to_try = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    layer_counts_to_try = [lc for lc in layer_counts_to_try if lc <= max_layers]
    
    # Time allocation: give more time to promising layer counts
    # Start with known good ones: 7, 9, 11 layers
    layer_counts_ordered = [7, 9, 11, 5, 13, 3, 15, 17, 19]
    layer_counts_ordered = [lc for lc in layer_counts_ordered if lc <= max_layers]
    
    # GA parameters
    pop_size = 200
    n_generations = 300
    crossover_rate = 0.7
    mutation_rate = 0.15
    
    for n_layers in layer_counts_ordered:
        if time.time() > deadline - 10:
            break
        
        # Encode: for each layer, (type: 0 or 1, thickness_idx: 0 to n_thick_opts-1)
        # So chromosome is a flat array of length 2*n_layers
        
        def decode(chromosome):
            """Decode chromosome to (labels, thicknesses)."""
            labels = []
            thicknesses = []
            for i in range(n_layers):
                t = chromosome[2*i]
                ti = chromosome[2*i + 1]
                if t < 0.5:
                    labels.append('L')
                else:
                    labels.append('H')
                thicknesses.append(thickness_grid[int(round(ti))])
            return labels, thicknesses
        
        def evaluate(chromosome):
            """Evaluate a chromosome: returns mean reflectance in percent."""
            labels, thicknesses = decode(chromosome)
            # Check total thickness constraint
            if sum(thicknesses) > max_total_nm:
                return 100.0 + (sum(thicknesses) - max_total_nm)
            return _eval_stack(labels, thicknesses, problem, wavelengths_nm)
        
        # Initialize population
        np.random.seed(42 + n_layers)
        pop = np.zeros((pop_size, 2 * n_layers))
        for i in range(pop_size):
            for j in range(n_layers):
                pop[i, 2*j] = np.random.random()  # type: 0 or 1
                pop[i, 2*j + 1] = np.random.randint(0, n_thick_opts)
        
        # Initial evaluation
        scores = np.array([evaluate(pop[i]) for i in range(pop_size)])
        
        # Track best
        best_idx = np.argmin(scores)
        best_chrom = pop[best_idx].copy()
        best_score_local = scores[best_idx]
        
        # GA loop
        for gen in range(n_generations):
            if time.time() > deadline - 10:
                break
            
            # Sort by fitness
            idx = np.argsort(scores)
            pop = pop[idx]
            scores = scores[idx]
            
            # Elitism: keep top 10%
            n_elite = max(5, pop_size // 10)
            
            # Create new population
            new_pop = np.zeros_like(pop)
            new_pop[:n_elite] = pop[:n_elite]
            
            for i in range(n_elite, pop_size):
                # Tournament selection
                def tournament_select(pop, scores, k=3):
                    candidates = np.random.choice(len(pop), size=k, replace=False)
                    best_c = candidates[np.argmin(scores[candidates])]
                    return pop[best_c].copy()
                
                parent1 = tournament_select(pop, scores)
                parent2 = tournament_select(pop, scores)
                
                # Crossover: uniform
                if np.random.random() < crossover_rate:
                    child = np.zeros_like(parent1)
                    for j in range(2 * n_layers):
                        if np.random.random() < 0.5:
                            child[j] = parent1[j]
                        else:
                            child[j] = parent2[j]
                else:
                    child = parent1.copy()
                
                # Mutation
                for j in range(2 * n_layers):
                    if np.random.random() < mutation_rate:
                        if j % 2 == 0:  # type
                            child[j] = 1.0 - child[j]  # flip
                        else:  # thickness
                            child[j] = np.random.randint(0, n_thick_opts)
                
                new_pop[i] = child
            
            # Evaluate new population
            new_scores = np.array([evaluate(new_pop[i]) for i in range(pop_size)])
            
            # Update best
            new_best_idx = np.argmin(new_scores)
            if new_scores[new_best_idx] < best_score_local:
                best_score_local = new_scores[new_best_idx]
                best_chrom = new_pop[new_best_idx].copy()
            
            # Replace population
            pop = new_pop
            scores = new_scores
        
        # Update global best
        if best_score_local < best_score:
            best_score = best_score_local
            best_labels, best_thicknesses = decode(best_chrom)
        
        # Early termination
        if best_score < 0.3:
            break
    
    # Fallback
    if best_labels is None:
        # Simple quarter-wave AR coating
        labels = ['L']
        thicknesses = [center_wl / (4.0 * materials['L'])]
    else:
        # Ensure constraints
        thicknesses = np.clip(best_thicknesses, 0, max_layer_nm)
        if sum(thicknesses) > max_total_nm:
            scale = max_total_nm / sum(thicknesses)
            thicknesses = [t * scale for t in thicknesses]
    
    return best_labels, [float(t) for t in best_thicknesses]
