"""Combined design program: Vectorized AR stack evaluation + DE global search + Simulated Annealing refinement."""
import numpy as np
import time
import random
import sys

# EVOLVE-BLOCK-START
def design(problem):
    """Design a broadband AR coating using vectorized evaluation + DE + SA optimization."""
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill (5s buffer for overhead)
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation (401 points from 400 to 800 nm)
    lambdas = np.linspace(band[0], band[1], 401)
    lambda_c = 0.5 * (band[0] + band[1])
    n_wl = len(lambdas)
    
    def mean_reflectance(materials, thicknesses):
        """Compute mean reflectance in percent over the band using vectorized characteristic matrix method."""
        n_layers = np.array([nL if m == 'L' else nH for m in materials], dtype=float)
        n = len(materials)
        if n == 0:
            r = (n0 - ns) / (n0 + ns)
            return (np.abs(r) ** 2) * 100.0
        
        # Vectorized computation over all wavelengths
        m_total = np.zeros((n_wl, 2, 2), dtype=complex)
        m_total[:, 0, 0] = 1.0
        m_total[:, 1, 1] = 1.0
        
        for i in range(n):
            ni = n_layers[i]
            di = thicknesses[i]
            delta = 2.0 * np.pi * ni * di / lambdas
            c = np.cos(delta)
            s = np.sin(delta)
            
            m_layer = np.zeros((n_wl, 2, 2), dtype=complex)
            m_layer[:, 0, 0] = c
            m_layer[:, 0, 1] = 1j * s / ni
            m_layer[:, 1, 0] = 1j * ni * s
            m_layer[:, 1, 1] = c
            
            m_total = np.einsum('wij,wjk->wik', m_total, m_layer)
        
        A = m_total[:, 0, 0]
        B = m_total[:, 0, 1]
        C = m_total[:, 1, 0]
        D = m_total[:, 1, 1]
        
        y_in = (C + D * ns) / (A + B * ns)
        r = (n0 - y_in) / (n0 + y_in)
        R = np.abs(r) ** 2
        
        return float(np.mean(R) * 100.0)
    
    def is_valid(materials, thicknesses):
        """Check constraints."""
        if len(materials) != len(thicknesses):
            return False
        if len(materials) > max_layers:
            return False
        if any(t < -1e-9 or t > max_layer_nm + 1e-9 for t in thicknesses):
            return False
        if sum(thicknesses) > max_total_nm + 1e-9:
            return False
        return True
    
    def clip_thicknesses(materials, thicknesses):
        """Clip thicknesses to valid range and ensure total constraint."""
        t = np.array(thicknesses, dtype=float)
        t = np.clip(t, 0.0, max_layer_nm)
        total = np.sum(t)
        if total > max_total_nm:
            scale = max_total_nm / total
            t = t * scale
        return [float(x) for x in t]
    
    # --- Generate candidate structures ---
    candidates = []
    
    qw_L = lambda_c / (4.0 * nL)
    qw_H = lambda_c / (4.0 * nH)
    
    # Standard alternating structures
    for n_layers in range(1, 11):
        # LHLH...
        mat = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        thick = [qw_L if i % 2 == 0 else qw_H for i in range(n_layers)]
        candidates.append((mat, thick))
        
        # HLHL...
        mat = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        thick = [qw_H if i % 2 == 0 else qw_L for i in range(n_layers)]
        candidates.append((mat, thick))
        
        # Add some random structures for diversity
        for _ in range(2):
            mat = [random.choice(['L', 'H']) for _ in range(n_layers)]
            # Avoid trivial all-same or too short
            if len(set(mat)) == 1:
                continue
            thick = [random.uniform(qw_L*0.5, qw_L*1.5) if m=='L' else random.uniform(qw_H*0.5, qw_H*1.5) for m, t in zip(mat, [0]*n_layers)]
            # Ensure total thickness is reasonable
            total_t = sum(thick)
            if total_t > max_total_nm:
                scale = max_total_nm / total_t
                thick = [t * scale for t in thick]
            candidates.append((mat, thick))
    
    # Evaluate all candidates and find the best
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    # Sort candidates by initial score to prioritize good starting points
    scored_candidates = []
    for mat, thick in candidates:
        if not is_valid(mat, thick):
            continue
        score = mean_reflectance(mat, thick)
        scored_candidates.append((score, mat, thick))
        
        if score < best_score:
            best_score = score
            best_materials = mat
            best_thicknesses = thick
    
    if best_materials is None:
        return ["L"], [qw_L]
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # --- Differential Evolution global search ---
    def differential_evolution(materials, x0, pop_size=40, max_iter=100, F=0.8, CR=0.9):
        """Differential Evolution for global optimization of thicknesses."""
        n = len(x0)
        
        # Initialize population
        pop = [list(x0)]
        for _ in range(pop_size - 1):
            ind = []
            for i in range(n):
                # Sample around the initial guess with some noise
                val = x0[i] + random.gauss(0, 20)
                ind.append(val)
            ind = clip_thicknesses(materials, ind)
            pop.append(ind)
        
        # Evaluate initial population
        fvals = [mean_reflectance(materials, ind) for ind in pop]
        
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx]
        best_f = fvals[best_idx]
        
        for gen in range(max_iter):
            if (time.time() - start_time) > DEADLINE:
                break
            
            new_pop = [list(pop[best_idx])]
            
            for i in range(pop_size - 1):
                # Select three distinct random indices different from i
                while True:
                    a, b, c = random.sample(range(pop_size), 3)
                    if a != i and b != i and c != i:
                        break
                
                # Mutation: v = pop[a] + F * (pop[b] - pop[c])
                v = []
                for j in range(n):
                    val = pop[a][j] + F * (pop[b][j] - pop[c][j])
                    v.append(val)
                v = clip_thicknesses(materials, v)
                
                # Crossover
                u = list(pop[i])
                for j in range(n):
                    if random.random() < CR or j == random.randint(0, n - 1):
                        u[j] = v[j]
                u = clip_thicknesses(materials, u)
                
                # Selection
                fu = mean_reflectance(materials, u)
                if fu <= fvals[i]:
                    new_pop.append(u)
                    fvals[i] = fu
                else:
                    new_pop.append(list(pop[i]))
            
            pop = new_pop
            
            # Track global best
            curr_best_idx = np.argmin(fvals)
            if fvals[curr_best_idx] < best_f:
                best_f = fvals[curr_best_idx]
                best_x = pop[curr_best_idx]
                
        return best_x, best_f

    # Run DE on top 3 candidates
    for score, materials, thick in scored_candidates[:3]:
        if (time.time() - start_time) > DEADLINE:
            break
        
        x_de, score_de = differential_evolution(materials, thick)
        
        if score_de < best_score:
            best_score = score_de
            best_materials = materials
            best_thicknesses = x_de
    
    # --- Simulated Annealing refinement ---
    def simulated_annealing(materials, x0, T_initial=10.0, T_final=0.1, cooling=0.995, max_iter=2000):
        """Simulated Annealing for local optimization."""
        n = len(x0)
        T = T_initial
        current_x = list(x0)
        current_f = mean_reflectance(materials, current_x)
        
        best_x = list(current_x)
        best_f = current_f
        
        for _ in range(max_iter):
            if (time.time() - start_time) > DEADLINE:
                break
                
            # Generate neighbor
            neighbor_x = list(current_x)
            i = random.randint(0, n - 1)
            step_size = random.uniform(1, 10)
            neighbor_x[i] += random.gauss(0, step_size)
            neighbor_x = clip_thicknesses(materials, neighbor_x)
            
            neighbor_f = mean_reflectance(materials, neighbor_x)
            
            # Acceptance criterion
            delta_f = neighbor_f - current_f
            if delta_f < 0 or random.random() < np.exp(-delta_f / T):
                current_x = neighbor_x
                current_f = neighbor_f
                
                if current_f < best_f:
                    best_f = current_f
                    best_x = list(current_x)
            
            T *= cooling
            if T < T_final:
                break
                
        return best_x, best_f

    # Run SA on the best candidate found so far
    x_sa, score_sa = simulated_annealing(best_materials, best_thicknesses)
    
    if score_sa < best_score:
        best_score = score_sa
        best_thicknesses = x_sa
    
    # Final validation and clamping
    best_thicknesses = clip_thicknesses(best_materials, best_thicknesses)
    
    if not is_valid(best_materials, best_thicknesses):
        # Fallback to first valid candidate
        for mat, thick in candidates:
            if is_valid(mat, thick):
                return mat, thick
        return ["L"], [qw_L]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
