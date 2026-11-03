"""Model-and-optimize approach for signed step functions minimizing max|autoconv|/(sum x)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _objective_grad_smooth(x, beta=50.0):
    """
    Compute soft-max surrogate objective and its gradient using log-sum-exp.
    Uses 3N-sized FFT for speed (sufficient for linear correlation of length N with kernel 2N-1).
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return 1e10, np.zeros_like(x)
    
    # Soft-max via log-sum-exp
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    val = soft_max / (s * s)
    
    # Gradient computation
    # d(soft_max)/dx[j] = sum_n w[n]*sign(ac[n]) * (x[j+n] + x[j-n])
    # where x[k]=0 for k outside [0,N-1]
    # Use FFT with size >= 3N-1 to avoid circular aliasing
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    L_fft = 1
    while L_fft < 3 * N - 1:
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    # term1[j] = sum_n g[n] * x[j+n] = cross-correlation
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    
    # term2[j] = sum_n g[n] * x[j-n]
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx = term1 + term2
    
    grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
    
    return val, grad

def _peak_targeted_polish(x, max_rounds=200):
    """
    Iteratively target the dominant autoconvolution lag.
    For the lag with max |ac|, compute the gradient of |ac[lag]| and take a
    small step in the negative direction to reduce that specific peak.
    This is a non-smooth local search that can escape from the soft-max plateau.
    """
    best_x = x.copy()
    best_val = _objective(best_x)
    
    for rnd in range(max_rounds):
        if _check_deadline():
            break
        
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        peak_lag = np.argmax(abs_ac)
        peak_sign = np.sign(ac[peak_lag])
        
        # Gradient of ac[peak_lag] w.r.t. x[j]:
        # ac[k] = sum_j x[j]*x[j+k-N+1] (for k in [0, 2N-2])
        # d(ac[peak_lag])/dx[j] = x[j+peak_lag-N+1] + x[j-peak_lag+N-1] (with bounds)
        lag = peak_lag - (N - 1)  # actual lag offset
        
        grad_peak = np.zeros(N)
        for j in range(N):
            # Contribution from x[j] * x[j+lag]
            j_plus = j + lag
            if 0 <= j_plus < N:
                grad_peak[j] += best_x[j_plus]
            # Contribution from x[j-lag] * x[j]
            j_minus = j - lag
            if 0 <= j_minus < N:
                grad_peak[j] += best_x[j_minus]
        
        # We want to reduce |ac[peak_lag]|, so move against peak_sign * grad_peak
        # Normalize step
        norm = np.linalg.norm(grad_peak)
        if norm < 1e-15:
            break
        
        # Try a few step sizes
        for step_size in [0.02, 0.01, 0.005, 0.001]:
            trial = best_x - peak_sign * step_size * grad_peak / norm
            val = _objective(trial)
            if val < best_val:
                best_val = val
                best_x = trial.copy()
                break
    
    return best_x, best_val

def _coordinate_descent(x, max_iters=3000, step_scale=0.05):
    """
    Coordinate descent polish: adjust individual coordinates to reduce objective.
    """
    best_val = _objective(x)
    best_x = x.copy()
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        improved = False
        for j in range(N):
            for delta in [-step_scale, -step_scale*0.5, step_scale*0.5, step_scale]:
                trial = best_x.copy()
                trial[j] += delta
                val = _objective(trial)
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_x, best_val

def _spectral_design(a, tilt=0.0, mod=0.0):
    """
    Generate a step function from spectral parameters.
    """
    a = np.clip(a, 0.5, 3.0)
    tilt = np.clip(tilt, -1.0, 1.0)
    mod = np.clip(mod, 0.0, 1.0)
    
    k = np.arange(N)
    c = 0.5 + a
    kf = k / N
    
    mag = np.exp(-c * kf**2)
    mag = mag * np.exp(tilt * kf)
    
    X = mag.astype(complex)
    x = np.fft.ifft(X).real
    x = x / np.max(np.abs(x))
    
    if mod > 0.01:
        t = np.linspace(0, 1, N)
        x = x * (1.0 + mod * 0.5 * np.sin(2*np.pi*t))
        x = x / np.max(np.abs(x))
    
    return x

def _prolate_spectral_design(bandwidth, phase_params=None):
    """
    Generate step function using prolate spheroidal sequence amplitude with optimized phase.
    """
    k = np.arange(N)
    kf = k / N
    
    # Prolate-like amplitude: concentrated in low frequencies
    mag = np.exp(-bandwidth * kf**2)
    
    if phase_params is not None:
        # Phase: sum of sinusoidal components
        phase = np.zeros(N)
        for freq, amp, ph in phase_params:
            phase += amp * np.sin(2 * np.pi * freq * kf + ph)
        X = mag * np.exp(1j * phase)
    else:
        X = mag.astype(complex)
    
    x = np.fft.ifft(X).real
    x = x / np.max(np.abs(x))
    
    return x

def _nsga2_search(x0, pop_size=40, generations=150, elite_size=10):
    """
    NSGA-II style multi-objective search.
    Objective 1: max |autoconv| / (sum x)^2  (peak ratio)
    Objective 2: sqrt(sum(autoconv^2)) / (sum x)^2  (L2 sidelobe energy ratio)
    The final score uses Objective 1, but exploring the Pareto front helps escape local minima.
    """
    rng = np.random.default_rng(999)
    
    def _compute_objectives(x):
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10, 1e10
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        s2 = s * s
        obj1 = np.max(abs_ac) / s2
        obj2 = np.sqrt(np.sum(ac * ac)) / s2
        return obj1, obj2
    
    # Initialize population
    population = []
    for i in range(pop_size):
        if i == 0:
            x = x0.copy()
        elif i == 1:
            x = x0 + rng.normal(0, 0.1, size=N)
        elif i == 2:
            x = x0 * (1 + 0.1 * rng.normal(size=N))
        else:
            # Mix of small and large perturbations
            scale = 0.05 * (1 + i / pop_size)
            x = x0 + rng.normal(0, scale, size=N)
        population.append(x)
    
    # Evaluate initial population
    pop_objs = np.array([_compute_objectives(x) for x in population])
    
    def _non_dominated_sort(objs):
        """Return list of fronts, each front is a list of indices."""
        n = len(objs)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [set() for _ in range(n)]
        fronts = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(objs[i] <= objs[j]) and np.any(objs[i] < objs[j]):
                    domination_count[j] += 1
                    dominated_set[i].add(j)
                elif np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                    domination_count[i] += 1
                    dominated_set[j].add(i)
        
        current_front = []
        for i in range(n):
            if domination_count[i] == 0:
                current_front.append(i)
        
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front
        
        # Assign remaining
        assigned = set()
        for f in fronts:
            assigned.update(f)
        if len(assigned) < n:
            remaining = [i for i in range(n) if i not in assigned]
            fronts.append(remaining)
        
        return fronts
    
    def _crowding_distance(objs, front_indices):
        """Compute crowding distance for indices in a front."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        distances = np.zeros(n)
        m = objs.shape[1]
        
        for obj_dim in range(m):
            sorted_idx = np.argsort(objs[front_indices, obj_dim])
            sorted_vals = objs[front_indices[sorted_idx], obj_dim]
            
            if sorted_vals[-1] - sorted_vals[0] < 1e-15:
                continue
            
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            
            for k in range(1, n - 1):
                distances[sorted_idx[k]] += (sorted_vals[k+1] - sorted_vals[k-1]) / (sorted_vals[-1] - sorted_vals[0])
        
        return distances
    
    best_obj1 = np.min(pop_objs[:, 0])
    best_x = population[np.argmin(pop_objs[:, 0])].copy()
    
    for gen in range(generations):
        if _check_deadline():
            break
        
        # Combine parent and offspring
        # Create offspring via crossover and mutation
        offspring = []
        for _ in range(pop_size):
            # Tournament selection
            idx1, idx2 = rng.choice(pop_size, 2, replace=False)
            if pop_objs[idx1, 0] <= pop_objs[idx2, 0]:
                parent1 = population[idx1]
            else:
                parent1 = population[idx2]
            
            if rng.random() < 0.5:
                idx3, idx4 = rng.choice(pop_size, 2, replace=False)
                if pop_objs[idx3, 0] <= pop_objs[idx4, 0]:
                    parent2 = population[idx3]
                else:
                    parent2 = population[idx4]
                # Blend crossover
                alpha = rng.uniform(0.3, 0.7)
                child = alpha * parent1 + (1 - alpha) * parent2
            else:
                child = parent1.copy()
            
            # Mutation
            mutation_rate = 0.1 / (1 + gen * 0.01)
            mask = rng.random(N) < mutation_rate
            child[mask] += rng.normal(0, 0.05, size=np.sum(mask))
            
            offspring.append(child)
        
        # Evaluate offspring
        off_objs = np.array([_compute_objectives(x) for x in offspring])
        
        # Combine and select
        combined_pop = population + offspring
        combined_objs = np.vstack([pop_objs, off_objs])
        
        fronts = _non_dominated_sort(combined_objs)
        
        new_population = []
        new_objs = []
        
        for front in fronts:
            if len(new_population) + len(front) <= pop_size:
                new_population.extend([combined_pop[i] for i in front])
                new_objs.extend([combined_objs[i] for i in front])
            else:
                # Need to select from this front using crowding distance
                remaining = pop_size - len(new_population)
                cd = _crowding_distance(combined_objs, front)
                sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                selected = sorted_front[:remaining]
                new_population.extend([combined_pop[i] for i in selected])
                new_objs.extend([combined_objs[i] for i in selected])
                break
        
        population = new_population
        pop_objs = np.array(new_objs)
        
        # Track best
        min_idx = np.argmin(pop_objs[:, 0])
        if pop_objs[min_idx, 0] < best_obj1:
            best_obj1 = pop_objs[min_idx, 0]
            best_x = population[min_idx].copy()
    
    return best_x, best_obj1

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Structured initializations
    t = np.linspace(0, 1, N)
    initializations = [
        np.ones(N),
        np.linspace(2.0, 1.0, N),
        np.linspace(1.0, 2.0, N),
        1.0 + 0.5 * np.cos(np.pi * t),
        1.0 + np.cos(np.pi * t),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.sin(np.pi * t),
        np.exp(-2.0 * t),
        np.exp(2.0 * t - 2.0),
        1.0 + 0.2 * np.cos(2 * np.pi * 3 * t),
        1.0 + 0.3 * np.cos(2 * np.pi * 5 * t),
    ]
    
    for r in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 1.1, 1.2, 1.5, 2.0]:
        initializations.append(r ** np.arange(N))
    
    # Add some random smooth initializations
    rng_init = np.random.default_rng(123)
    for _ in range(30):
        freqs = rng_init.uniform(0.1, 3.0, size=5)
        phases = rng_init.uniform(0, 2*np.pi, size=5)
        amps = rng_init.uniform(0.1, 0.5, size=5)
        sig = 1.0 + sum(a * np.cos(2*np.pi*f*t + p) for a, f, p in zip(amps, freqs, phases))
        initializations.append(sig)
    
    # Add prolate spectral designs
    for bw in [0.5, 1.0, 1.5, 2.0, 3.0]:
        initializations.append(_prolate_spectral_design(bw))
    
    for x0 in initializations:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: L-BFGS-B with smooth surrogate - multiple beta values
    for beta in [30.0, 50.0, 100.0]:
        if _check_deadline():
            break
        
        def _smooth_obj(x, b=beta):
            val, grad = _objective_grad_smooth(x, beta=b)
            return val, grad
        
        try:
            result = minimize(_smooth_obj, best_x, method='L-BFGS-B', jac=True,
                            options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-14})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: NSGA-II multi-objective search to explore Pareto front
    if not _check_deadline():
        try:
            nsga_x, nsga_val = _nsga2_search(best_x, pop_size=30, generations=100)
            if nsga_val < best_val:
                best_val = nsga_val
                best_x = nsga_x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Spectral parameter optimization
    def _param_objective(params):
        x = _spectral_design(params[0], params[1], params[2])
        return _objective(x)
    
    param_initials = [
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.5, 0.5, 0.0],
        [1.5, -0.5, 0.0],
        [1.0, 0.0, 0.5],
        [2.0, 0.0, 0.5],
        [0.5, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
    ]
    
    for p0 in param_initials:
        if _check_deadline():
            break
        try:
            result = minimize(_param_objective, p0, method='Nelder-Mead', 
                            options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
            if result.fun < best_val:
                best_val = result.fun
                best_x = _spectral_design(result.x[0], result.x[1], result.x[2])
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Prolate spectral phase optimization
    def _prolate_phase_objective(params):
        bandwidth = params[0]
        phase_params = []
        idx = 1
        for i in range(3):
            freq = params[idx]; idx += 1
            amp = params[idx]; idx += 1
            ph = params[idx]; idx += 1
            phase_params.append((freq, amp, ph))
        x = _prolate_spectral_design(bandwidth, phase_params)
        return _objective(x)
    
    prolate_initials = []
    for bw in [0.5, 1.0, 1.5, 2.0]:
        for _ in range(3):
            p0 = [bw]
            for _ in range(3):
                p0.append(rng_init.uniform(0.5, 3.0))  # freq
                p0.append(rng_init.uniform(0.1, 1.0))  # amp
                p0.append(rng_init.uniform(0, 2*np.pi))  # phase
            prolate_initials.append(p0)
    
    for p0 in prolate_initials:
        if _check_deadline():
            break
        try:
            result = minimize(_prolate_phase_objective, p0, method='Nelder-Mead', 
                            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_val:
                best_val = result.fun
                bandwidth = result.x[0]
                phase_params = []
                idx = 1
                for i in range(3):
                    freq = result.x[idx]; idx += 1
                    amp = result.x[idx]; idx += 1
                    ph = result.x[idx]; idx += 1
                    phase_params.append((freq, amp, ph))
                best_x = _prolate_spectral_design(bandwidth, phase_params)
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Peak-targeted polish (attacks dominant lag directly)
    best_x, best_val = _peak_targeted_polish(best_x, max_rounds=300)
    
    if _check_deadline():
        return best_x
    
    # Phase 7: Coordinate descent polish
    best_x, best_val = _coordinate_descent(best_x, max_iters=3000, step_scale=0.03)
    
    if _check_deadline():
        return best_x
    
    # Phase 8: Another round of peak-targeted polish
    best_x, best_val = _peak_targeted_polish(best_x, max_rounds=200)
    
    if _check_deadline():
        return best_x
    
    # Phase 9: Another NSGA-II pass from the improved starting point
    if not _check_deadline():
        try:
            nsga_x2, nsga_val2 = _nsga2_search(best_x, pop_size=25, generations=80)
            if nsga_val2 < best_val:
                best_val = nsga_val2
                best_x = nsga_x2.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 10: Final high-precision L-BFGS-B polish with very tight tolerance
    def _smooth_obj_final(x):
        val, grad = _objective_grad_smooth(x, beta=200.0)
        return val, grad
    
    try:
        result = minimize(_smooth_obj_final, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 1000, 'gtol': 1e-12, 'ftol': 1e-16, 'maxls': 50})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 11: Final peak-targeted polish after BFGS
    best_x, best_val = _peak_targeted_polish(best_x, max_rounds=100)
    
    if _check_deadline():
        return best_x
    
    # Phase 12: Random perturbation refinement with adaptive step
    rng = np.random.default_rng(42)
    step_size = 0.02
    stall_count = 0
    
    for iteration in range(20000):
        if _check_deadline():
            break
        
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        
        val = _objective(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            stall_count = 0
            step_size = max(0.001, step_size * 0.9999)
        else:
            stall_count += 1
            if stall_count % 5000 == 0:
                step_size = min(0.2, step_size * 1.05)
    
    # Phase 13: Final polish: tight BFGS on exact objective
    if not _check_deadline():
        try:
            result = minimize(_objective, best_x, method='BFGS', 
                            options={'maxiter': 300, 'gtol': 1e-14})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Final peak-targeted micro-polish
    if not _check_deadline():
        best_x, best_val = _peak_targeted_polish(best_x, max_rounds=50)
    
    return best_x
# EVOLVE-BLOCK-END
