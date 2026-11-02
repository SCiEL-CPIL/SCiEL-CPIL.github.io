import numpy as np
import time

N = 100
W = 2.0 / N
TARGET_SUM = N / 2.0

def _compute_overlap_max(h):
    """
    Compute the maximum overlap of h with shifted 1-h.
    overlap[k] = W * (sum_h - autocorr[k])
    We want to minimize max(overlap[k]).
    """
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    sum_h = np.sum(h)
    overlaps = W * (sum_h - autocorr)
    return np.max(overlaps)

def _project_to_simplex_box(h, target_sum):
    """
    Project h onto the intersection of the box [0,1] and the hyperplane sum(h) = target_sum.
    Uses binary search on the Lagrange multiplier lambda.
    """
    lo = np.min(h) - 1.0
    hi = np.max(h) + 1.0
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        proj = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(proj)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
            
    lambda_opt = (lo + hi) / 2.0
    proj = np.clip(h - lambda_opt, 0.0, 1.0)
    return proj

def _spectral_to_time(magnitudes, phases):
    """
    Convert spectral magnitudes and phases to time domain.
    magnitudes: array of shape (N,)
    phases: array of shape (N,)
    Returns h (real part), ensuring symmetry for real signal.
    """
    # Construct complex spectrum
    # For a real signal, H[k] = conj(H[N-k])
    # We define magnitudes and phases for k=0..N-1
    # To ensure real output, we enforce Hermitian symmetry.
    # Easier: just build H from mag/phase, take real part of ifft.
    # But this might introduce imaginary residue if not careful.
    # Standard trick: H = M * exp(i*P). ifft(H).real
    # However, ifft of arbitrary complex H is complex.
    # To get a real signal h, we should ensure H is Hermitian.
    # Let's define independent phases for k=1..N/2-1 and k=N/2 (if even).
    # k=0 and k=N/2 (if even) must be real.
    
    H = np.zeros(N, dtype=complex)
    
    # DC component (k=0)
    H[0] = magnitudes[0] # Phase 0 for DC to keep mean positive/real
    
    # Nyquist (k=N/2) if N is even
    if N % 2 == 0:
        H[N//2] = magnitudes[N//2] # Phase 0 for Nyquist
    
    # Positive frequencies
    for k in range(1, N//2):
        H[k] = magnitudes[k] * np.exp(1j * phases[k])
        H[N-k] = np.conj(H[k])
        
    h = np.fft.ifft(H).real
    return h

def _local_search(h, time_budget):
    """
    Perform local search in time domain to refine h.
    Uses small random perturbations and projection.
    """
    best_h = h.copy()
    best_score = _compute_overlap_max(best_h)
    
    np.random.seed(int(time.time() * 1000) % 10000)
    
    start = time.time()
    step_size = 0.05
    
    while time.time() - start < time_budget:
        # Random perturbation
        noise = np.random.normal(0, step_size, N)
        h_new = best_h + noise
        h_new = _project_to_simplex_box(h_new, TARGET_SUM)
        
        score_new = _compute_overlap_max(h_new)
        
        if score_new < best_score:
            best_h = h_new
            best_score = score_new
            # Reduce step size slightly on improvement
            step_size *= 0.95
        else:
            # Occasionally increase step size to escape local minima
            if np.random.random() < 0.05:
                step_size *= 1.1
                step_size = min(step_size, 0.2)
                
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # Strategy:
    # 1. Define a spectral magnitude profile that encourages low autocorrelation.
    #    A flat magnitude spectrum (white noise) has a "spike" at lag 0 and near-zero elsewhere.
    #    We want to minimize the max overlap.
    #    overlap[k] = W * (N/2 - autocorr[k]).
    #    Minimizing max overlap is equivalent to maximizing min autocorr[k] for k != 0?
    #    Actually, autocorr[0] is fixed by sum of squares.
    #    If we make the spectrum "flat" but with specific phases, we can shape the autocorrelation.
    #    However, a simpler structural approach:
    #    Use a Genetic Algorithm to optimize the PHASES of the spectral components.
    #    The magnitudes are fixed to a profile that ensures the signal is in [0,1] after projection
    #    and has the correct mean.
    
    # Fixed Magnitudes:
    # To ensure h is in [0,1] and sum=N/2, the DC component (mean) must be 0.5.
    # Sum h = N * 0.5 = N/2.
    # H[0] = N * 0.5 = N/2.
    # For other frequencies, we choose magnitudes that allow h to stay bounded.
    # A common choice for bounded signals is to decay magnitudes with frequency.
    # Let's try a magnitude profile: M[k] = A / (1 + (k/Beta)^2)
    # We need to tune A and Beta.
    # Let's start with a "flat" spectrum for k>0 but scaled down to prevent large oscillations.
    # If we use white noise phases, the signal is Gaussian-like.
    # Std dev of white noise with flat spectrum M[k]=c is c * sqrt(N).
    # We want std dev small enough so that 0.5 + 3*std < 1 and 0.5 - 3*std > 0.
    # So std < 1/6 approx.
    # Variance = (1/N) sum |H[k]|^2.
    # If M[k] = c for k=1..N-1, Var = c^2.
    # So c = 0.15 gives std ~ 0.15.
    
    magnitudes = np.zeros(N)
    magnitudes[0] = N / 2.0  # DC component for mean 0.5
    
    # For k > 0, use a profile that decays slightly to ensure smoothness and bounds
    # Let's try a flat profile first, then optimize phases.
    # If flat profile leads to h out of [0,1], projection will clip it, which changes the spectrum.
    # So we must be careful.
    
    # Alternative: Optimize BOTH phases and a global amplitude scale?
    # No, let's fix magnitudes to a "safe" profile and optimize phases.
    # Profile: M[k] = 0.1 for k=1..N-1.
    # This gives a signal with mean 0.5 and std ~0.1.
    # h will likely be in [0.2, 0.8] before projection.
    
    for k in range(1, N):
        # Slight decay to reduce high freq noise which might cause sharp spikes
        magnitudes[k] = 0.1 * (1.0 / (1.0 + k/10.0))
        
    # Genetic Algorithm to optimize phases
    # Chromosome: phases for k=1..N/2-1 (and N/2 if even, but phase 0 there)
    # Number of independent phases: N//2 - 1 (if even) or (N-1)//2
    n_phases = N // 2 - 1 if N % 2 == 0 else (N - 1) // 2
    
    pop_size = 50
    generations = 100
    
    # Initialize population with random phases
    population = np.random.uniform(0, 2 * np.pi, (pop_size, n_phases))
    
    best_h_global = np.full(N, 0.5)
    best_score_global = _compute_overlap_max(best_h_global)
    
    # Time budget for GA
    ga_time_budget = 60
    local_search_budget = 20
    
    gen_start = time.time()
    
    for gen in range(generations):
        if time.time() > start_time + ga_time_budget:
            break
            
        scores = np.zeros(pop_size)
        h_list = []
        
        for i in range(pop_size):
            phases = population[i]
            
            # Construct full phase array
            full_phases = np.zeros(N)
            if N % 2 == 0:
                # k=1..N/2-1
                full_phases[1:N//2] = phases
                # k=N/2 is 0
                # k=N/2+1..N-1 are determined by conjugate symmetry
            else:
                full_phases[1:(N+1)//2] = phases
                
            h = _spectral_to_time(magnitudes, full_phases)
            
            # Project to feasible region
            h = _project_to_simplex_box(h, TARGET_SUM)
            
            h_list.append(h)
            scores[i] = _compute_overlap_max(h)
            
        # Track global best
        idx_best = np.argmin(scores)
        if scores[idx_best] < best_score_global:
            best_score_global = scores[idx_best]
            best_h_global = h_list[idx_best].copy()
            
        # Selection: Keep top 50%
        sorted_indices = np.argsort(scores)
        survivors = sorted_indices[:pop_size // 2]
        
        # Crossover and Mutation
        new_population = np.zeros((pop_size, n_phases))
        
        # Elitism: Keep the best one
        new_population[0] = population[survivors[0]]
        
        for i in range(1, pop_size):
            # Select parents
            p1_idx = survivors[np.random.randint(0, len(survivors))]
            p2_idx = survivors[np.random.randint(0, len(survivors))]
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            # Uniform Crossover
            mask = np.random.rand(n_phases) < 0.5
            child = np.where(mask, p1, p2)
            
            # Mutation
            if np.random.random() < 0.3:
                # Gaussian mutation on phases
                mutation_mask = np.random.rand(n_phases) < 0.1
                child[mutation_mask] += np.random.normal(0, 0.5, np.sum(mutation_mask))
                child[mutation_mask] %= (2 * np.pi)
                
            new_population[i] = child
            
        population = new_population
        
    # Final Local Search on the best found
    if time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 2:
            best_h_global, best_score_global = _local_search(best_h_global, remaining_time)
            
    # Final check
    best_h_global = _project_to_simplex_box(best_h_global, TARGET_SUM)
    
    if not np.isclose(np.sum(best_h_global), TARGET_SUM, atol=1e-5):
        best_h_global = np.full(N, 0.5)
        
    return best_h_global
