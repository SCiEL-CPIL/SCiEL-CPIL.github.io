"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 135.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_overlaps(h):
    """Compute R_k for k=1..N-1. R_k = (2/N) sum h[i]h[i+k]."""
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    return corrs

def _project_to_simplex(h, target_sum):
    """Project h to satisfy sum(h) = target_sum and 0<=h<=1."""
    h = np.clip(h, 0, 1)
    diff = target_sum - np.sum(h)
    for _ in range(20):
        if abs(diff) < 1e-12:
            break
        mask = (h > 1e-8) & (h < 1 - 1e-8)
        if np.sum(mask) == 0:
            break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0, 1)
        diff = target_sum - np.sum(h)
    # Final exact adjustment
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-10:
        h += diff / len(h)
        h = np.clip(h, 0, 1)
    return h

def _fourier_to_h(coeffs, n):
    """Convert Fourier coefficients (DC removed) to h values.
    coeffs: array of shape (n,) where coeffs[0] is DC (ignored), coeffs[1:n] are complex.
    Returns h in [0,1] with sum = n/2.
    """
    # Build h from Fourier: h[j] = 0.5 + sum_{k=1}^{n-1} Re(c_k * exp(2*pi*i*k*j/n))
    # Use only first few harmonics for stability
    h = np.full(n, 0.5)
    for k in range(1, n):
        if k >= len(coeffs) or np.abs(coeffs[k]) < 1e-15:
            continue
        c = coeffs[k]
        h += np.real(c * np.exp(2j * np.pi * k * np.arange(n) / n))
    h = np.clip(h, 0, 1)
    # Project to exact sum
    h = _project_to_simplex(h, n / 2.0)
    return h

def _h_to_fourier(h, n):
    """Convert h to Fourier coefficients (DC removed)."""
    h_centered = h - np.mean(h)
    coeffs = np.zeros(n, dtype=complex)
    for k in range(1, n):
        coeffs[k] = np.sum(h_centered * np.exp(-2j * np.pi * k * np.arange(n) / n)) / n
    return coeffs

def _max_overlap_subgrad(h):
    """Compute subgradient of max_k R_k w.r.t. h.
    Returns (max_val, subgradient) where subgradient points in direction to decrease max overlap.
    """
    n = len(h)
    corrs = _compute_overlaps(h)
    max_val = np.max(corrs)
    
    # Find which k achieves max (or near-max for smoothing)
    threshold = max_val - 1e-6
    active = corrs >= threshold
    
    subgrad = np.zeros(n)
    for k in range(1, n):
        if not active[k-1]:
            continue
        w = (2.0 / n) * corrs[k-1] / max_val if max_val > 1e-15 else (2.0/n)
        subgrad[:n-k] += w * h[k:]
        subgrad[k:] += w * h[:n-k]
    
    # Normalize subgradient
    norm = np.linalg.norm(subgrad)
    if norm > 1e-15:
        subgrad /= norm
    
    return max_val, subgrad

def _ising_local_field(h, n, mean_field_strength=0.1):
    """Compute local field for each site in the Ising model.
    Antiferromagnetic coupling: neighbors should have opposite values.
    Mean-field term enforces sum constraint.
    """
    # Spin: s_i = 2*h_i - 1 (maps [0,1] to [-1,1])
    s = 2.0 * h - 1.0
    
    # Antiferromagnetic interaction: -J * s_i * s_{i+1} with J>0
    # Energy contribution to site i: J * (s_{i-1} + s_{i+1})
    J = 1.0  # coupling strength
    field_afm = np.zeros(n)
    for i in range(n):
        left = s[(i-1) % n]
        right = s[(i+1) % n]
        field_afm[i] = J * (left + right)
    
    # Mean-field term to enforce mean(s) = 0 (i.e., sum(h) = n/2)
    mean_s = np.mean(s)
    field_mf = -mean_field_strength * mean_s
    
    # Total local field
    field = field_afm + field_mf
    
    return field

def _glauber_step(h, temperature=0.1):
    """Perform one Glauber dynamics step on the spin system.
    Returns updated h.
    """
    n = len(h)
    field = _ising_local_field(h, n)
    
    # For each site, compute probability to flip
    # s_i = 2*h_i - 1, flip means s_i -> -s_i
    # Delta E for flipping site i: 2 * s_i * field_i (for AFM)
    s = 2.0 * h - 1.0
    delta_E = 2.0 * s * field
    
    # Metropolis acceptance: accept if delta_E < 0, else with prob exp(-delta_E/T)
    flips = np.zeros(n, dtype=bool)
    for i in range(n):
        if delta_E[i] < 0:
            flips[i] = True
        else:
            if np.random.random() < np.exp(-delta_E[i] / max(temperature, 1e-10)):
                flips[i] = True
    
    # Apply flips: s_i -> -s_i means h_i -> 1 - h_i
    h_new = h.copy()
    h_new[flips] = 1.0 - h_new[flips]
    
    # Project to satisfy constraints
    h_new = _project_to_simplex(h_new, n / 2.0)
    
    return h_new

def _soft_max_overlap(h, beta=10.0):
    """Compute smooth approximation of max overlap using log-sum-exp.
    Returns (soft_max_value, weights) where weights indicate which lags contribute.
    """
    n = len(h)
    corrs = _compute_overlaps(h)
    
    # Log-sum-exp for numerical stability
    max_c = np.max(corrs)
    if max_c < 1e-15:
        return 0.0, np.ones(len(corrs)) / len(corrs)
    
    # soft_max = max_c + (1/beta) * log(sum(exp(beta*(c_i - max_c))))
    scaled = beta * (corrs - max_c)
    # Clip to avoid overflow
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    soft_max = max_c + (1.0 / beta) * log_sum
    
    # Weights: proportional to exp(beta*(c_i - max_c))
    weights = np.exp(scaled)
    weights /= np.sum(weights)
    
    return soft_max, weights

def _soft_max_subgrad(h, beta=10.0):
    """Compute subgradient of soft max overlap.
    Returns (soft_max_value, subgradient).
    """
    n = len(h)
    corrs = _compute_overlaps(h)
    
    max_c = np.max(corrs)
    if max_c < 1e-15:
        return 0.0, np.zeros(n)
    
    scaled = beta * (corrs - max_c)
    scaled = np.clip(scaled, -50, 50)
    weights = np.exp(scaled)
    weights /= np.sum(weights)
    
    soft_max = max_c + (1.0 / beta) * np.log(np.sum(np.exp(scaled)))
    
    # Subgradient: weighted sum of individual correlation gradients
    subgrad = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1]
        if w < 1e-10:
            continue
        # dR_k/dh: R_k = (2/n) sum_{i=0}^{n-k-1} h[i]*h[i+k]
        # dR_k/dh[i] = (2/n)*(h[i+k] if i < n-k else 0) + (2/n)*(h[i-k] if i >= k else 0)
        subgrad[:n-k] += w * (2.0/n) * h[k:]
        subgrad[k:] += w * (2.0/n) * h[:n-k]
    
    return soft_max, subgrad

def construct():
    n = N
    target_sum = n / 2.0
    
    best_h = np.full(n, 0.5)
    best_score = float('inf')
    
    def _score(h):
        corrs = _compute_overlaps(h)
        return 1.0 - np.min(corrs)
    
    best_score = _score(best_h)
    
    np.random.seed(42)
    
    # Phase 1: Ising Glauber dynamics with simulated annealing
    # Start from various initial conditions
    
    initial_configs = []
    
    # 1. All 0.5 (trivial)
    initial_configs.append(np.full(n, 0.5))
    
    # 2. Alternating pattern
    alt = np.zeros(n)
    alt[::2] = 1.0
    alt[1::2] = 0.0
    initial_configs.append(_project_to_simplex(alt, target_sum))
    
    # 3. Random binary
    for _ in range(5):
        rand_bin = np.random.choice([0, 1], size=n).astype(float)
        initial_configs.append(_project_to_simplex(rand_bin, target_sum))
    
    # 4. Fourier-based initializations
    for n_harm in [1, 2, 3]:
        coeffs = np.zeros(n, dtype=complex)
        for k in range(1, n_harm+1):
            amp = np.random.uniform(0.1, 0.3)
            phase = np.random.uniform(0, 2*np.pi)
            coeffs[k] = amp * np.exp(1j * phase)
        initial_configs.append(_fourier_to_h(coeffs, n))
    
    for init_idx, h_init in enumerate(initial_configs):
        if _time_remaining() < 10.0:
            break
        
        h_current = h_init.copy()
        current_score = _score(h_current)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h_current.copy()
        
        # Simulated annealing schedule
        T_start = 1.0
        T_end = 0.01
        n_steps = 3000
        cooling = 0.995
        
        T = T_start
        for step in range(n_steps):
            if _time_remaining() < 5.0:
                break
            
            # Glauber step
            h_trial = _glauber_step(h_current, temperature=T)
            trial_score = _score(h_trial)
            
            # Metropolis acceptance for the overall objective
            if trial_score < current_score:
                h_current = h_trial
                current_score = trial_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = h_current.copy()
            else:
                # Accept with probability exp(-(trial_score - current_score)/T)
                delta = trial_score - current_score
                if delta < 1e-12 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                    h_current = h_trial
                    current_score = trial_score
            
            # Cool down
            T *= cooling
            if T < T_end:
                T = T_end
        
        # Phase 2: Soft-max gradient descent polish
        if _time_remaining() > 5.0:
            h_polish = h_current.copy()
            lr = 0.01
            beta = 5.0
            
            for step in range(500):
                if _time_remaining() < 2.0:
                    break
                
                soft_val, subgrad = _soft_max_subgrad(h_polish, beta=beta)
                
                # Gradient descent step
                h_new = h_polish - lr * subgrad
                h_new = _project_to_simplex(h_new, target_sum)
                
                new_score = _score(h_new)
                if new_score < _score(h_polish):
                    h_polish = h_new
                    lr = min(lr * 1.02, 0.05)
                    if new_score < best_score:
                        best_score = new_score
                        best_h = h_polish.copy()
                else:
                    lr *= 0.9
                    if lr < 1e-6:
                        break
                
                # Increase beta for sharper max approximation
                if step % 100 == 0:
                    beta *= 2.0
                    beta = min(beta, 50.0)
    
    # Phase 3: Local search on best solution
    if _time_remaining() > 15.0:
        h_local = best_h.copy()
        local_score = best_score
        
        for step in range(500):
            if _time_remaining() < 3.0:
                break
            
            # Try small perturbations
            n_perturb = 10
            for _ in range(n_perturb):
                i = np.random.randint(n)
                delta = np.random.uniform(-0.1, 0.1)
                h_trial = h_local.copy()
                h_trial[i] += delta
                h_trial = _project_to_simplex(h_trial, target_sum)
                
                trial_score = _score(h_trial)
                if trial_score < local_score:
                    h_local = h_trial
                    local_score = trial_score
                    if local_score < best_score:
                        best_score = local_score
                        best_h = h_local.copy()
    
    # Final verification and return
    best_h = _project_to_simplex(best_h, target_sum)
    return best_h
# EVOLVE-BLOCK-END
