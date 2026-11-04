import numpy as np
import time

N = 100
DELTA = 2.0 / N
SUM_TARGET = N / 2.0

def compute_overlaps_fft(h):
    """Compute overlaps O_k = sum_i h_i * (1 - h_{i+k}) * DELTA using FFT."""
    # O_k = sum_i h_i - sum_i h_i * h_{i+k}
    # sum_i h_i = SUM_TARGET
    # sum_i h_i * h_{i+k} is the circular autocorrelation of h
    # Use FFT for fast circular correlation
    H = np.fft.fft(h)
    corr = np.fft.ifft(H * np.conj(H)).real
    # corr[k] = sum_i h_i * h_{i+k}
    # O_k = (SUM_TARGET - corr[k]) * DELTA
    overlaps = (SUM_TARGET - corr) * DELTA
    return overlaps

def project_to_constraint(h_raw):
    """Project h_raw onto {h : 0 <= h_i <= 1, sum(h) = SUM_TARGET}."""
    lo = np.min(h_raw) - 1.0
    hi = np.max(h_raw) + 1.0
    
    for _ in range(50):
        mid = (lo + hi) / 2.0
        h_proj = np.clip(h_raw - mid, 0.0, 1.0)
        s = np.sum(h_proj)
        if s > SUM_TARGET:
            lo = mid
        else:
            hi = mid
            
    h_final = np.clip(h_raw - (lo + hi) / 2.0, 0.0, 1.0)
    diff = SUM_TARGET - np.sum(h_final)
    if abs(diff) > 1e-12:
        mask = (h_final > 1e-9) & (h_final < 1.0 - 1e-9)
        if np.any(mask):
            idx = np.where(mask)[0][0]
            h_final[idx] += diff
            h_final[idx] = np.clip(h_final[idx], 0.0, 1.0)
            
    return h_final

def compute_max_overlap_circular(h):
    overlaps = compute_overlaps_fft(h)
    return np.max(overlaps)

def try_binary_patterns():
    """Try various binary patterns with exactly N/2 ones."""
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(123)
    
    # Try random binary patterns
    for trial in range(500):
        indices = rng.choice(N, size=N//2, replace=False)
        h = np.zeros(N)
        h[indices] = 1.0
        score = compute_max_overlap_circular(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Try structured patterns: m-sequences, difference sets
    # For N=100, try patterns based on quadratic residues mod p
    # or other number-theoretic constructions
    
    return best_h, best_score

def simulated_annealing(h_init, time_limit):
    """Run simulated annealing to minimize max overlap."""
    h = h_init.copy()
    best_h = h.copy()
    best_score = compute_max_overlap_circular(h)
    
    T = 0.1  # Initial temperature
    cooling = 0.999
    
    overlaps = compute_overlaps_fft(h)
    max_ov = np.max(overlaps)
    
    while T > 0.001:
        # Pick a random position and perturb
        i = np.random.randint(N)
        delta = np.random.uniform(-0.1, 0.1)
        
        h_new = h.copy()
        h_new[i] = np.clip(h_new[i] + delta, 0.0, 1.0)
        
        # Project to constraint
        h_new = project_to_constraint(h_new)
        
        score_new = compute_max_overlap_circular(h_new)
        score_old = compute_max_overlap_circular(h)
        
        # Accept if better or with probability exp(-(score_new - score_old)/T)
        if score_new < score_old or np.random.rand() < np.exp(-(score_new - score_old) / T):
            h = h_new
            if score_new < best_score:
                best_score = score_new
                best_h = h.copy()
        
        T *= cooling
        
        if time.time() > time_limit:
            break
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    # Start with uniform
    best_h = np.full(N, 0.5)
    best_score = compute_max_overlap_circular(best_h)
    
    # Try binary patterns first
    binary_h, binary_score = try_binary_patterns()
    if binary_score < best_score:
        best_h = binary_h
        best_score = binary_score
    
    # Use simulated annealing from best starting point
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        sa_h, sa_score = simulated_annealing(best_h, deadline - 2.0)
        if sa_score < best_score:
            best_h = sa_h
            best_score = sa_score
    
    return best_h
