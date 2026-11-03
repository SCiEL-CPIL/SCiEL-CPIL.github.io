"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def _overlap(h, shift):
    """Compute max overlap of h with h shifted by 'shift' positions."""
    n = len(h)
    # h_shifted[i] = h[(i - shift) % n]
    # overlap = sum_i h[i] * h[(i - shift) % n]
    # We want to minimize max over shifts of sum_i h[i] * (1 - h[(i-shift)%n])
    # = n/2 - sum_i h[i]*h[(i-shift)%n]  (since sum h = n/2)
    # So minimizing max overlap of h with 1-h shifted is equivalent to
    # minimizing max_shift (n/2 - autocorr(shift)) = n/2 - min_shift autocorr(shift)
    # Wait, let me re-read the task.
    # "minimize the maximum overlap of h with shifted 1-h"
    # Overlap of h and (1-h) shifted by s: sum_i h[i] * (1 - h[(i-s) % n])
    # = sum_i h[i] - sum_i h[i]*h[(i-s)%n]
    # = n/2 - autocorr(s)
    # So we want to minimize max_s (n/2 - autocorr(s))
    # = n/2 - min_s autocorr(s)
    # To minimize this, we want to maximize min_s autocorr(s)
    # autocorr(0) = sum h[i]^2 which is fixed-ish but varies
    # For s=0: overlap = sum h[i]*(1-h[i]) = n/2 - sum h[i]^2
    # For other s: overlap = n/2 - autocorr(s)
    
    # Actually, let's just compute it directly for clarity
    h_shifted = np.roll(h, shift)
    return np.sum(h * (1 - h_shifted))

def _max_overlap(h):
    """Compute the maximum overlap over all non-zero shifts."""
    n = len(h)
    max_ov = 0.0
    for s in range(1, n):
        ov = _overlap(h, s)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _autocorr_min(h):
    """Return the minimum autocorrelation value over all non-zero shifts."""
    n = len(h)
    min_ac = np.inf
    for s in range(1, n):
        ac = np.sum(h * np.roll(h, s))
        if ac < min_ac:
            min_ac = ac
    return min_ac

def construct():
    deadline = time.time() + 110  # safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Start with a random binary-like initialization
    # Use a sequence that has good autocorrelation properties
    # Try to use a known good sequence or random search
    
    # Strategy: use simulated annealing or random restarts with local search
    # The objective is to maximize min autocorrelation over non-zero shifts
    # (equivalently, minimize max overlap with 1-h shifted)
    
    # Let's try a few different initializations and local search
    
    np.random.seed(42)
    
    # Try a periodic pattern first
    # For N=100, try patterns like alternating blocks
    candidates = []
    
    # Pattern 1: all 0.5
    candidates.append(np.full(N, 0.5))
    
    # Pattern 2: two-valued with period 2
    for p in [2, 3, 4, 5, 7, 10, 13, 17, 20, 25]:
        if N % p == 0:
            for val in [0.3, 0.4, 0.5, 0.6, 0.7]:
                pat = np.tile([val, 1-val] * (p//2) if p % 2 == 0 else [val] * (p//2) + [1-val] * (p//2 + 1), N // p)
                if len(pat) == N:
                    candidates.append(pat)
    
    # Pattern 3: random binary scaled
    for _ in range(20):
        b = np.random.randint(0, 2, N).astype(float)
        # Scale so sum is N/2
        if b.sum() > 0:
            # Adjust to make sum exactly N/2
            # This is tricky for binary. Let's use a different approach.
            # Use a mixture: some 0s, some 1s, and adjust a few to fractional
            pass
    
    # Better approach: use continuous variables in [0,1] with sum constraint
    # Use projected gradient or simple local search
    
    # Local search: perturb one or two elements, project to sum constraint
    def project_sum(h, target_sum):
        """Project h onto the set of vectors with sum = target_sum and elements in [0,1]."""
        # Simple projection: shift all values to achieve target sum
        # This is approximate; for better projection, use water-filling
        diff = target_sum - np.sum(h)
        # Distribute diff evenly
        h = h + diff / N
        # Clip to [0, 1]
        h = np.clip(h, 0, 1)
        # After clipping, sum may not be target_sum. Iterate a few times.
        for _ in range(10):
            diff = target_sum - np.sum(h)
            if abs(diff) < 1e-10:
                break
            h = h + diff / N
            h = np.clip(h, 0, 1)
        return h
    
    # Simulated annealing
    current_h = np.full(N, 0.5)
    current_score = _max_overlap(current_h)
    
    T = 0.1
    T_min = 1e-4
    cooling = 0.999
    
    steps = 0
    max_steps = 50000
    
    while steps < max_steps:
        if time.time() > deadline:
            break
        
        # Pick a random index and perturb
        i = np.random.randint(N)
        delta = np.random.uniform(-0.1, 0.1)
        new_h = current_h.copy()
        new_h[i] = np.clip(new_h[i] + delta, 0, 1)
        
        # Project to sum constraint
        new_h = project_sum(new_h, N / 2)
        
        new_score = _max_overlap(new_h)
        
        if new_score < current_score or np.random.random() < np.exp((current_score - new_score) / T):
            current_h = new_h
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
        
        T *= cooling
        steps += 1
    
    # Final polish: local search with smaller steps
    for _ in range(5000):
        if time.time() > deadline:
            break
        i = np.random.randint(N)
        delta = np.random.uniform(-0.01, 0.01)
        new_h = best_h.copy()
        new_h[i] = np.clip(new_h[i] + delta, 0, 1)
        new_h = project_sum(new_h, N / 2)
        new_score = _max_overlap(new_h)
        if new_score < best_score:
            best_h = new_h
            best_score = new_score
    
    return best_h
# EVOLVE-BLOCK-END
