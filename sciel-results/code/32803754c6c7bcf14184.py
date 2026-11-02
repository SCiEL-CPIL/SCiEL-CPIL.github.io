"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 135.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_max_overlap(h):
    """
    Compute the maximum overlap of h with shifted 1-h.
    Overlap_k = sum_{i=0}^{N-1-k} h[i] * (1 - h[i+k])
    We want to minimize max(Overlap_k).
    """
    n = len(h)
    max_ov = 0.0
    # Use FFT for efficient cross-correlation calculation if N is large, 
    # but for N=100, direct dot product is fast enough and precise.
    # However, let's use a vectorized approach for speed in loops.
    # Overlap_k = sum(h[:N-k] * (1 - h[k:]))
    # Note: 1-h[k:] is the shifted complement.
    # Let's compute all overlaps.
    # For N=100, there are 99 overlaps.
    # Each overlap is a dot product of length N-k.
    for k in range(1, n):
        ov = np.dot(h[:n-k], 1.0 - h[k:])
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _objective_and_grad(params):
    """
    Objective: minimize max overlap.
    We use a soft-min approximation to make it differentiable.
    params: h (length N)
    """
    h = params
    n = len(h)
    
    # Calculate all overlaps
    overlaps = np.zeros(n-1)
    for k in range(1, n):
        overlaps[k-1] = np.dot(h[:n-k], 1.0 - h[k:])
    
    # Soft min of overlaps? No, we want to minimize the MAX.
    # So we use soft-max.
    beta = 50.0
    shifted = overlaps - np.max(overlaps)
    log_sum = np.log(np.sum(np.exp(beta * shifted)))
    soft_max = np.max(overlaps) + log_sum / beta
    
    # Gradient of soft-max
    exp_beta = np.exp(beta * shifted)
    weights = exp_beta / np.sum(exp_beta)
    
    grad_h = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1]
        if w < 1e-15:
            continue
        # d/dh_i of dot(h[:n-k], 1-h[k:])
        # Term i in h[:n-k] contributes h[i] * (1-h[i+k]) -> grad w.r.t h[i] is (1-h[i+k])
        # Term i in h[k:] contributes h[i-k] * (1-h[i]) -> grad w.r.t h[i] is -h[i-k]
        
        # Indices in h[:n-k] are 0 to n-k-1
        # Indices in h[k:] are k to n-1
        
        # For i in 0..n-k-1:
        grad_h[:n-k] += w * (1.0 - h[k:])
        # For i in k..n-1:
        grad_h[k:] += w * (-1.0) * h[:n-k]
        
    return soft_max, grad_h

def _project_to_simplex(h, target_sum):
    """Project h to satisfy sum(h) = target_sum and 0<=h<=1."""
    h = np.clip(h, 0, 1).copy()
    for _ in range(50):
        diff = target_sum - np.sum(h)
        if abs(diff) < 1e-12:
            break
        # Adjust elements not at bounds
        mask = (h > 1e-10) & (h < 1 - 1e-10)
        if np.sum(mask) == 0:
            # If stuck, relax bounds slightly or break
            break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0, 1)
    # Final exact adjustment
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-10:
        # Distribute to elements with most room
        rooms = np.where(h > 0.5, 1 - h, h)
        total_room = np.sum(rooms)
        if total_room > 1e-10:
            h += diff * rooms / total_room
            h = np.clip(h, 0, 1)
    return h

def _generate_m_sequence_segment(n, seed):
    """
    Generate a segment of an m-sequence (maximum length sequence).
    m-sequences have low autocorrelation properties.
    We generate a long m-sequence and take a window of length n.
    """
    rng = np.random.RandomState(seed)
    # Use a linear feedback shift register (LFSR) for m-sequence
    # For simplicity, we'll use a pseudo-random binary sequence with good balance
    # True m-sequences require specific polynomial. 
    # Let's use a known good primitive polynomial for degree m.
    # N=100, so we need m such that 2^m - 1 >= 100. m=7 gives 127.
    # Primitive polynomial for m=7: x^7 + x^4 + 1 (or similar)
    # Let's just use a high-quality random binary sequence with exact balance,
    # as constructing exact m-sequence segments for arbitrary N is complex.
    # However, the prompt suggests using algebraic properties.
    # Let's try to construct a sequence based on a quadratic residue or similar.
    
    # Alternative: Use a random permutation of 50 ones and 50 zeros.
    # This ensures the duty cycle is exactly 1/2.
    seq = np.zeros(n)
    ones = np.arange(n)
    rng.shuffle(ones)
    seq[ones[:n//2]] = 1.0
    return seq

def _local_search_binary(h, max_iters=200):
    """
    Local search on binary sequence (flip 1 to 0 or 0 to 1).
    Maintains sum = N/2.
    """
    n = len(h)
    current_h = h.copy()
    current_score = _compute_max_overlap(current_h)
    
    ones = np.where(current_h == 1.0)[0]
    zeros = np.where(current_h == 0.0)[0]
    
    for _ in range(max_iters):
        if _time_remaining() < 2.0:
            break
        improved = False
        # Try swapping a 1 with a 0
        if len(ones) == 0 or len(zeros) == 0:
            break
            
        # Randomly pick a few pairs to try
        for _ in range(50):
            i = np.random.choice(ones)
            j = np.random.choice(zeros)
            if i == j:
                continue
            test_h = current_h.copy()
            test_h[i] = 0.0
            test_h[j] = 1.0
            score = _compute_max_overlap(test_h)
            if score < current_score - 1e-12:
                current_h = test_h
                current_score = score
                # Update sets
                ones = np.where(current_h == 1.0)[0]
                zeros = np.where(current_h == 0.0)[0]
                improved = True
                break
        if not improved:
            # Try hill climbing with smaller steps? 
            # Or break if no improvement in this round
            pass
            
    return current_h, current_score

def construct():
    n = N
    target_sum = n / 2.0
    best_h = np.full(n, 0.5)
    best_score = _compute_max_overlap(best_h)
    
    np.random.seed(42)
    
    # Phase 1: Structural Binary Search
    # Generate diverse binary candidates
    candidates = []
    
    # 1. Random balanced binaries
    for seed in range(50):
        h_bin = _generate_m_sequence_segment(n, seed)
        candidates.append(h_bin)
        
    # 2. Patterns
    # Alternating
    h_alt = np.zeros(n)
    h_alt[::2] = 1.0
    candidates.append(h_alt)
    h_alt2 = np.zeros(n)
    h_alt2[1::2] = 1.0
    candidates.append(h_alt2)
    
    # Blocks
    for block_size in [2, 4, 5, 10, 20]:
        for _ in range(5):
            h_block = np.zeros(n)
            # Create random blocks
            idx = np.arange(n)
            rng = np.random.RandomState(np.random.randint(1000))
            rng.shuffle(idx)
            # Assign 1s in blocks
            num_ones = n // 2
            assigned = 0
            h_block = np.zeros(n)
            for i in range(0, n, block_size):
                end = min(i + block_size, n)
                if assigned + (end - i) <= num_ones:
                    h_block[i:end] = 1.0
                    assigned += (end - i)
                else:
                    # Partial fill
                    need = num_ones - assigned
                    h_block[i:i+need] = 1.0
                    assigned += need
                    break
            candidates.append(h_block)

    # Evaluate binary candidates
    for h_bin in candidates:
        if _time_remaining() < 5.0:
            break
        # Project to exact sum (should be close)
        h_bin = _project_to_simplex(h_bin, target_sum)
        score = _compute_max_overlap(h_bin)
        if score < best_score:
            best_score = score
            best_h = h_bin.copy()
            
        # Refine with local binary search
        h_refined, score_refined = _local_search_binary(h_bin, max_iters=50)
        h_refined = _project_to_simplex(h_refined, target_sum)
        if score_refined < best_score:
            best_score = score_refined
            best_h = h_refined.copy()

    # Phase 2: Continuous Optimization from Best Binary
    # The optimal solution might not be strictly binary.
    # Use L-BFGS-B to refine the best binary solution.
    
    if _time_remaining() > 10.0:
        h_init = best_h.copy()
        
        try:
            # Use a soft-max objective to handle the non-differentiability of max
            res = minimize(
                _objective_and_grad,
                h_init,
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, 1)] * n,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
            )
            
            h_new = _project_to_simplex(res.x, target_sum)
            score_new = _compute_max_overlap(h_new)
            
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass

    # Phase 3: Spectral Initialization and Optimization
    # Try a few spectral initializations if time permits
    if _time_remaining() > 30.0:
        # Initialize with frequency domain structure
        # Low frequency components often lead to smoother h, which might reduce max overlap?
        # Actually, high frequency (noise) reduces correlation.
        # Let's try a "white noise" initialization in frequency domain, 
        # but constrained to real-valued h.
        
        for trial in range(5):
            if _time_remaining() < 10.0:
                break
            # Random phase, flat magnitude
            freqs = np.fft.rfftfreq(n)
            # Zero out DC component (index 0) to ensure mean 0.5 later? 
            # No, we project to sum N/2.
            # Let's just generate random h and project.
            # Better: Use the spectral method to shape the autocorrelation.
            # We want to maximize min autocorrelation.
            # This is equivalent to minimizing the variance of the autocorrelation?
            
            # Let's just try more random restarts in continuous space
            h_rand = np.random.uniform(0, 1, n)
            h_rand = _project_to_simplex(h_rand, target_sum)
            
            try:
                res = minimize(
                    _objective_and_grad,
                    h_rand,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
                )
                h_new = _project_to_simplex(res.x, target_sum)
                score_new = _compute_max_overlap(h_new)
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new
            except Exception:
                pass

    # Final Polish
    # Ensure exact sum
    best_h = _project_to_simplex(best_h, target_sum)
    
    return best_h
# EVOLVE-BLOCK-END
