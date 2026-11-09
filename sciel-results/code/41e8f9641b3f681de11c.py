"""Improved program using simulated annealing with FFT-based evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
from math import log, log2, exp
import time
import random

def compute_score_exact(U):
    """Compute the exact score for a given set U."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0
    
    denominator = log(2 * max_u + 1)
    if denominator <= 0:
        return 0.0
    
    return 1.0 + log(ratio) / denominator

def evaluate_fft(U, max_val):
    """
    Estimate |U+U| and |U-U| using FFT convolution.
    Returns (sumset_count, diffset_count).
    """
    n = len(U)
    if n == 0:
        return 0, 0
    
    # Create binary vector
    size = 2 * max_val + 1
    # We need to handle negative differences, so shift
    # For sumset: values range from 0 to 2*max_val
    # For diffset: values range from -max_val to max_val
    
    # Sumset via convolution
    f = np.zeros(2 * max_val + 1, dtype=np.float64)
    for u in U:
        if 0 <= u <= 2 * max_val:
            f[u] = 1.0
    
    # FFT convolution for sumset
    nfft = 1
    while nfft < 2 * (2 * max_val + 1):
        nfft *= 2
    F = np.fft.rfft(f, n=nfft)
    conv_sum = np.fft.irfft(F * F, n=nfft)
    # Sumset elements are at indices 0 to 2*max_val
    sumset_count = np.sum(conv_sum[:2 * max_val + 1] >= 0.5)
    
    # Difference set: U-U
    # This is equivalent to convolution of U with reversed U
    # Or we can compute it as: for each a in U, for each b in U, a-b
    # Using FFT: create vector g where g[u] = 1 if u in U
    # Then diffset count = number of distinct (a-b) values
    # We can use: autocorrelation-like approach
    # Actually, U-U = {a-b : a,b in U}
    # This is the same as the support of the cross-correlation
    
    # Let's use a different approach: 
    # Create vector h where h[u] = 1 if u in U (for u in [0, max_val])
    # Then the difference set is related to h convolved with reverse(h)
    
    h = np.zeros(max_val + 1, dtype=np.float64)
    for u in U:
        if 0 <= u <= max_val:
            h[u] = 1.0
    
    # Reverse h
    h_rev = h[::-1]
    
    nfft2 = 1
    while nfft2 < 2 * (max_val + 1):
        nfft2 *= 2
    
    H = np.fft.rfft(h, n=nfft2)
    H_rev = np.fft.rfft(h_rev, n=nfft2)
    conv_diff = np.fft.irfft(H * H_rev, n=nfft2)
    
    # The difference a-b where a,b in [0, max_val] ranges from -max_val to max_val
    # The convolution gives values at indices corresponding to a + (max_val - b)
    # So index i in conv_diff corresponds to a + max_val - b = i
    # Which means a - b = i - max_val
    # We want to count distinct (a-b) values, which is the number of indices where conv_diff > 0.5
    # The range of a-b is [-max_val, max_val], so we look at indices [0, 2*max_val]
    diffset_count = np.sum(conv_diff[:2 * max_val + 1] >= 0.5)
    
    return int(sumset_count), int(diffset_count)

def construct():
    """
    Construct a finite set U of nonnegative integers containing 0 to maximize
    the sumset/difference-set exponent lower bound.
    """
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Start with a reasonable initial set
    # The paper mentions constructions of size 2003 and 54265
    # Let's start with a set that has good structure
    
    # Initial strategy: use a set with elements that are powers or sparse
    # to make the difference set large relative to sumset
    
    # Start with a simple set and improve via simulated annealing
    # Key insight: we want |U-U| >> |U+U|
    # This happens when U is "spread out" in a way that differences are more varied than sums
    
    # A good starting point: use elements that grow geometrically
    # e.g., {0, 1, 2, 4, 8, ...} but this makes both sets large
    
    # Better: use a set where elements are chosen to create many distinct differences
    # but sums collide more often
    
    # Let's start with a moderate-size set
    initial_size = 50
    # Use a random initial set with 0
    max_val_init = 200
    U = [0]
    random.seed(42)
    candidates = list(range(1, max_val_init + 1))
    random.shuffle(candidates)
    U.extend(candidates[:initial_size - 1])
    U.sort()
    
    best_U = U[:]
    best_score = compute_score_exact(best_U)
    
    # Simulated annealing parameters
    T = 50.0  # Initial temperature
    cooling_rate = 0.9995
    T_min = 0.1
    
    # Try different initial sizes and structures
    # Also try structured sets
    
    # First, try some structured initial sets
    structured_sets = []
    
    # Powers of 2
    powers = [0]
    val = 1
    while len(powers) < 20:
        powers.append(val)
        val *= 2
    structured_sets.append(powers[:20])
    
    # Sparse random
    for _ in range(5):
        s = [0]
        rand_vals = random.sample(range(1, 500), 30)
        s.extend(rand_vals)
        s.sort()
        structured_sets.append(s)
    
    # Evaluate structured sets
    for s in structured_sets:
        sc = compute_score_exact(s)
        if sc > best_score:
            best_score = sc
            best_U = s[:]
    
    U = best_U[:]
    current_score = best_score
    
    # Simulated annealing
    step = 0
    while time.time() < deadline and T > T_min:
        step += 1
        
        # Generate a perturbation
        perturbation_type = random.random()
        
        if perturbation_type < 0.3:
            # Add an element
            max_u = max(U)
            new_val = random.randint(1, max_u * 2 + 10)
            if new_val not in U:
                U.append(new_val)
                U.sort()
        elif perturbation_type < 0.6:
            # Remove an element (keep 0)
            if len(U) > 1:
                idx = random.randint(1, len(U) - 1)
                U.pop(idx)
        elif perturbation_type < 0.8:
            # Shift an element
            if len(U) > 1:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                new_val = max(1, old_val + random.randint(-50, 50))
                if new_val not in U or new_val == old_val:
                    U[idx] = new_val
                    U.sort()
        else:
            # Swap two elements
            if len(U) > 2:
                i, j = random.sample(range(1, len(U)), 2)
                U[i], U[j] = U[j], U[i]
                U.sort()
        
        # Evaluate
        max_u = max(U)
        if max_u > 10000:
            # Too large, revert
            U = best_U[:]
            continue
        
        # Use FFT for quick evaluation
        sumset_count, diffset_count = evaluate_fft(U, max_u)
        
        if sumset_count == 0:
            new_score = 0.0
        else:
            ratio = diffset_count / sumset_count
            if ratio <= 1:
                new_score = 1.0
            else:
                denominator = log(2 * max_u + 1)
                new_score = 1.0 + log(ratio) / denominator
        
        # Accept or reject
        if new_score >= current_score:
            current_score = new_score
            U_new = U[:]
            if new_score > best_score:
                best_score = new_score
                best_U = U_new
        else:
            # Accept with probability exp((new_score - current_score) / T)
            delta = new_score - current_score
            if delta < 0:
                accept_prob = exp(delta / T)
                if random.random() < accept_prob:
                    current_score = new_score
                    U_new = U[:]
                else:
                    # Revert
                    U = best_U[:]
                    current_score = best_score
            else:
                U_new = U[:]
        
        # Cool down
        T *= cooling_rate
        
        # Periodically re-evaluate best exactly
        if step % 100 == 0:
            exact_best = compute_score_exact(best_U)
            if exact_best > best_score:
                best_score = exact_best
    
    # Final exact evaluation
    final_score = compute_score_exact(best_U)
    if final_score > best_score:
        best_score = final_score
    
    return best_U
# EVOLVE-BLOCK-END
