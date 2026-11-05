"""Program for low discrepancy on all arithmetic progressions.
Uses local search with efficient discrepancy evaluation.
"""
import numpy as np
import time

N = 256

def _compute_all_ap_sums(signs):
    """Compute sums of signs over all arithmetic progressions in [0, N).
    Returns a list of (sum_value, indices_tuple) but for efficiency we just
    track the max absolute sum.
    
    Actually, for local search we need to know which APs are violated and
    how flipping a sign changes things. Let's precompute for each position i,
    which APs contain i and what their current sum is.
    
    Better approach: precompute all APs as lists of indices. Then maintain
    an array of sums. When we flip sign at position i, we update all APs
    that contain i.
    """
    pass

def _max_ap_discrepancy(signs):
    """Compute the maximum absolute signed sum over all arithmetic progressions.
    signs is a numpy array of length N with values ±1.
    """
    max_abs = 0
    # For each start s (0-indexed) and step d, compute all AP sums
    # An AP is defined by start s, step d, and length L >= 1
    # The AP contains positions s, s+d, s+2d, ... < N
    # For each such AP, we need the max |partial sum| over all prefixes? 
    # No - the task says "maximum absolute signed sum on ANY arithmetic progression"
    # This means for each AP (defined by start, step, and length), we compute the sum
    # of signs over that specific AP, and we want the max |sum| over all such APs.
    
    # Wait, re-reading: "maximum absolute signed sum on ANY arithmetic progression 
    # contained in [1,256], any start, positive step and length including one"
    # So an arithmetic progression is specified by (start, step, length). The sum is
    # over exactly those 'length' terms. We want max |sum| over all valid (start, step, length).
    
    for s in range(N):
        for d in range(1, N - s):
            # AP: s, s+d, s+2d, ... as long as < N
            # For each length L >= 1, compute sum
            total = 0
            for k in range(s, N, d):
                total += signs[k]
                abs_total = abs(total)
                if abs_total > max_abs:
                    max_abs = abs_total
    return max_abs

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    np.random.seed(42)
    
    # Start with a random assignment
    signs = np.random.choice([-1, 1], size=N)
    best_signs = signs.copy()
    best_disc = _max_ap_discrepancy(signs)
    
    # Local search: try flipping individual signs
    # Precompute for each position i, the list of (s, d) pairs such that i is in the AP starting at s with step d
    # Actually, for efficiency, let's just do simple hill climbing
    
    iteration = 0
    while time.time() < deadline:
        # Try each position flip
        improved = False
        for i in range(N):
            old_val = signs[i]
            signs[i] = -old_val
            disc = _max_ap_discrepancy(signs)
            if disc < best_disc:
                best_disc = disc
                best_signs = signs.copy()
                improved = True
            else:
                signs[i] = old_val
        
        if not improved:
            # Try random restarts or perturbations
            # Perturb a few random positions
            num_perturb = np.random.randint(1, 5)
            for _ in range(num_perturb):
                i = np.random.randint(0, N)
                signs[i] = -signs[i]
            # Continue searching from perturbed state
            # Recompute best_disc for current signs
            current_disc = _max_ap_discrepancy(signs)
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs.copy()
        
        iteration += 1
        if iteration % 50 == 0:
            # Check time
            if time.time() > deadline:
                break
    
    return [int(x) for x in best_signs]
