"""Low discrepancy on all arithmetic progressions - Simulated Annealing with Vectorized Evaluation."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    signs: numpy array of +1/-1 indexed 0..n-1.
    Returns the max absolute sum.
    """
    n = len(signs)
    s = signs  # assume int16 array
    
    max_abs = 0
    
    # Step 1: singletons are always 1, so max_abs >= 1
    # We need to check all steps d >= 1
    # For efficiency, we can limit d. If d is large, the APs are short.
    # Max length of AP with step d is roughly n/d.
    # If n/d < 2, i.e., d > n/2, the APs have length 1. Sum is 1.
    # So we only need to check d up to n//2.
    
    # Actually, let's be more precise.
    # For a fixed d, the APs are s[i], s[i+d], s[i+2d], ...
    # We want max |prefix_sum| for each such sequence.
    
    # Optimization:
    # For d=1, it's just the max absolute prefix sum of the whole array.
    # For d>1, we can process residues modulo d.
    
    # Let's iterate d from 1 to n//2.
    # For each d, we have d residue classes (starts 0..d-1).
    # For each start i, we extract the subsequence s[i], s[i+d], ...
    # Compute prefix sums and track max abs.
    
    # To vectorize:
    # For a fixed d, we can create an array of shape (d, max_len) ?
    # But lengths vary.
    # Alternative: for each d, loop i from 0 to d-1.
    # Inside, use numpy slicing s[i::d] to get the subsequence.
    # Compute cumsum and max abs.
    # This is O(n/d) per i, so O(n) per d. Total O(n^2) which is 65536 for n=256.
    # This is very fast in numpy if we minimize overhead.
    
    # Let's try this approach.
    for d in range(1, n // 2 + 1):
        # For each start i in [0, d)
        # We can vectorize the "max abs prefix sum" calculation?
        # Not easily because lengths differ.
        # But we can loop i. d is at most 128.
        # So at most 128 * (n/d) operations per d.
        
        # Actually, we can do better.
        # For a fixed d, consider the matrix M where M[i, k] = s[i + k*d].
        # This is a ragged array.
        
        # Let's just loop i for now and see if it's fast enough.
        # n=256, d up to 128.
        # Total iterations of inner loop: sum_{d=1}^{128} d * (n/d) = 128 * 256 = 32768.
        # Each iteration does a numpy slice and cumsum.
        # This should be fast.
        
        for i in range(d):
            # Extract subsequence
            sub = s[i::d]
            if len(sub) <= 1:
                continue # max abs is 1, already covered
            cs = np.cumsum(sub)
            m = np.max(np.abs(cs))
            if m > max_abs:
                max_abs = m
                # Early exit? If max_abs is already high, maybe.
                # But we need the exact value for the SA cost.
    
    return int(max_abs)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 256
    rng = np.random.RandomState(42)
    
    # Initialize with random signs
    signs = rng.choice([-1, 1], size=n).astype(np.int16)
    best_signs = signs.copy()
    best_score = _max_ap_discrepancy(best_signs)
    
    # Simulated Annealing
    # Temperature schedule: linear or exponential cooling
    T_start = 10.0
    T_end = 0.1
    # Number of steps: we don't know exactly, so we run until deadline.
    
    # To speed up, we can pre-compute which indices affect which APs?
    # That's complex. Let's stick to full evaluation for now, but optimize the eval.
    
    # Optimization: Cache the discrepancy? No, it changes every step.
    # Optimization: Use a faster discrepancy calculator?
    # The current one is O(n^2) with small constants.
    # 32k numpy operations. Each is small.
    # Let's measure how fast it is.
    
    # Let's try to improve the discrepancy calculation.
    # Instead of looping i for each d, can we vectorize over i?
    # For a fixed d, we want max over i of max_abs_prefix_sum(s[i::d]).
    # This is hard to vectorize directly due to variable lengths.
    
    # However, we can note that for large d, the lengths are small.
    # For d > n/4, length is at most 4.
    # We can handle small lengths with explicit formulas?
    # Length 1: 1
    # Length 2: max(|s1|, |s1+s2|) = max(1, |s1+s2|). s1,s2 are +/-1.
    #   If s1==s2, sum=+/-2, abs=2. If s1!=s2, sum=0, abs=0. Max is 2.
    #   Wait, max abs prefix sum.
    #   Prefix sums: s1, s1+s2.
    #   |s1|=1. |s1+s2| is 0 or 2.
    #   So max is 2 if s1==s2, 1 if s1!=s2.
    # Length 3: s1, s1+s2, s1+s2+s3.
    #   Max abs.
    
    # This suggests that for large d, the contribution is small (<= length).
    # The bottleneck is small d (1, 2, 3, ...).
    
    # Let's optimize the loop for small d.
    # For d=1, it's just max abs prefix sum of s.
    # For d=2, it's max of max_abs_prefix(s[0::2]) and max_abs_prefix(s[1::2]).
    
    # Let's rewrite _max_ap_discrepancy to be faster.
    
    def fast_max_ap_discrepancy(s):
        n = len(s)
        max_abs = 0
        
        # d=1
        cs = np.cumsum(s)
        max_abs = max(max_abs, int(np.max(np.abs(cs))))
        
        # d from 2 to n//2
        for d in range(2, n // 2 + 1):
            # We can break early if d > current max_abs?
            # No, because a long AP can have large sum.
            # But max possible sum for step d is n/d.
            # If n/d <= max_abs, then this d cannot improve.
            if n // d <= max_abs:
                break
            
            for i in range(d):
                sub = s[i::d]
                if len(sub) <= 1:
                    continue
                cs_sub = np.cumsum(sub)
                m = int(np.max(np.abs(cs_sub)))
                if m > max_abs:
                    max_abs = m
                    if n // d <= max_abs:
                        break # No further i for this d can improve, and larger d will have even shorter max length
            
        return max_abs

    best_score = fast_max_ap_discrepancy(best_signs)
    
    # SA Loop
    step = 0
    while time.time() < deadline:
        step += 1
        # Update temperature
        # Simple linear cooling over time? Or over steps?
        # Since we don't know total steps, use time-based or fixed steps.
        # Let's use a fixed number of steps per "epoch" and cool down.
        # Or just exponential decay based on step count, assuming ~10k steps.
        # Let's assume we can do ~5000 steps in 85s.
        # T = T_start * (T_end/T_start) ** (step / max_steps)
        max_steps = 5000
        T = T_start * (T_end / T_start) ** (step / max_steps)
        
        # Flip a random index
        idx = rng.randint(0, n - 1)
        old_val = signs[idx]
        signs[idx] = -old_val
        
        new_score = fast_max_ap_discrepancy(signs)
        
        delta = new_score - best_score
        
        if delta < 0 or (rng.random() < np.exp(-delta / T)):
            # Accept
            if new_score < best_score:
                best_score = new_score
                best_signs = signs.copy()
        else:
            # Reject
            signs[idx] = old_val
            
        # Check time
        if time.time() > deadline - 1.0:
            break
    
    return best_signs.tolist()
# EVOLVE-BLOCK-END
