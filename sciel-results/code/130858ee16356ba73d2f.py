"""
Strategy: Structured Family Search over Geometric Progressions.
Instead of treating the set as an arbitrary binary vector and using stochastic local search (Simulated Annealing),
we restrict the search space to sets formed by unions of geometric progressions.
Specifically, we explore sets U = { floor(b * r^i) for i=0..k } and its variants.
This family is known to produce sets with large difference sets relative to sum sets (Sidon-like properties).
We perform a deterministic grid search over the base 'b' and ratio 'r', and then a local refinement
step that allows small perturbations to the generated set to fine-tune the score.
"""
import time
import math
import numpy as np

def compute_score_fast(U_list, max_val_hint=None):
    """
    Computes the lower bound exponent using FFT for sumset and bitsets for diffset.
    U_list: list of nonnegative integers
    Returns: score
    """
    if len(U_list) < 2:
        return 0.0
    
    U = sorted(set(U_list))
    if 0 not in U:
        # The problem requires 0. If it's missing in intermediate steps, we can't score it properly
        # in the context of the final answer, but for search we might allow it? 
        # No, the score formula is mathematical. But the task says U contains 0.
        # Let's just compute the score for the set as is, assuming the evaluator
        # will reject if 0 is missing, but for internal optimization we want to know
        # the potential score.
        pass
        
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    # 1. Compute Sumset |U+U| using FFT
    max_sum = 2 * maxU
    n_fft = 1
    while n_fft < max_sum + 1:
        n_fft *= 2
        
    A = np.zeros(n_fft, dtype=np.float64)
    for u in U:
        A[u] = 1.0
        
    A_fft = np.fft.rfft(A)
    C_fft = A_fft * A_fft
    C = np.fft.irfft(C_fft, n=n_fft)
    
    # Count non-zero entries in C[0..2*maxU]
    sum_set_size = np.sum(C[:max_sum + 1] > 0.5)
    
    # 2. Compute Difference Set |U-U|
    # For small sets (size < 100), direct set comprehension is fast.
    # For larger sets, we might need FFT for difference too, but let's check size first.
    # If |U| is small, N^2 is small.
    diff_set = set()
    for u1 in U:
        for u2 in U:
            diff_set.add(u1 - u2)
            
    size_diff = len(diff_set)
    size_sum = sum_set_size
    
    if size_sum == 0:
        return 0.0
        
    log_ratio = math.log(size_diff / size_sum)
    log_range = math.log(2 * maxU + 1)
    
    return 1.0 + log_ratio / log_range

def generate_geometric_set(b, r, max_val):
    """
    Generates a set of integers { floor(b * r^i) } that are <= max_val.
    Ensures 0 is included.
    """
    if b <= 0 or r <= 1:
        return [0]
    
    U = [0]
    val = b
    while val <= max_val:
        U.append(int(val))
        val *= r
    return sorted(set(U))

def local_refinement(U, max_val, time_limit):
    """
    Takes a starting set U and tries to improve it by adding or removing elements
    near the existing elements or at strategic positions.
    """
    current_U = sorted(U)
    current_score = compute_score_fast(current_U)
    best_U = list(current_U)
    best_score = current_score
    
    start_time = time.time()
    
    # Candidate elements to try:
    # 1. All integers from 1 to max_val (if max_val is small enough)
    # 2. If max_val is large, only try neighbors of existing elements and some random spots.
    
    if max_val <= 500:
        candidates = list(range(1, max_val + 1))
    else:
        # Generate a sparse set of candidates
        candidates = set()
        for u in current_U:
            for delta in range(-10, 11):
                if 0 < u + delta <= max_val:
                    candidates.add(u + delta)
        # Add some random candidates
        for _ in range(50):
            candidates.add(np.random.randint(1, max_val + 1))
        candidates = sorted(candidates)
    
    # We will iterate through candidates, trying to add or remove them.
    # To be efficient, we only consider adding if it's not in U, and removing if it is.
    # We'll do a few passes.
    
    for _ in range(3): # 3 passes of local search
        if time.time() > time_limit:
            break
            
        for c in candidates:
            if time.time() > time_limit:
                break
                
            if c in current_U:
                # Try removing c (but keep 0)
                if c == 0:
                    continue
                new_U = [u for u in current_U if u != c]
                if len(new_U) < 2:
                    continue
                new_score = compute_score_fast(new_U)
                if new_score > current_score:
                    current_U = new_U
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_U = list(current_U)
            else:
                # Try adding c
                new_U = current_U + [c]
                new_U.sort()
                new_score = compute_score_fast(new_U)
                if new_score > current_score:
                    current_U = new_U
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_U = list(current_U)
                        
    return best_U, best_score

def construct():
    """
    Entry point. Returns a list of nonnegative integers containing 0.
    Uses Structured Family Search over Geometric Progressions.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_overall_score = 0.0
    best_overall_U = [0, 1]
    
    # Define search space for geometric progressions
    # b: base (first non-zero element)
    # r: ratio (integer or float)
    # We limit max_val to keep computation fast.
    # Score ~ 1 + log(ratio)/log(2M+1).
    # If M is too large, the log denominator grows, reducing the gain.
    # If M is too small, the ratio might not be optimal.
    # Let's test M up to 200.
    
    M_max = 200
    
    # Grid search parameters
    bases = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
    ratios = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.5, 3.0, 4.0]
    
    for b in bases:
        for r in ratios:
            if time.time() > deadline - 10:
                break
                
            # Generate the base set
            U_gen = generate_geometric_set(b, r, M_max)
            
            # Ensure 0 is in the set
            if 0 not in U_gen:
                U_gen.append(0)
            U_gen.sort()
            
            # Local refinement
            U_refined, score_refined = local_refinement(U_gen, M_max, deadline - 5)
            
            if score_refined > best_overall_score:
                best_overall_score = score_refined
                best_overall_U = U_refined
    
    # Also try a few random dense sets with small max to see if we beat geometric
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        M_rand = np.random.randint(10, 50)
        # Random subset of [0, M_rand]
        size = np.random.randint(3, M_rand // 2 + 1)
        U_rand = [0]
        others = np.random.choice(range(1, M_rand + 1), size=size, replace=False)
        U_rand.extend(others)
        U_rand.sort()
        
        U_refined, score_refined = local_refinement(U_rand, M_rand, deadline - 5)
        
        if score_refined > best_overall_score:
            best_overall_score = score_refined
            best_overall_U = U_refined
            
    # Ensure 0 is in the final result
    if 0 not in best_overall_U:
        best_overall_U.append(0)
        
    return sorted(best_overall_U)

# EVOLVE-BLOCK-END
