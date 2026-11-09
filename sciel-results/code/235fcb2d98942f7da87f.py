"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _overlap_at_shift(h, shift_idx):
    """Compute overlap integral I_s = sum_i h_i * h_{i+shift_idx} (with wraparound? No, standard shift).
    Wait, the problem says 'shifted 1-h'.
    Let h be defined on [0, 2].
    Shift by s.
    The overlap of h(x) and (1-h)(x-s) is integral over x where both are defined?
    Or is it on the circle?
    'Erdos minimum overlap' usually refers to the circle problem.
    If it's the circle, h is periodic.
    The task says 'bounded between 0 and 1 with integral 1' on [0,2].
    If it's the circle, the integral is 1 over length 2.
    Shift by s (distance).
    Overlap = ∫ h(x) * (1 - h(x-s)) dx?
    Actually, usually the overlap of two sets A and B is |A ∩ B|.
    Here h is a function? 'step heights'.
    'max overlap of h with shifted 1-h'.
    This likely means: for a shift s, calculate the integral of min(h(x), 1-h(x-s))?
    Or ∫ h(x) * (1-h(x-s)) dx?
    The seed program returns 0.5 everywhere.
    If h=0.5, 1-h=0.5. Overlap of 0.5 and 0.5 is 0.5 (if min) or 0.25 (if product)?
    Score 1.0 corresponds to constant 0.5.
    If the score is the upper bound on C5, and constant 0.5 gives 1.0.
    Let's assume the metric is max_s ∫_0^2 h(x) (1-h(x-s)) dx (circular).
    For h=0.5, integral is ∫ 0.5 * 0.5 = 0.25 * 2 = 0.5.
    Wait, if score 1.0 is the seed, and seed returns 0.5, maybe the score IS the value 0.5?
    Or maybe the score is normalized such that 0.5 -> 1.0?
    'score is the upper bound on C5'.
    'evaluator measures the raw value (lower is better)'.
    'search score is seed/value, so 1.0 equals the seed program'.
    If seed value is V_seed, and score is 1.0, then seed_value / V_seed = 1.
    So V_seed is the value produced by the seed.
    Seed returns 0.5.
    If the metric is max overlap, and for h=0.5 the overlap is 0.5 (area of intersection of two constant functions 0.5 and 0.5 is 0.5*Length=1? No.
    If overlap is ∫ min(h, 1-h_shifted), for h=0.5, min(0.5, 0.5)=0.5. Integral over length 2 is 1.0.
    If overlap is ∫ h * (1-h_shifted), for h=0.5, integral is 0.25*2 = 0.5.
    Which one is it?
    Erdos overlap problem usually involves sets.
    If h is a characteristic function (0 or 1), overlap is area of intersection.
    Here h is in [0,1].
    Let's look at the 'inspiration': 'minimizes boundary contributions to linear overlap'.
    'Linear overlap' suggests the interval [0,2] is not a circle, but the shift causes parts to fall off the edge?
    If it's linear:
    h is on [0,2].
    Shift 1-h by s.
    Overlap(s) = ∫ max(0, min(1, x+s)) ... intersection of supports?
    If s=0, overlap of h and 1-h is ∫ min(h, 1-h).
    If h=0.5, min(0.5, 0.5)=0.5. Integral is 1.0.
    If the seed value is 1.0, then score 1.0 makes sense if the raw value is 1.0.
    Let's assume the metric is M = max_{s \in [0,2]} \int_{0}^{2} \min(h(x), 1-h(x-s)) dx ?
    Or maybe it's the L1 distance?
    Let's try to find a function that minimizes the max overlap.
    If we make h a triangle: 0 at 0, 1 at 1, 0 at 2?
    Integral is 1.
    Let's code a general optimizer for the metric M(h) = max_s \int_0^2 \min(h(x), 1-h(x-s)) dx (with h(x)=0 outside [0,2]).
    Actually, if it's 'step heights', maybe it's a staircase function.
    The seed is N=100 steps of 0.5.
    I will implement a search over the vector h of length N.
    Constraints: 0 <= h_i <= 1, sum(h_i) * (2/N) = 1 => sum(h_i) = N/2.
    Objective: Minimize max_{k=0..N-1} \sum_{i} h_i * h_{i+k} ? No, that's autocorrelation.
    The term is overlap of h and 1-h.
    If h is a step function, the overlap integral for shift k steps (delta = k*2/N) is:
    Sum over i of (width) * min(h_i, 1-h_{i+k})?
    Wait, if x is in bin i, x-s is in bin i-k (roughly).
    Let's use a continuous approximation or just optimize the discrete vector.
    To be safe, I will optimize the discrete vector h directly using a local search (hill climbing) or gradient-free optimization.
    Given time limit 120s, I can do many iterations.
    """
    pass

def construct():
    # Start with 0.5
    h = np.full(N, 0.5)
    
    # We want to minimize the max overlap.
    # Let's define the overlap function.
    # Assumption: Circular shift or Linear?
    # 'bounded between 0 and 1 with integral 1' on [0,2].
    # If linear, shifting 1-h by s moves it.
    # Let's assume circular for now as it's common in such problems (Erdos on circle).
    # But 'linear overlap' in inspiration suggests linear.
    # Let's try linear: h is 0 outside [0,2].
    # Overlap(s) = ∫ min(h(x), 1-h(x-s)) dx.
    
    # For a discrete step function with step size dx = 2/N.
    # h is defined on bins 0..N-1.
    # Shift by k bins.
    # x in bin i corresponds to x in [i*dx, (i+1)*dx).
    # 1-h(x-s) at x depends on h at x-s.
    # This is complicated for linear shifts because of the edges.
    
    # Alternative interpretation:
    # The problem might be about the "Erdos Overlap Conjecture" which states
    # max_{shift} measure(A ∩ (A+shift)) >= ...
    # Here we have h and 1-h.
    # Let A be the set where h is high?
    
    # Let's stick to the discrete formulation:
    # Objective: min_h max_k O_k
    # O_k = sum_i w_i * min(h_i, 1 - h_{i+k mod N}) (circular)
    # or
    # O_k = sum_i w_i * min(h_i, 1 - h_{i+k}) (linear, with 0 padding)
    
    # Given the inspiration mentions "boundary contributions", linear is likely.
    # I will implement a local search on the linear overlap metric.
    
    dx = 2.0 / N
    
    def calc_overlap_linear(h):
        # h is length N
        # Calculate O_k for all k
        # To be efficient, we can use FFT?
        # min(a,b) = (a+b - |a-b|)/2
        # Sum min(h_i, 1-h_{i+k}) = 0.5 * (Sum h_i + Sum 1-h_{i+k} - Sum |h_i - (1-h_{i+k})|)
        # Sum h_i = N/2 * dx = 1.
        # Sum 1-h_{i+k} = N*dx - Sum h_{i+k} = 2 - 1 = 1.
        # So O_k = 0.5 * (1 + 1 - Sum |h_i - 1 + h_{i+k}|) = 1 - 0.5 * Sum |h_i + h_{i+k} - 1|.
        
        # We need to minimize max_k O_k.
        # Maximizing min_k (Sum |h_i + h_{i+k} - 1|) ? No, O_k = 1 - 0.5 L_k.
        # Minimize max O_k <=> Minimize max (1 - 0.5 L_k) <=> Maximize min L_k.
        # So we want to maximize the minimum L_k over all shifts k.
        # L_k = Sum_{i: i+k < N} |h_i + h_{i+k} - 1|  (for linear shift, only valid i)
        # Wait, for linear shift, if i+k >= N, the term 1-h_{i+k} is 1 (since h is 0 outside).
        # So for i such that i+k >= N, term is |h_i + 0 - 1| = |h_i - 1| = 1 - h_i.
        # For i such that i+k < N, term is |h_i + h_{i+k} - 1|.
        
        # Let's compute L_k for all k.
        Ls = np.zeros(N)
        for k in range(N):
            # Split indices
            # i from 0 to N-1
            # i+k < N => i < N-k
            # i >= N-k => i+k >= N
            
            if k == 0:
                # i+k = i
                # Term: |h_i + h_i - 1| = |2h_i - 1|
                Ls[k] = np.sum(np.abs(2*h - 1))
            else:
                # i in [0, N-k-1]
                idx_valid = np.arange(N-k)
                term_valid = np.abs(h[idx_valid] + h[idx_valid+k] - 1)
                
                # i in [N-k, N-1]
                idx_out = np.arange(N-k, N)
                term_out = np.abs(h[idx_out] - 1) # 1 - h_i
                
                Ls[k] = np.sum(term_valid) + np.sum(term_out)
                
        # O_k = 1 - 0.5 * L_k
        # Max O_k = 1 - 0.5 * Min L_k
        # We want to Minimize Max O_k.
        # So we want to Maximize Min L_k.
        return np.max(1 - 0.5 * Ls)

    # Start value
    current_score = calc_overlap_linear(h)
    
    # Local search
    # Randomly perturb h, if score improves, keep.
    # Ensure sum constraint: if we change h_i by delta, change h_j by -delta.
    # Or just project back to sum=N/2 after perturbation.
    
    t_start = time.time()
    timeout = 110.0
    step_size = 0.1
    attempts = 0
    
    best_h = h.copy()
    best_score = current_score
    
    while time.time() - t_start < timeout:
        # Pick two random indices
        i, j = np.random.randint(0, N, 2)
        if i == j: continue
        
        # Perturb
        delta = np.random.uniform(-step_size, step_size)
        h_new = h.copy()
        h_new[i] += delta
        h_new[j] -= delta
        
        # Clip to [0,1]
        # This might break sum constraint.
        # Better: swap mass.
        # Move mass from i to j.
        mass = np.random.uniform(-0.1, 0.1)
        h_new = h.copy()
        # h_new[i] -= mass
        # h_new[j] += mass
        # This preserves sum.
        # Check bounds
        if h[i] - mass < 0 or h[i] - mass > 1: continue
        if h[j] + mass < 0 or h[j] + mass > 1: continue
        
        h_new[i] -= mass
        h_new[j] += mass
        
        new_score = calc_overlap_linear(h_new)
        if new_score < best_score - 1e-6:
            h = h_new
            best_h = h_new.copy()
            best_score = new_score
            step_size = max(0.01, step_size * 1.05) # Increase step if improving? Or decrease?
            # Usually decrease step size to converge.
            step_size = max(0.001, step_size * 0.99)
        else:
            step_size = max(0.001, step_size * 0.98)
            
        attempts += 1
        if attempts % 10000 == 0:
            # Optional: restart or random jump
            pass

    return best_h
# EVOLVE-BLOCK-END
