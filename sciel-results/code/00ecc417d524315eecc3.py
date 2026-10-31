"""Rewritten program for Erdos minimum overlap task."""
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 100

def _max_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h over all shifts."""
    # h is a sequence of length N with values in [0,1]
    # We consider circular shifts? Or linear shifts?
    # The task says "shifted 1-h". Let's assume circular convolution for maximum overlap.
    # Overlap at shift k: sum_i h[i] * (1 - h[(i-k) % N])
    # Actually, let's think about this more carefully.
    # "maximum overlap of h with shifted 1-h"
    # This likely means: max over all shifts s of sum_i h[i] * (1 - h[(i+s) mod N])
    # Or perhaps linear shifts? Given N=100 and step heights, circular is typical for such problems.
    # Let's use circular shifts.
    
    # Compute (1-h)
    one_minus_h = 1.0 - h
    
    # For each shift s, compute sum_i h[i] * one_minus_h[(i+s) % N]
    # We can use FFT for efficiency, but N=100 is small, so direct computation is fine.
    # However, for optimization, we need this to be fast. Let's use a loop.
    
    max_ov = 0.0
    for s in range(N):
        # shifted one_minus_h: one_minus_h[(i+s) % N]
        shifted = np.roll(one_minus_h, s)
        ov = np.dot(h, shifted)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _objective(params):
    """
    params: [block_start, transition_width, left_val, right_val]
    - block_start: start index of transition (0 to N)
    - transition_width: width of transition region W (1 to N)
    - left_val: value on the left side (0 to 1)
    - right_val: value on the right side (0 to 1)
    
    We construct h as a piecewise constant function with a smooth transition.
    h[i] = left_val for i < block_start
    h[i] = right_val for i >= block_start + transition_width
    In between, we use a sigmoidal transition.
    
    But wait, the sum must be exactly N/2 = 50.
    So we need to ensure sum(h) = 50.
    
    Let's parameterize differently:
    - We have two blocks: left block of value a, right block of value b.
    - The transition region has width W and uses a sigmoid.
    - We need to adjust to ensure the sum is exactly 50.
    
    Alternative: 
    Let h be constructed from a binary-like pattern with a smooth transition.
    Let's say for i in [0, N), h[i] = a if i < start, b if i >= start, with a smooth transition of width W around 'start'.
    
    To ensure sum = 50, we can scale or adjust. But scaling might violate [0,1] bounds.
    
    Better approach: 
    Use a parametric form where the sum constraint is automatically satisfied.
    For example, let h[i] = 0.5 + 0.5 * f(i) where f is a step-like function with mean 0.
    Then sum(h) = N * 0.5 + 0.5 * sum(f) = 50 + 0.5 * sum(f).
    For sum(h) = 50, we need sum(f) = 0.
    
    Let f be a function that is -1 on the left, +1 on the right, with a smooth transition.
    Then h = 0.5 + 0.5 * f, so h is in [0,1].
    Sum(h) = 50 + 0.5 * sum(f). We need sum(f) = 0.
    
    So we need to design f such that sum(f) = 0.
    Let f[i] = -1 for i < start - W/2, +1 for i > start + W/2, and a sigmoid in between.
    We can adjust the transition width and position to make sum(f) = 0.
    
    Actually, let's just use a simpler approach:
    Define h[i] = 0.5 + 0.5 * s(i) where s(i) is a step function with a smooth transition.
    We'll parameterize s by its transition center c and width w.
    s(i) = -1 if i < c - w/2
    s(i) = +1 if i > c + w/2
    s(i) = sigmoid((i - c) / (w/2)) - 1 if c - w/2 <= i <= c + w/2
    Wait, sigmoid goes from 0 to 1. Let's use:
    s(i) = 2 * sigmoid((i - c) / (w/4)) - 1
    This goes from -1 to +1.
    
    But we need sum(s) = 0 for sum(h) = 50.
    This is a constraint. We can use it to determine one parameter, or we can add a penalty.
    
    Let's use a penalty approach: if sum(h) != 50, add a large penalty.
    But the problem says "sum is exactly len/2", so we must ensure this.
    
    Alternative: 
    After constructing h with the parametric form, normalize it to have sum exactly 50, while keeping values in [0,1].
    But normalization might push values out of [0,1].
    
    Let's try a different parameterization:
    Let h be a sequence where h[i] = a for i in [0, k) and h[i] = b for i in [k, N), with a smooth transition of width W around k.
    We have 4 parameters: a, b, k, W.
    Constraint: sum(h) = 50.
    We can solve for one parameter in terms of the others, or use a penalty.
    
    Given the complexity, let's use a penalty method in the objective function.
    """
    c, w, a, b = params
    # c: transition center (0 to N)
    # w: transition width (1 to N)
    # a: left value (0 to 1)
    # b: right value (0 to 1)
    
    i = np.arange(N)
    # Compute s(i) = 2 * sigmoid((i - c) / (w/4)) - 1
    # But if w is very small, the sigmoid is very steep.
    # Let's use a smoother transition: s(i) = tanh((i - c) / (w/2))
    # tanh goes from -1 to 1.
    
    # Avoid division by zero
    if w < 1e-6:
        w = 1e-6
    
    s = np.tanh((i - c) / (w / 2.0))
    
    # h = a + (b - a) * (s + 1) / 2
    # When s = -1, h = a. When s = 1, h = b.
    h = a + (b - a) * (s + 1.0) / 2.0
    
    # Clamp to [0, 1]
    h = np.clip(h, 0.0, 1.0)
    
    # Check sum constraint
    current_sum = np.sum(h)
    target_sum = N / 2.0
    
    # Penalty for sum constraint
    sum_penalty = 1000.0 * (current_sum - target_sum) ** 2
    
    # Compute max overlap
    ov = _max_overlap(h)
    
    return ov + sum_penalty

def construct():
    """
    Construct the sequence of step heights.
    """
    # Define bounds for parameters
    # c: transition center, 0 to N
    # w: transition width, 1 to N
    # a: left value, 0 to 1
    # b: right value, 0 to 1
    bounds = [(0, N), (1, N), (0, 1), (0, 1)]
    
    # Use differential evolution for global search
    result = differential_evolution(_objective, bounds, seed=42, maxiter=100, tol=1e-6, polish=False)
    
    best_params = result.x
    c, w, a, b = best_params
    
    i = np.arange(N)
    if w < 1e-6:
        w = 1e-6
    
    s = np.tanh((i - c) / (w / 2.0))
    h = a + (b - a) * (s + 1.0) / 2.0
    h = np.clip(h, 0.0, 1.0)
    
    # Ensure sum is exactly N/2
    # Adjust h to have sum exactly N/2 while keeping values in [0,1]
    current_sum = np.sum(h)
    diff = N / 2.0 - current_sum
    
    # Add diff/N to each element
    h = h + diff / N
    
    # Clamp again
    h = np.clip(h, 0.0, 1.0)
    
    # After clamping, the sum might not be exactly N/2.
    # We need to do a more careful adjustment.
    # Let's use a iterative approach: adjust the largest elements if sum is too low, or the smallest if sum is too high.
    
    for _ in range(100):
        current_sum = np.sum(h)
        diff = N / 2.0 - current_sum
        if abs(diff) < 1e-10:
            break
        
        if diff > 0:
            # Need to increase sum. Increase elements that are < 1.
            mask = h < 1.0
            if not np.any(mask):
                break
            # Increase the smallest elements first
            indices = np.where(mask)[0]
            # Sort indices by h value
            sorted_indices = indices[np.argsort(h[indices])]
            # Add a small amount to the smallest elements
            step = diff / len(sorted_indices)
            for idx in sorted_indices:
                h[idx] = min(1.0, h[idx] + step)
        else:
            # Need to decrease sum. Decrease elements that are > 0.
            mask = h > 0.0
            if not np.any(mask):
                break
            indices = np.where(mask)[0]
            sorted_indices = indices[np.argsort(-h[indices])]
            step = diff / len(sorted_indices)
            for idx in sorted_indices:
                h[idx] = max(0.0, h[idx] + step)
    
    return h

# EVOLVE-BLOCK-START
# (The construct function is defined above, but we need to keep the interface)
# The task says "Keep the function names and signatures of the seed program."
# The seed program has construct() -> sequence of step heights.
# So we just need to make sure construct() is defined and returns the right thing.
# EVOLVE-BLOCK-END
