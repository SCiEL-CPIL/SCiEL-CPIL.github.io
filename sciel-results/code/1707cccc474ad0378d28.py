"""
Strategy: Two-Block Disjoint Rectangles
Represents the step function as a sum of two disjoint rectangular blocks.
This reduces the 600-dimensional optimization problem to a 4-parameter
optimization problem (width1, pos1, width2, pos2), which can be solved
globally and quickly using a grid search or local refinement.
The autoconvolution of two disjoint rectangles is a piecewise linear function
(trapezoids) whose maximum can be evaluated analytically or via fast sampling.
"""
import time
import numpy as np
from scipy import optimize

N = 600

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

def _construct_from_blocks(w1, p1, w2, p2):
    """
    Construct a step function from two disjoint rectangular blocks.
    w1, w2: widths (number of intervals)
    p1, p2: start positions (integer indices)
    The blocks must be disjoint and within [0, N-1].
    Returns an array of length N with 1s in the blocks and 0s elsewhere.
    """
    h = np.zeros(N)
    
    # Ensure integers
    w1 = int(round(w1))
    p1 = int(round(p1))
    w2 = int(round(w2))
    p2 = int(round(p2))
    
    # Bounds check
    if w1 < 1 or w2 < 1:
        return h
    if p1 < 0 or p1 + w1 > N:
        return h
    if p2 < 0 or p2 + w2 > N:
        return h
        
    # Disjoint check
    # Block 1: [p1, p1+w1)
    # Block 2: [p2, p2+w2)
    # They overlap if p1 < p2+w2 and p2 < p1+w1
    if p1 < p2 + w2 and p2 < p1 + w1:
        return h
        
    h[p1:p1+w1] = 1.0
    h[p2:p2+w2] = 1.0
    
    return h

def _score_blocks(w1, p1, w2, p2):
    """
    Fast scoring for block parameters.
    Uses the analytic property or fast convolution.
    Since N=600 is small, np.convolve is fast enough for many evaluations.
    """
    h = _construct_from_blocks(w1, p1, w2, p2)
    if np.sum(h) == 0:
        return 1e10
    return compute_score(h)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def check_time():
        return time.time() > deadline

    best_h = np.ones(N)
    best_score = compute_score(best_h)

    # Strategy: Search for two disjoint blocks.
    # Parameters: w1, p1, w2, p2
    # Constraints: w1, w2 >= 1, p1, p2 >= 0, p1+w1 <= N, p2+w2 <= N, disjoint.
    
    # We can use scipy.optimize.minimize on the continuous relaxation,
    # then round to integers.
    # Objective: _score_blocks(w1, p1, w2, p2)
    
    # However, _score_blocks is discrete and non-smooth.
    # A grid search is more robust for discrete parameters.
    # N=600.
    # w1, w2 in [1, N]
    # p1, p2 in [0, N]
    
    # To make it faster, we can exploit symmetry.
    # Assume w1 >= w2.
    # Assume p1 <= p2 (sorted by position).
    
    # Let's try a coarse grid search first.
    # Step size for widths: 5
    # Step size for positions: 5
    
    # Even better: Use the analytic structure.
    # For two disjoint blocks of widths w1, w2, separated by gap g.
    # The autoconvolution has peaks at the self-correlation of each block
    # and cross-correlation terms.
    # The max of the autoconvolution is likely at the center of the larger block
    # or in the overlap region if they were close (but they are disjoint).
    # Actually, if they are disjoint, the cross terms are shifted.
    # The max value is max( w1^2, w2^2, cross_max ).
    # Wait, the autoconvolution of a block of width w is a triangle with peak w^2.
    # If we have two blocks, the autoconvolution is the sum of:
    # 1. Autoconv of block 1 (peak w1^2)
    # 2. Autoconv of block 2 (peak w2^2)
    # 3. Cross terms (block 1 * block 2 and block 2 * block 1)
    # The cross terms are shifted by the distance between blocks.
    # If the blocks are far apart, the cross terms don't overlap with the self-peaks.
    # If they are close, they might add up.
    
    # The objective is max(conv) / (w1+w2)^2.
    
    # Let's perform a randomized local search on the discrete parameters.
    
    # Initial guess: Two equal blocks separated by some gap.
    # Let's try a few random configurations and refine.
    
    np.random.seed(42)
    
    def objective(params):
        if check_time():
            return 1e10
        w1, p1, w2, p2 = params
        # Enforce ordering p1 <= p2
        if p1 > p2:
            w1, p1, w2, p2 = w2, p2, w1, p1
        return _score_blocks(w1, p1, w2, p2)
    
    # Use differential_evolution for global search on the discrete problem?
    # DE works on continuous, but we can round in the objective.
    # Bounds:
    # w1: [1, N]
    # p1: [0, N-1]
    # w2: [1, N]
    # p2: [0, N-1]
    
    # To handle the disjoint constraint and ordering, we can penalize invalid configs.
    
    def objective_de(x):
        if check_time():
            return 1e10
        w1, p1, w2, p2 = x
        # Round
        w1 = int(round(w1))
        p1 = int(round(p1))
        w2 = int(round(w2))
        p2 = int(round(p2))
        
        # Clamp
        w1 = np.clip(w1, 1, N)
        w2 = np.clip(w2, 1, N)
        p1 = np.clip(p1, 0, N - w1)
        p2 = np.clip(p2, 0, N - w2)
        
        # Check disjoint
        if p1 < p2 + w2 and p2 < p1 + w1:
            return 1e10
            
        return _score_blocks(w1, p1, w2, p2)
    
    # Run DE for a short time
    try:
        # DE is stochastic, so set seed
        res = optimize.differential_evolution(
            objective_de,
            bounds=[(1, N), (0, N), (1, N), (0, N)],
            maxiter=100,
            popsize=10,
            tol=1e-4,
            seed=42,
            polish=False # We will polish manually
        )
        
        if not check_time():
            w1, p1, w2, p2 = res.x
            w1 = int(round(w1))
            p1 = int(round(p1))
            w2 = int(round(w2))
            p2 = int(round(p2))
            
            # Ensure validity
            w1 = np.clip(w1, 1, N)
            w2 = np.clip(w2, 1, N)
            p1 = np.clip(p1, 0, N - w1)
            p2 = np.clip(p2, 0, N - w2)
            
            if not (p1 < p2 + w2 and p2 < p1 + w1):
                h_cand = _construct_from_blocks(w1, p1, w2, p2)
                score_cand = compute_score(h_cand)
                if score_cand < best_score:
                    best_score = score_cand
                    best_h = h_cand
    except Exception:
        pass

    # Local refinement of the best candidate found by DE or a good guess
    if not check_time():
        # Extract best parameters from best_h?
        # It's hard to extract parameters from an arbitrary h.
        # Let's just run a local search around the best DE solution.
        
        # If DE didn't find anything better than uniform, let's try a heuristic.
        # Uniform score is 1/N.
        # Two blocks of width N/2 separated by 0 (touching) is one block of width N. Score 1/N.
        # Two blocks of width N/4 separated by N/2?
        # Let's try a grid of "balanced" configurations.
        
        # We will do a fine grid search on the "gap" and "width ratio".
        # Assume total length L = w1 + w2 is fixed? No, L can vary.
        # But max(conv) is dominated by max(w1, w2)^2.
        # Normalizer is (w1+w2)^2.
        # Ratio r = w1/w2.
        # Score ~ max(1, r^2) / (1+r)^2.
        # This is minimized when r=1 (equal widths).
        # So w1 ~= w2.
        
        # Let w1 = w2 = w.
        # Score ~ w^2 / (2w)^2 = 1/4 = 0.25.
        # Uniform score is 1/600 ~ 0.0016.
        # So two blocks are much worse than uniform!
        
        # Wait. The seed program scores 1.0, which corresponds to the uniform distribution?
        # Let's check compute_score(np.ones(N)).
        # conv = triangle with peak N.
        # s = N.
        # score = N / N^2 = 1/N.
        # For N=600, score = 1/600 = 0.00166.
        
        # The task says "score is the upper bound on C1".
        # "search score is seed/value, so 1.0 equals the seed program".
        # If seed is uniform, value = 1/600.
        # If I find a value < 1/600, my score > 1.0.
        
        # My two-block strategy yields score ~ 0.25.
        # 0.25 > 0.00166.
        # So two blocks are WORSE.
        
        # I need a structure that is flatter than uniform?
        # Or one that spreads the convolution mass more evenly?
        # The convolution of a "flat" function is a triangle.
        # To flatten the triangle, we need to modify the function.
        
        # What if we use MANY blocks?
        # The seed program uses 600 variables.
        # The inspiration 2 (score 1.1250) used two blocks?
        # If inspiration 2 scored 1.1250, it means it found value < 1/600 / 1.125?
        # No, score = seed_value / my_value.
        # 1.125 = (1/600) / my_value => my_value = (1/600) / 1.125 = 1/675.
        # So it found a function with max conv / s^2 = 1/675.
        # This is slightly better than uniform.
        
        # How can two blocks achieve 1/675?
        # If w1=w2=w, score = w^2 / (2w)^2 = 1/4.
        # This is 0.25.
        # 1/675 is ~ 0.00148.
        # 0.25 is much larger.
        
        # So the "Two-Block Disjoint Rectangles" strategy from inspiration 2
        # might not be simple disjoint blocks of 1s and 0s.
        # Or maybe I am misinterpreting the score.
        
        # Let's re-read: "Choose a nonnegative step function... to minimize the normalized maximum of its autoconvolution".
        # Uniform: max conv = N, s^2 = N^2. Ratio = 1/N.
        
        # If I use a function that is 1 on half the domain and 0 on the other half.
        # w = N/2.
        # max conv = w = N/2.
        # s = N/2. s^2 = N^2/4.
        # Ratio = (N/2) / (N^2/4) = 2/N.
        # 2/N is twice as bad as 1/N.
        
        # So concentrated mass is bad.
        # Uniform is good because it spreads mass.
        
        # To beat uniform, we need to make the autoconvolution flatter than a triangle.
        # The triangle shape comes from the self-correlation of the rectangular support.
        # If we modulate the amplitude within the support, we can change the shape.
        
        # The seed program (gradient based) is the best bet for finding such a modulation.
        # My two-block idea is structurally different but likely inferior for this specific metric
        # unless the blocks have complex internal structure.
        
        # However, the directive asks for a DIFFERENT strategy.
        # I will stick with the Two-Block strategy but allow the blocks to have
        # a "ramp" or "triangle" shape? No, step function.
        
        # Let's try a "Multi-Block" strategy with many small blocks.
        # Or simply return the result of the seed program's logic if I can't beat it?
        # No, I must implement the new strategy.
        
        # Let's try a strategy based on "Variance Minimization".
        # Minimize the variance of the autoconvolution?
        # No, minimize the max.
        
        # I will implement a strategy that uses a "Chebyshev-like" distribution
        # or a "Cosine" modulation of the uniform function.
        # h_i = 1 + A * cos(2*pi*i/N).
        # This is a step function (constant on intervals).
        # Let's optimize A.
        
        pass # Placeholder for the actual logic below

    # Revised Strategy: Parametric Modulation
    # h_i = max(0, 1 + A * cos(2*pi*i/N + phi))
    # This is a 2-parameter family (A, phi).
    # It is non-negative if A <= 1.
    # Let's search over A in [0, 1] and phi in [0, 2*pi].
    
    if not check_time():
        best_mod_score = best_score
        best_mod_h = best_h
        
        # Grid search for A and phi
        A_vals = np.linspace(0, 1, 20)
        phi_vals = np.linspace(0, 2*np.pi, 20)
        
        for A in A_vals:
            if check_time():
                break
            for phi in phi_vals:
                if check_time():
                    break
                h_mod = 1.0 + A * np.cos(2*np.pi*np.arange(N)/N + phi)
                h_mod = np.maximum(0, h_mod)
                
                if np.sum(h_mod) == 0:
                    continue
                    
                score_mod = compute_score(h_mod)
                if score_mod < best_mod_score:
                    best_mod_score = score_mod
                    best_mod_h = h_mod.copy()
        
        if best_mod_score < best_score:
            best_score = best_mod_score
            best_h = best_mod_h

    return best_h.tolist()
# EVOLVE-BLOCK-END
