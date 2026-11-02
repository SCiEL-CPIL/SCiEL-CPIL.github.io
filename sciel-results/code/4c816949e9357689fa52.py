import numpy as np
import time

N = 600

def _autoconv_peak(x):
    """
    Compute the normalized maximum of the autoconvolution of x.
    x is an array of length N representing step heights on equal intervals of [-1/4, 1/4].
    
    The autoconvolution is (x * x)(t) = integral x(s) x(t-s) ds.
    For step functions on equal intervals, this is a discrete convolution scaled by interval width.
    The "normalized maximum" likely refers to max(autoconv) / (sum(x))^2 or similar.
    Let's think about the normalization.
    
    If x is a step function with height h_i on interval [a_i, a_{i+1}], width w = 1/2 / N.
    The L1 norm is sum(h_i * w).
    The autoconvolution at 0 is sum h_i^2 * w.
    The max of the autoconvolution is at most (sum h_i * w)^2 / w = (sum h_i)^2 * w.
    Wait, the autoconvolution of a function f with itself is (f*f)(t).
    If we normalize by the square of the L1 norm, i.e., max(f*f) / (||f||_1^2), 
    for the uniform function x_i = 1, ||f||_1 = N*w.
    (f*f)(t) is a triangular function with peak at t=0.
    Peak = sum h_i^2 w = N * 1 * w = N*w.
    ||f||_1^2 = (N*w)^2.
    Ratio = (N*w) / (N*w)^2 = 1/(N*w) = 1/(N * 1/(2N)) = 2.
    
    Let's check the seed score. The seed returns [1.0]*N.
    The task says "score is the upper bound on C1".
    "1.0 equals the seed program".
    So if the seed score is 1.0, and the seed is [1.0]*N, then the metric for [1.0]*N is 1.0.
    
    Let's re-evaluate the normalization.
    Maybe it's max(autoconv) / max_possible_autoconv?
    Or maybe the interval width is not included in the "normalized" part?
    If we just take the discrete convolution c = np.convolve(x, x, mode='full'),
    the peak of the uniform convolution is N.
    If we divide by N, we get 1.0.
    So the metric might be max(convolve(x, x)) / N.
    Let's assume the metric is `np.max(np.convolve(x, x, mode='full')) / N`.
    For x = ones(N), convolve is [1, 2, ..., N, ..., 2, 1]. Max is N. N/N = 1.0.
    This matches the seed score of 1.0.
    
    So the objective is to minimize max(convolve(x, x)) / N, subject to x_i >= 0.
    Note: The problem says "nonnegative step function". It does NOT say the area is fixed.
    Wait, if the area is not fixed, I can just set all x_i to 0, and the score is 0.
    "Choose a nonnegative step function ... to minimize the normalized maximum of its autoconvolution".
    Usually in these problems, there is a constraint like ||x||_1 = 1 or ||x||_2 = 1 or sum(x) = 1.
    The seed is [1.0]*N. Sum is N.
    If I scale x by alpha, convolve scales by alpha^2.
    If there is no constraint, the minimum is 0.
    Since the seed is 1.0 and not 0, there must be a normalization or constraint.
    "normalized maximum".
    If the metric is max(conv) / (sum(x))^2 * N?
    For seed: max(conv)=N, sum=N. N / N^2 * N = 1.
    If I scale x by 2, max(conv) = 4N, sum=2N. 4N / 4N^2 * N = 1.
    So the metric is scale-invariant.
    This implies the constraint is likely sum(x) = constant, or the metric is scale-invariant.
    Given "normalized maximum", `max(conv) / (sum(x)^2 / N)` seems plausible.
    Let's assume the metric is `M(x) = max(np.convolve(x, x, mode='full')) / (np.sum(x)**2 / N)`.
    For x=ones(N), M = N / (N^2/N) = 1.
    
    Let's verify this interpretation.
    If I put all mass in one bin, x = [N, 0, ..., 0].
    sum = N. sum^2/N = N.
    conv = [N^2, 0, ...]. Max = N^2.
    M = N^2 / N = N. This is much worse than 1.
    So spreading mass is better.
    The uniform distribution minimizes the peak of the autoconvolution relative to the square of the total mass.
    This is a known result: the uniform distribution on an interval minimizes the L-infinity norm of the autoconvolution relative to the L1 norm squared.
    
    So the task is to find x >= 0 that minimizes `max(convolve(x, x)) / (sum(x)^2 / N)`.
    Since it's scale invariant, we can fix `sum(x) = N` (or 1) to simplify.
    Let's fix `sum(x) = N`. Then the metric is `max(convolve(x, x)) / N`.
    We want to minimize `max(convolve(x, x))`.
    
    Strategy: Multi-start Simulated Annealing.
    1. Initialize x with random non-negative values, normalize to sum=N.
    2. Compute current score.
    3. Loop:
       - Pick a random index i.
       - Perturb x[i] by adding Gaussian noise.
       - Ensure x[i] >= 0.
       - Renormalize? No, if we fix sum, we need to adjust another index or rescale.
       - Simpler: Perturb two indices i and j. Add delta to x[i], subtract delta from x[j]. Keep sum constant.
       - Or: Just perturb x[i] and rescale the whole vector to sum=N. This changes all values slightly.
       - Let's use the rescaling approach. It's simpler.
       - Compute new score.
       - Accept if better, or with probability exp(-delta/T).
       - Cool T.
    
    Deadline guard: 150s limit. Use 140s for computation.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial best
    x_best = np.ones(N)
    score_best = 1.0
    
    def compute_score(x):
        # x is sum-normalized to N
        c = np.convolve(x, x, mode='full')
        return np.max(c) / N
    
    # Multi-start
    num_starts = 5
    rng = np.random.default_rng(42)
    
    for s in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        # Initialize random x
        if s == 0:
            x = np.ones(N)
        else:
            # Random exponential distribution
            x = rng.exponential(1.0, size=N)
        
        x = x / np.sum(x) * N
        
        current_score = compute_score(x)
        if current_score < score_best:
            score_best = current_score
            x_best = x.copy()
            
        # Simulated Annealing
        T = 1.0
        cooling = 0.999
        min_T = 1e-6
        
        while T > min_T and time.time() < deadline - 5:
            # Perturbation
            i = rng.integers(0, N)
            noise = rng.normal(0, 1.0)
            x_new = x.copy()
            x_new[i] += noise
            
            # Enforce non-negativity
            if x_new[i] < 0:
                x_new[i] = 0
            
            # Rescale to sum=N
            s_new = np.sum(x_new)
            if s_new == 0:
                x_new = np.ones(N)
            else:
                x_new = x_new / s_new * N
            
            new_score = compute_score(x_new)
            delta = new_score - current_score
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                x = x_new
                current_score = new_score
                if current_score < score_best:
                    score_best = current_score
                    x_best = x.copy()
            
            T *= cooling
            
            # Fast cooling if no improvement for a while? 
            # For now, fixed cooling.
            
    return list(x_best)

def construct():
    return _autoconv_peak_wrapper()

def _autoconv_peak_wrapper():
    # The function above is the logic. Let's wrap it properly.
    return _run_sa()

def _run_sa():
    N = 600
    start_time = time.time()
    deadline = start_time + 140.0
    
    x_best = np.ones(N)
    score_best = 1.0
    
    def compute_score(x):
        c = np.convolve(x, x, mode='full')
        return np.max(c) / N
    
    num_starts = 5
    rng = np.random.default_rng(42)
    
    for s in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        if s == 0:
            x = np.ones(N)
        else:
            x = rng.exponential(1.0, size=N)
        
        x = x / np.sum(x) * N
        
        current_score = compute_score(x)
        if current_score < score_best:
            score_best = current_score
            x_best = x.copy()
            
        T = 1.0
        cooling = 0.999
        min_T = 1e-6
        
        while T > min_T and time.time() < deadline - 5:
            i = rng.integers(0, N)
            noise = rng.normal(0, 1.0)
            x_new = x.copy()
            x_new[i] += noise
            
            if x_new[i] < 0:
                x_new[i] = 0
            
            s_new = np.sum(x_new)
            if s_new == 0:
                x_new = np.ones(N)
            else:
                x_new = x_new / s_new * N
            
            new_score = compute_score(x_new)
            delta = new_score - current_score
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                x = x_new
                current_score = new_score
                if current_score < score_best:
                    score_best = current_score
                    x_best = x.copy()
            
            T *= cooling
            
    return list(x_best)
