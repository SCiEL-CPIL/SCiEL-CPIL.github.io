"""
Third autocorrelation inequality: minimize the upper bound on C3
by choosing a signed step function with 400 equal intervals on [-1/4, 1/4].

Strategy: Greedy coordinate descent over the 2^400 sign space, using FFT
to evaluate the autoconvolution in O(N log N) per flip. Start from a
sine-derived initial pattern and polish with greedy flips within a time budget.
"""
import numpy as np
import time

N = 400
TIME_LIMIT = 150.0  # seconds
SAFETY_MARGIN = 5.0  # return this many seconds before the limit


def _autoconv_max_abs(signs):
    """
    Compute the maximum absolute value of the autoconvolution of the
    step function defined by the sign vector.
    
    The step function has N equal intervals on [-1/4, 1/4], each of width
    1/(2*N). The autoconvolution at lag k (in interval units) is:
        (autoconv)[k] = sum_i signs[i] * signs[i+k] * (width^2)
    for k = -(N-1), ..., N-1.
    
    The upper bound on C3 involves the L-infinity norm of the autoconvolution
    of |f| or similar. The task says "score is the upper bound on C3" and
    "lower is better". We compute the max absolute value of the circular
    autoconvolution (or linear) as a proxy.
    
    Actually, re-reading: "autoconvolution upper bound". The C3 constant
    in the third autocorrelation inequality typically involves:
        C3 >= max_k |autoconv_f(k)| / (integral(f)^2) or similar.
    
    For a signed step function with values +/-1 on intervals of width w=1/(2N),
    the autoconvolution (linear) at lag m (in units of w) is:
        A[m] = w^2 * sum_i f_i * f_{i+m}  for m = 0, 1, ..., N-1
        A[-m] = A[m] by symmetry if we consider the full function.
    
    The "upper bound on C3" is likely proportional to max_m |A[m]|.
    Since w^2 is a constant factor, minimizing max_m |sum_i f_i f_{i+m}|
    is equivalent.
    
    We'll compute the linear autoconvolution via FFT and take the max abs.
    """
    # Pad to avoid circular convolution artifacts
    size = 2 * N
    f = np.zeros(size)
    f[:N] = signs
    # Linear autoconvolution via FFT
    F = np.fft.fft(f)
    autoconv = np.fft.ifft(F * np.conj(F)).real
    # The autoconvolution has non-zero values from index 0 to 2N-2
    # Take the max absolute value
    return np.max(np.abs(autoconv))


def _greedy_polish(signs, start_time, time_limit):
    """
    Greedy coordinate descent: repeatedly flip the sign that most reduces
    the autoconvolution max-abs value. Use a time budget.
    """
    current = signs.copy().astype(np.float64)
    current_val = _autoconv_max_abs(current)
    best_val = current_val
    best_signs = current.copy()
    
    # Precompute for speed: we'll use FFT-based evaluation
    # For each flip at position j, the change in autoconvolution can be computed
    # But for simplicity and correctness, we evaluate via FFT each time.
    # To speed up, we can precompute the FFT of the current signal.
    
    n_passes = 0
    no_improve = 0
    max_no_improve = 3  # stop if 3 full passes with no improvement
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit - SAFETY_MARGIN:
            break
        
        n_passes += 1
        improved_this_pass = False
        
        # Random order to avoid getting stuck in a particular pattern
        order = np.random.permutation(N)
        
        for j in order:
            elapsed = time.time() - start_time
            if elapsed > time_limit - SAFETY_MARGIN:
                return best_signs, best_val
            
            # Try flipping position j
            new_signs = current.copy()
            new_signs[j] *= -1
            
            new_val = _autoconv_max_abs(new_signs)
            
            if new_val < current_val - 1e-12:
                current = new_signs
                current_val = new_val
                improved_this_pass = True
                if current_val < best_val:
                    best_val = current_val
                    best_signs = current.copy()
        
        if not improved_this_pass:
            no_improve += 1
            if no_improve >= max_no_improve:
                break
        else:
            no_improve = 0
    
    return best_signs, best_val


def construct():
    start_time = time.time()
    
    # Initial pattern: sine-based sign vector
    # This gives a smooth transition between positive and negative blocks
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    # Low-frequency sine to create blocks of same sign
    # Period chosen so we get a few sign changes
    freq = 2.0  # 2 full periods over [-1/4, 1/4]
    sine_vals = np.sin(2 * np.pi * freq * x)
    signs = np.sign(sine_vals)
    # Ensure no zeros (replace 0 with 1)
    signs[signs == 0] = 1
    signs = signs.astype(np.float64)
    
    # First greedy polish
    best_signs, best_val = _greedy_polish(signs, start_time, TIME_LIMIT)
    
    # If time remains, try a few random restarts
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - SAFETY_MARGIN - 10:
        n_restarts = 3
        for r in range(n_restarts):
            elapsed = time.time() - start_time
            if elapsed > TIME_LIMIT - SAFETY_MARGIN - 5:
                break
            
            # Random restart with some structure
            # Mix of random and sine-based
            if r % 2 == 0:
                # Random signs
                restart_signs = np.random.choice([-1.0, 1.0], size=N)
            else:
                # Different frequency sine
                freq_r = 1.0 + r * 0.5
                sine_r = np.sin(2 * np.pi * freq_r * x)
                restart_signs = np.sign(sine_r)
                restart_signs[restart_signs == 0] = 1
                restart_signs = restart_signs.astype(np.float64)
            
            polished_signs, polished_val = _greedy_polish(
                restart_signs, start_time, TIME_LIMIT
            )
            
            if polished_val < best_val:
                best_val = polished_val
                best_signs = polished_signs
    
    # Final check
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        # Return best found so far
        pass
    
    # The score is the upper bound on C3, which is proportional to
    # max |autoconv|. The constant factor w^2 = (1/(2N))^2 = 1/(4*N^2)
    # But the evaluator measures the "raw value" - we need to figure out
    # what exactly is being measured.
    #
    # Looking at the seed: it returns np.ones(N) and scores 1.0.
    # The score is "seed/value" so value = seed_score / score = 1.0/1.0 = 1.0
    # Wait, "the search score is seed/value, so 1.0 equals the seed program"
    # So if seed scores 1.0, then value_seed = seed_score / 1.0 = 1.0
    # And for our program, score = seed_score / value_ours
    # We want score > 1.0, so value_ours < value_seed.
    #
    # The "value" is the "upper bound on C3". For the all-ones signal:
    # autoconv[0] = sum of 1*1 over N terms = N (times w^2)
    # autoconv[m] = (N-m) * w^2 for m = 1, ..., N-1
    # max |autoconv| = N * w^2 = N * (1/(2N))^2 = 1/(4N)
    #
    # But what is the actual C3 upper bound? It might involve normalization.
    # Let's just return the sign vector and let the evaluator compute the score.
    # The evaluator will compute the "raw value" from our returned sequence.
    #
    # Return the best sign vector as a sequence of signed step heights.
    return best_signs.tolist()
