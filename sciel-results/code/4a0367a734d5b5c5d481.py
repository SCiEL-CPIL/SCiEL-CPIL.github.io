"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(x):
    """Compute the upper bound on C3 for a signed step function x of length N.
    
    The autoconvolution is (x * x)[k] = sum_i x[i] x[i+k].
    The 'upper bound on C3' is interpreted as the maximum absolute value
    of the autoconvolution, excluding the zero-lag term (which is sum x^2 = N for +/-1).
    Actually, looking at the task: "score is the upper bound on C3".
    The seed program returns np.ones(N), which has autoconvolution:
    k=0: N
    k>0: N-k
    max abs is N (at k=0).
    
    However, the task says "Paper and published result notebook place the absolute value differently; execution must explicitly select a metric."
    
    Let's assume the metric is the max of |autocorr[k]| for k != 0, or perhaps the L-infinity norm of the autocorrelation excluding the peak.
    Or maybe it's the sum of squares of the autocorrelation?
    
    Let's look at the "Greedy coordinate descent" hint: "each flip is evaluated in O(N log N) via FFT".
    If we flip x[i] to -x[i], how does the autocorrelation change?
    Let x' be the new vector.
    autocorr'[k] = sum x'[i] x'[i+k]
    
    If the metric is max_k |autocorr[k]|, minimizing this is a minimax problem.
    
    Let's assume the score is max_{k != 0} |autocorr[k]|.
    For x = ones(N), autocorr[k] = N-|k| for |k|<N.
    max_{k!=0} |N-k| = N-1 = 399.
    
    If we use a random sign vector, autocorr[k] is roughly sum of random +/-1, so magnitude ~ sqrt(N).
    max over k would be ~ sqrt(N) * sqrt(log N) ~ 20 * 3 = 60.
    This is much better than 399.
    
    So the goal is to minimize the maximum absolute value of the autoconvolution (off-peak).
    
    Let's implement a score function that computes the autoconvolution via FFT and returns the max absolute value excluding k=0.
    """
    # Compute autoconvolution
    # x is length N. We want circular or linear?
    # "signed step function with 400 equal intervals on [-1/4, 1/4]"
    # This suggests a function on a bounded interval. Autoconvolution usually implies linear convolution (or circular if periodic).
    # Given "step function on [-1/4, 1/4]", it's likely a linear convolution of the step function with itself.
    # But FFT is faster for circular. We can zero-pad to 2N-1 for linear convolution.
    
    n = len(x)
    # Zero-pad to next power of 2 for efficiency, or just 2n
    size = 1
    while size < 2 * n:
        size *= 2
    fft_x = np.fft.rfft(x, size)
    auto = np.fft.irfft(fft_x * fft_x, size)
    # The first n elements are the linear autoconvolution (lags 0 to n-1).
    # auto[0] is sum x^2.
    # We want max |auto[k]| for k != 0.
    # Since x is +/-1, auto[0] = N.
    # We ignore k=0.
    return np.max(np.abs(auto[1:n]))

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Initial guess: random signs
    x = np.random.choice([-1, 1], size=N)
    best_x = x.copy()
    best_score = _score(best_x)
    
    # Greedy coordinate descent
    # For each index i, try flipping x[i] and see if score improves.
    # To make it faster, we can use the property that flipping x[i] only affects
    # autocorr[k] for all k, but the change is structured.
    # However, O(N) per flip * N flips * iterations = O(N^2) per iteration.
    # N=400, N^2=160,000. 1000 iterations = 160M ops. Might be slow in Python.
    
    # Better: Use FFT to compute the full autocorrelation, then for each flip,
    # calculate the change in autocorrelation analytically?
    # Let x' = x with x[i] flipped. x'[i] = -x[i].
    # autocorr'[k] = autocorr[k] - 2*x[i]*sum_{j: j+k=i} x[j+k] ... this is getting complex.
    # Actually, autocorr[k] = sum_j x[j] x[j+k].
    # If we flip x[i], the terms involving x[i] are:
    # 1. j=i, term x[i]x[i+k] -> -x[i]x[i+k] (change -2 x[i] x[i+k])
    # 2. j+k=i, term x[j]x[i] -> x[j](-x[i]) (change -2 x[j] x[i])
    # So autocorr'[k] = autocorr[k] - 2*x[i]*(x[i+k] + x[i-k]) (with bounds check).
    
    # This allows O(N) update of the autocorrelation vector for each flip.
    # Then we need to find the max abs value of the updated vector.
    # This is O(N) to update, O(N) to find max.
    # Total O(N) per flip.
    # N flips per pass. O(N^2) per pass.
    # With N=400, N^2=160k. 100 passes = 16M ops. Very fast.
    
    # Let's implement this.
    
    def get_autocorr(x):
        # Linear autocorrelation
        n = len(x)
        size = 1
        while size < 2 * n:
            size *= 2
        fft_x = np.fft.rfft(x, size)
        auto = np.fft.irfft(fft_x * fft_x, size)
        return auto[:n]  # lags 0 to n-1

    auto = get_autocorr(x)
    
    def score_from_auto(auto):
        return np.max(np.abs(auto[1:]))
    
    current_score = score_from_auto(auto)
    
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            # Try flipping x[i]
            # Calculate change in auto
            # auto'[k] = auto[k] - 2*x[i]*(x[i+k] + x[i-k])
            # Note: x[i+k] is 0 if i+k >= N, x[i-k] is 0 if i-k < 0
            # Wait, the sum is over j such that j and j+k are in [0, N-1].
            # The terms involving x[i] are:
            # j=i -> x[i]x[i+k] (if i+k < N)
            # j=i-k -> x[i-k]x[i] (if i-k >= 0)
            
            change = np.zeros(N)
            # Term 1: j=i
            if i < N:
                # k can be 0 to N-1-i
                # auto[k] gets -2*x[i]*x[i+k]
                # For k in [0, N-1-i], x[i+k] is valid.
                change[0:N-i] -= 2 * x[i] * x[i:N]
            
            # Term 2: j=i-k
            # k can be 0 to i
            # auto[k] gets -2*x[i]*x[i-k]
            # For k in [0, i], x[i-k] is valid.
            change[0:i+1] -= 2 * x[i] * x[0:i+1][::-1]
            
            new_auto = auto + change
            new_score = score_from_auto(new_auto)
            
            if new_score < current_score:
                x[i] = -x[i]
                auto = new_auto
                current_score = new_score
                improved = True
        
        if not improved:
            # Try a few random restarts if stuck? Or just break.
            # Let's try a few random flips to escape local minima.
            for _ in range(10):
                if time.time() > deadline:
                    break
                i = np.random.randint(N)
                x[i] = -x[i]
                # Recalculate auto (expensive but rare)
                auto = get_autocorr(x)
                current_score = score_from_auto(auto)
                # Run one pass of greedy
                improved_inner = False
                for j in range(N):
                    change = np.zeros(N)
                    if j < N:
                        change[0:N-j] -= 2 * x[j] * x[j:N]
                    change[0:j+1] -= 2 * x[j] * x[0:j+1][::-1]
                    new_auto = auto + change
                    new_score = score_from_auto(new_auto)
                    if new_score < current_score:
                        x[j] = -x[j]
                        auto = new_auto
                        current_score = new_score
                        improved_inner = True
                if not improved_inner:
                    break

    # Final polish: use scipy or just return best
    if current_score < best_score:
        best_x = x.copy()
        best_score = current_score
    
    return best_x

# EVOLVE-BLOCK-END
