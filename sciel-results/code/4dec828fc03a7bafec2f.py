"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))

def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _construct():
    deadline = time.time() + 145.0
    
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    rng = np.random.default_rng(42)
    
    # Strategy: Spectral design via magnitude-phase decomposition.
    # We parameterize the signal x by its FFT magnitude |X| and phase theta.
    # We want to minimize the peak of |IFFT(|X|^2)|.
    # This is equivalent to minimizing the peak of the power spectrum's inverse transform.
    # By keeping |X| flat (or close to it) and optimizing the phase, we can potentially
    # spread the energy more evenly in the time domain of the autoconvolution.
    #
    # However, a more direct approach is to optimize the phase of X such that
    # the autoconvolution is as flat as possible.
    # Note: The autoconvolution is IFFT(|X|^2). The phase of X does NOT appear in |X|^2.
    # This is a critical insight: The autoconvolution depends ONLY on the magnitude of X.
    # Therefore, optimizing the phase of x is useless for this specific metric!
    #
    # So the problem reduces to: Find a magnitude spectrum |X(f)| such that
    # IFFT(|X(f)|^2) is as flat as possible (minimizing its peak), subject to
    # x being real (which |X| always satisfies if we enforce Hermitian symmetry)
    # and x having a non-zero sum (DC term).
    #
    # Wait, x must be real. If we define x via IFFT(X), and X is Hermitian symmetric,
    # x is real.
    # The autoconvolution is IFFT(|X|^2).
    # We want to minimize max_k | (IFFT(|X|^2))_k | / (sum x)^2.
    # sum x = X[0] (DC term).
    # So we want to minimize max_k | c_k | / X[0]^2, where c = IFFT(|X|^2).
    #
    # Let's parameterize |X|^2 directly. Let P_k = |X_k|^2.
    # P must be non-negative and satisfy Hermitian symmetry (P_k = P_{N-k}).
    # Also, for x to be real, X must be Hermitian, which implies P is real and symmetric.
    # The constraint that x is real is automatically satisfied if we construct x from P?
    # No, x = IFFT(X). If we only fix P, we can choose any phase for X.
    # But the metric only depends on P.
    # So we can choose P to minimize the peak of IFFT(P), and then choose any phase
    # for X (e.g., zero phase) to construct x.
    #
    # Is there any constraint on x? "Signed step heights". No specific constraint on x values
    # other than they are real numbers.
    # So the problem is: Minimize max_k | (IFFT(P))_k | / P[0] subject to P >= 0, P symmetric.
    # Note: X[0] = sqrt(P[0]). So (sum x)^2 = P[0].
    #
    # This is a convex optimization problem?
    # Let y = IFFT(P). We want to minimize max |y_k| / P[0].
    # P = FFT(y).
    # This is not obviously convex.
    #
    # Let's try a different perspective.
    # We want the autoconvolution to be as "flat" as possible.
    # A flat autoconvolution (in the non-DC part) would be ideal.
    # If c_k = 0 for k != 0, then x is an orthogonal sequence (like a ZCZ sequence).
    # But x is a vector of length N. The autoconvolution has length 2N-1.
    # If c_k = 0 for all k != 0, then x is an impulse. But then sum x = x_0, and c_0 = x_0^2.
    # Metric = x_0^2 / x_0^2 = 1.
    # Can we do better?
    # If we allow c_k to be small but non-zero, can we reduce the peak?
    # The peak is c_0 = sum x_i^2.
    # We want to minimize (sum x_i^2) / (sum x_i)^2.
    # By Cauchy-Schwarz, (sum x_i)^2 <= N * sum x_i^2.
    # So (sum x_i^2) / (sum x_i)^2 >= 1/N.
    # The minimum is achieved when all x_i are equal.
    # If x_i = 1 for all i, then sum x_i = N, sum x_i^2 = N.
    # Metric = N / N^2 = 1/N = 1/400 = 0.0025.
    # Wait, the metric is max |c_k| / (sum x)^2.
    # For x = ones, c_0 = N. c_k = N - |k| for |k| < N.
    # Max |c_k| = N (at k=0).
    # Metric = N / N^2 = 1/N.
    #
    # Is it possible to get a metric < 1/N?
    # Let's check the constraint again.
    # Metric = max |c_k| / (sum x)^2.
    # We know c_0 = sum x_i^2.
    # So max |c_k| >= c_0 = sum x_i^2.
    # So Metric >= (sum x_i^2) / (sum x_i)^2.
    # As shown above, this ratio is >= 1/N.
    # So the theoretical lower bound is 1/N.
    #
    # The current program scores 1.0765. The seed (ones) scores 1.0.
    # Wait, the seed is ones. Metric(ones) = 1/N = 0.0025.
    # The score is seed/value. So score = 0.0025 / value.
    # If value = 0.0025, score = 1.0.
    # If the current program scores 1.0765, its value is 0.0025 / 1.0765 = 0.002322.
    # This is LOWER than 1/N.
    # How is this possible?
    # My bound max |c_k| >= c_0 assumes c_0 is the maximum.
    # But c_0 is positive. Other c_k can be negative.
    # If c_k are all small in absolute value, but c_0 is large?
    # No, c_0 = sum x_i^2.
    # If we make x_i have both positive and negative values, sum x_i can be small,
    # while sum x_i^2 stays large.
    # But the denominator is (sum x_i)^2.
    # If sum x_i -> 0, the metric -> infinity.
    # So we want sum x_i to be large.
    #
    # Let's re-evaluate the bound.
    # Metric = max |c_k| / (sum x)^2.
    # We want to minimize this.
    # If we choose x such that the autoconvolution is very flat, say c_k = C for all k,
    # then max |c_k| = C.
    # But c_0 = sum x_i^2.
    # If c_k is constant, then x is a chirp or similar?
    # No, a constant autoconvolution implies x is an impulse (in infinite length).
    # For finite length, we can approximate it.
    #
    # Let's look at the inspiration 1 score 1.2718.
    # This implies a value of 0.0025 / 1.2718 = 0.001966.
    # This is significantly lower than 1/N.
    # This suggests that making c_0 NOT the maximum is the key.
    # We want to minimize the maximum absolute value of the autoconvolution.
    # We can do this by making the autoconvolution oscillate, so that its peak is lower
    # than c_0.
    # Wait, c_0 is always the sum of squares, which is positive.
    # If we can make all other c_k have absolute value < c_0, then max |c_k| = c_0.
    # But if we can make c_0 small relative to the other peaks? No, we want to minimize the max.
    #
    # Actually, if we can make the autoconvolution have a "negative" peak that cancels out?
    # No, we take the absolute value.
    #
    # Let's try to construct a signal where the autoconvolution is approximately a constant
    # minus a delta.
    # If c_k = A - B * delta(k), then max |c_k| is max(A, A-B).
    #
    # Let's try a different approach: Random search with local refinement, but using
    # a better metric approximation or a different search space.
    #
    # Given the time constraint, I will implement a robust local search that starts
    # from a random initialization and tries to minimize the metric directly.
    # I will also try a spectral approach where I optimize the magnitude spectrum P
    # to minimize the peak of IFFT(P).
    
    def _metric_fast(x):
        """Fast metric calculation."""
        s = np.sum(x)
        if abs(s) < 1e-12:
            return float('inf')
        fx = np.fft.rfft(x, _FFT_SIZE)
        conv = np.fft.irfft(fx * fx, _FFT_SIZE)
        return float(np.max(np.abs(conv[:2 * N - 1]))) / (s * s)

    def _local_search(x, deadline, rng, max_rounds=20000):
        """
        Local search to refine the solution.
        Uses coordinate descent style updates.
        """
        best_x_local = x.copy()
        best_val_local = _metric_fast(x)
        no_improve = 0
        step_scale = 0.5
        
        for round_idx in range(max_rounds):
            if time.time() > deadline:
                break
            
            if no_improve > 50:
                step_scale *= 0.8
                no_improve = 0
                if step_scale < 0.01:
                    break
            
            # Pick a random index
            idx = rng.integers(0, N)
            
            # Try a few step sizes
            for sign in [1, -1]:
                for scale in [step_scale, step_scale * 0.5, step_scale * 0.1]:
                    x_test = best_x_local.copy()
                    x_test[idx] += sign * scale
                    
                    # Normalize to keep sum constant?
                    # No, the metric is scale invariant in x (numerator scales by x^2, denominator by x^2).
                    # Wait, if x -> alpha x, then conv -> alpha^2 conv, sum -> alpha sum.
                    # Metric -> alpha^2 conv / alpha^2 sum^2 = Metric.
                    # So we can fix the sum to any constant, e.g., N.
                    # This reduces the dimensionality by 1.
                    
                    # Let's fix sum(x) = N.
                    # This means we should normalize x after each update.
                    # But normalization changes all elements.
                    # Alternative: Optimize in the subspace sum(x) = N.
                    # This is complex for local search.
                    
                    # Simpler: Just normalize.
                    S_new = np.sum(x_test)
                    if abs(S_new) > 1e-6:
                        x_test = x_test * (N / S_new)
                    
                    v = _metric_fast(x_test)
                    if v < best_val_local:
                        best_val_local = v
                        best_x_local = x_test
                        no_improve = 0
                        break
                else:
                    continue
                break
            else:
                no_improve += 1
        
        return best_x_local, best_val_local

    # Main loop with restarts
    num_restarts = 20
    time_per_restart = 5.0
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Create a random initial signal
        if i == 0:
            x_init = np.ones(N)
        elif i % 2 == 0:
            # Random smooth signal
            freqs = np.fft.rfftfreq(N)
            amp = 1.0 / (1.0 + 5.0 * freqs)
            phase = rng.uniform(0, 2 * np.pi, size=len(amp))
            X_init = amp * np.exp(1j * phase)
            X_init[0] = np.abs(X_init[0])
            x_init = np.fft.irfft(X_init, N)
        else:
            # Random binary-like signal
            x_init = rng.choice([-1, 1], size=N).astype(float)
            x_init += rng.normal(0, 0.5, size=N)
        
        # Normalize
        S = np.sum(x_init)
        if abs(S) > 1e-6:
            x_init = x_init * (N / S)
        
        # Local refinement
        x_ref, val_ref = _local_search(x_init, deadline - 5, rng, max_rounds=5000)
        
        if val_ref < best_val:
            best_val = val_ref
            best_x = x_ref

    return best_x

def construct():
    return _construct()
# EVOLVE-BLOCK-END
