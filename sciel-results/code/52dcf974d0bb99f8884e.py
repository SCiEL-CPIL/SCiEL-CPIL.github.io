"""
Strategy: Continuous Spectral Amplitude Optimization.
Instead of discrete bit-flips on a rugged max|acf| landscape, we:
1. Define a smooth objective: minimize the soft-max of the power spectrum |X|^2.
   This is equivalent to minimizing the H-infinity norm of the sequence's transfer function.
2. Use L-BFGS-B to optimize the amplitudes |X| in the frequency domain.
   The phase is fixed (e.g., zero) during optimization, which is valid because 
   the objective depends only on the power spectrum for the "average" case, 
   but we must be careful: max|acf| is not just related to the integral of the spectrum, 
   it's the inverse FFT. However, minimizing the max of the spectrum (flatness) 
   is a strong proxy for minimizing the max of the time-domain autocorrelation, 
   especially for large N.
3. After finding optimal amplitudes, we need to construct a time-domain sequence.
   We can reconstruct x by taking the IFFT of the optimized spectrum with a random phase, 
   then thresholding to signs. 
   Alternatively, we can use the optimized spectrum to guide a spectral phase retrieval 
   or simply use the fact that for binary sequences, a flat power spectrum is the target.
   
   Actually, a more direct approach:
   The problem is min max_lag |sum x_i x_{i+lag}| / (sum x_i^2).
   For x in {-1, 1}^N, sum x_i^2 = N.
   So we want min max_lag |ACF(lag)|.
   
   The "flat spectrum" approach:
   If the power spectrum |X|^2 is constant, then the ACF is a delta function (at lag 0) 
   and zero elsewhere (in the circular sense). This is the ideal.
   We optimize the amplitudes a_k = |X_k| such that a_k^2 is as flat as possible.
   Specifically, we minimize: sum_k (a_k^2 - mean(a^2))^2 or max(a^2).
   
   But wait, the current program's score is 0.02, which is very low. 
   The seed (ones) scores 1.0. 
   If we can get a flat spectrum, the max ACF (excluding lag 0) should be very small.
   For random signs, the ACF is roughly Gaussian with std ~ sqrt(N). 
   Max of N Gaussian samples is ~ sqrt(2 log N). 
   Normalized by N, this is ~ sqrt(2 log N)/N.
   For N=400, sqrt(2*6)/400 ~ 3.5/400 ~ 0.009.
   The current score 0.02 is consistent with random-ish performance.
   
   We can do better by enforcing flatness.
   
   Let's optimize the amplitudes directly.
   Let z_k be the complex DFT coefficients. We fix the phase to be random or zero?
   If we fix phase to zero, x is real and even. This restricts the search space.
   Better: optimize the amplitudes a_k, and use a random phase initialization, 
   then perform a few steps of spectral phase retrieval to find a phase that 
   yields a real x, and then threshold x to signs.
   
   Actually, the most robust "different" strategy is:
   1. Start with random amplitudes a_k.
   2. Minimize the peak of the reconstructed ACF (or its smooth surrogate) 
      with respect to the amplitudes a_k, keeping the phases fixed at their initial random values.
      This is not quite right because changing amplitudes changes the ACF non-locally.
      
   Let's stick to the "Flatten the Spectrum" idea.
   Objective: Minimize the variance of the power spectrum P_k = |X_k|^2.
   Or Minimize max(P_k).
   Since we can't easily optimize max(P_k) directly with L-BFGS without smoothing, 
   we use the soft-max: M_beta(P) = (1/beta) log(sum exp(beta P_k)).
   
   We optimize the amplitudes a_k (where X_k = a_k * exp(i phi_k)).
   We can fix phi_k = 0 for simplicity in the first pass, or use random phases.
   If we use random phases, the resulting x will be complex. We need real x.
   So we must enforce Hermitian symmetry: X_{N-k} = conj(X_k).
   
   Revised Plan:
   1. Parameterize the real, symmetric power spectrum P_k.
   2. Optimize P_k to be as flat as possible (minimize variance or soft-max).
   3. Once we have a flat target spectrum, use Spectral Phase Retrieval (Gerchberg-Saxton) 
      to find a phase that corresponds to a real, sign-like sequence.
      - Start with random phase.
      - Iterate:
        a. IFFT to get x.
        b. Threshold x to signs (or project to nearest signs).
        c. FFT to get new X.
        d. Replace |X| with sqrt(P_target).
   4. Return the signs.
   
   This combines spectral optimization (to find a good target) with phase retrieval (to find a valid sequence).
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _compute_score(x):
    """Compute the score: max|autoconvolution| / (integral)^2.
    For x in {-1, 1}^N, integral^2 = N.
    So score = max|acf| / N.
    """
    X = np.fft.rfft(x)
    power = X * np.conj(X)
    acf = np.fft.irfft(power, N)
    acf_max = np.max(np.abs(acf))
    norm_sq = np.sum(x * x)
    if norm_sq < 1e-12:
        return np.inf
    return acf_max / norm_sq

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # 1. Optimize the target power spectrum to be flat.
    # We want to find a flat spectrum that is achievable by a binary sequence.
    # A perfectly flat spectrum corresponds to a delta ACF.
    # For binary sequences, the DC term (lag 0) is fixed at N.
    # The other terms should be small.
    # Let's try to minimize the variance of the power spectrum, excluding DC.
    
    # Parameter: amplitudes of the non-DC frequencies.
    # X has N/2 + 1 bins (0 to N/2).
    # Bin 0 is DC. Bin N/2 is Nyquist.
    # We optimize bins 1 to N/2 - 1 (and N/2 if N is even).
    
    n_freqs = N // 2 + 1
    # Indices to optimize: 1 to n_freqs - 1
    # We keep DC (bin 0) fixed? No, DC is determined by sum(x). 
    # For balanced signs, sum(x) ~ 0, so DC ~ 0.
    # Let's fix DC to 0 and optimize the rest.
    
    # Initial guess: flat spectrum with some noise.
    # The total energy is N (sum x^2 = N).
    # Parseval: sum |X_k|^2 = N * sum x^2 = N^2? 
    # Wait. np.fft.fft normalization: 
    # X_k = sum x_n exp(-2j pi k n / N)
    # sum |X_k|^2 = N * sum |x_n|^2 = N * N = N^2.
    # So average power per bin is N^2 / N = N.
    # So we want |X_k|^2 ~ N for all k.
    
    # Let's define the variable as the log-amplitudes to keep them positive.
    # Or just amplitudes.
    # Let's use amplitudes a_k for k=1..N/2.
    # We will enforce symmetry.
    
    # Variable vector: a[1..N/2]
    # Size: N/2
    size_opt = N // 2
    a0 = np.ones(size_opt) * np.sqrt(N)  # Initial guess: flat
    
    def flatness_objective(a):
        # a corresponds to bins 1..N/2
        # Construct full power spectrum P
        P = np.zeros(n_freqs)
        P[0] = 0  # DC fixed to 0 for now (balanced signs)
        P[1:] = a**2
        
        # Check time: if we are getting close to deadline, return a large value?
        # No, just let it finish the current iteration.
        
        # Objective: Minimize the variance of P, or the soft-max.
        # Variance is smooth.
        mean_P = np.mean(P)
        var_P = np.var(P)
        
        # Also penalize deviation from the mean to encourage flatness
        # We want P to be as close to N as possible.
        # So minimize sum (P_k - N)^2
        return np.sum((P - N)**2)
    
    # Optimize a
    # Use L-BFGS-B
    res = minimize(flatness_objective, a0, method='L-BFGS-B', 
                   options={'maxiter': 100, 'ftol': 1e-8})
    
    # The optimal flat spectrum is just P_k = N for all k.
    # So the optimization above is trivial. 
    # The hard part is finding a phase that realizes this spectrum with a binary sequence.
    
    # So, we set the target spectrum to be perfectly flat: P_target[k] = N for all k.
    # Except we might want to adjust it slightly to make phase retrieval easier?
    # No, let's try perfect flatness first.
    
    P_target = np.full(n_freqs, N)
    # DC term: For balanced signs, DC is 0. But if we force DC=0, 
    # the sequence must have zero sum. 
    # Let's allow the DC to be whatever the phase retrieval finds, 
    # but the target for other bins is N.
    # Actually, if we force P[0]=0, the resulting x will have sum 0.
    
    # 2. Spectral Phase Retrieval (Gerchberg-Saxton)
    # Initialize with random phase.
    rng = np.random.RandomState(42)
    
    # Best sequence found so far
    best_x = None
    best_score = np.inf
    
    # Run multiple restarts of phase retrieval
    num_restarts = 10
    iters_per_restart = 50
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random phase initialization
        phase = rng.uniform(0, 2 * np.pi, size=n_freqs)
        # Ensure Hermitian symmetry for real x?
        # np.fft.irfft takes rfft input, which is implicitly Hermitian.
        # So we just need to define the rfft spectrum.
        
        # X_target = sqrt(P_target) * exp(i * phase)
        # But we must handle the DC and Nyquist bins carefully.
        # For irfft, the input is length N//2 + 1.
        # The phase for DC (k=0) and Nyquist (k=N/2) must be 0 or pi (real).
        phase[0] = 0
        if N % 2 == 0:
            phase[-1] = 0
            
        X = np.sqrt(P_target) * np.exp(1j * phase)
        
        # Iterative refinement
        for i in range(iters_per_restart):
            if time.time() > deadline:
                break
                
            # 1. IFFT to get x
            x = np.fft.irfft(X, N)
            
            # 2. Project to signs
            # Threshold at 0
            x_signs = np.sign(x)
            # Handle zeros: assign random sign or 1
            x_signs[x_signs == 0] = 1.0
            
            # 3. FFT to get new spectrum
            X_new = np.fft.rfft(x_signs)
            
            # 4. Replace amplitude with target
            # Keep the phase of X_new
            phase_new = np.angle(X_new)
            # Handle DC and Nyquist
            phase_new[0] = 0
            if N % 2 == 0:
                phase_new[-1] = 0
                
            X = np.sqrt(P_target) * np.exp(1j * phase_new)
            
            # 5. Check score
            score = _compute_score(x_signs)
            if score < best_score:
                best_score = score
                best_x = x_signs.copy()
                
    # Fallback if no good result
    if best_x is None:
        # Return random signs
        best_x = rng.choice([-1.0, 1.0], size=N)
        
    return best_x
