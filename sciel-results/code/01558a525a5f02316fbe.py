import numpy as np
from scipy import optimize
from scipy.optimize import linprog

N = 600

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    conv = np.convolve(h, h, 'full')
    total = sum(h)
    if total == 0:
        return 0.0
    return max(conv) / (total * total)

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, leaving 10s margin
    
    # Initial guess: all ones (this gives score = 1.0)
    h = np.ones(N)
    best_h = h.copy()
    best_score = compute_autoconv_max(h)
    
    # Frequency-domain approach:
    # The autoconvolution in frequency domain is |H(k)|^2 where H is the DFT of h
    # The maximum of autoconvolution relates to peak of |H|^2
    # We want to minimize max_k |H(k)|^2 / (sum(h))^2
    # 
    # Key insight: By Parseval, sum(conv) = (sum h)^2
    # The peak of conv is bounded by properties of |H|^2
    #
    # Strategy: Minimize the peak of |H(k)|^2 subject to sum(h) = 1 and h >= 0
    # This is equivalent to making |H(k)| as flat as possible (equi-ripple)
    
    def freq_objective(x):
        """Objective: minimize the peak of |FFT(x)|^2 normalized by (sum x)^2"""
        h = np.maximum(x, 0)
        total = np.sum(h)
        if total < 1e-10:
            return 1e10
        # Normalize
        h_norm = h / total
        H = np.fft.fft(h_norm)
        power = np.abs(H)**2
        return np.max(power)
    
    def soft_freq_objective(x, T=0.01):
        """Smooth version using log-sum-exp on |H|^2"""
        h = np.maximum(x, 0)
        total = np.sum(h)
        if total < 1e-10:
            return 1e10
        h_norm = h / total
        H = np.fft.fft(h_norm)
        power = np.abs(H)**2
        pmax = np.max(power)
        shifted = (power - pmax) / T
        shifted = np.clip(shifted, -50, 50)
        soft_max = pmax + T * np.log(np.sum(np.exp(shifted)))
        return soft_max
    
    # Strategy: Use the fact that for a flat spectrum, we want h to be 
    # close to a Dirichlet kernel or similar low-peak-sidelobe sequence
    # 
    # Approach: Start with various initial conditions and optimize in frequency domain
    # The key is that minimizing peak |H|^2 with fixed sum(h) is a convex problem
    # when we work with the power spectrum
    
    # Method 1: Direct optimization of h to flatten |H(k)|
    # The constraint sum(h)=1 and h>=0, minimize max |H(k)|^2
    
    # For a step function with N intervals, the DFT has N points
    # H(k) = sum_j h_j * exp(-2*pi*i*j*k/N)
    # |H(k)|^2 = H(k) * conj(H(k))
    
    # Let's try a different approach: use the relationship between 
    # time-domain and frequency-domain to construct a good solution
    
    # Known result: For minimizing peak sidelobe, optimal sequences are related
    # to Barker sequences or low-peak-sidelobe sequences
    # 
    # For our continuous step function, we can use a window function approach
    # The optimal h should make |H(k)| as flat as possible
    
    # Try: h proportional to a raised cosine or other smooth window
    # that has good spectral properties
    
    # Method: Iteratively adjust h to equalize |H(k)|
    # This is similar to the "water-filling" or "equalization" approach
    
    def equalize_spectrum(h, n_iters=100, lr=0.01):
        """Iteratively equalize the spectrum by adjusting h"""
        h = np.maximum(h, 0).copy()
        for _ in range(n_iters):
            if time.time() > deadline - 15:
                break
            total = np.sum(h)
            if total < 1e-10:
                break
            h_norm = h / total
            H = np.fft.fft(h_norm)
            power = np.abs(H)**2
            pmax = np.max(power)
            # Gradient of max |H|^2 with respect to h
            # d|H(k)|^2/dh_j = 2*Re{conj(H(k)) * exp(-2*pi*i*j*k/N)}
            # For the peak, we focus on k where |H(k)| is maximal
            peak_indices = np.where(power > 0.9 * pmax)[0]
            if len(peak_indices) == 0:
                break
            # Compute gradient for peak frequencies
            grad = np.zeros(N)
            for k in peak_indices:
                Hk = H[k]
                # d|H(k)|^2/dh_j = 2*Re{conj(Hk) * exp(-2*pi*i*j*k/N)}
                phase = np.exp(-2j * np.pi * np.arange(N) * k / N)
                grad += 2 * np.real(np.conj(Hk) * phase)
            # Update: decrease h where gradient is positive (increases peak)
            h -= lr * grad / pmax
            h = np.maximum(h, 0)
        return h
    
    # Method 2: Use convex optimization with epigraph formulation
    # minimize t
    # subject to |H(k)|^2 <= t * (sum h)^2 for all k
    #            sum(h) = 1
    #            h >= 0
    #
    # This is not directly convex in h because of (sum h)^2, but if we fix sum(h)=1,
    # it becomes: minimize t, s.t. |H(k)|^2 <= t, sum(h)=1, h>=0
    # |H(k)|^2 is convex in h (it's a quadratic form), so this IS convex!
    
    # Let's use this convex formulation
    # Variables: h_0, ..., h_{N-1}, t
    # Minimize: t
    # Subject to: |H(k)|^2 <= t for all k=0,...,N-1
    #             sum(h) = 1
    #             h >= 0
    #
    # |H(k)|^2 = |sum_j h_j exp(-2pi i j k/N)|^2
    # This is a quadratic constraint, making this a SOCP/QP
    
    # For large N, solving this exactly is expensive. Let's use an iterative approach.
    
    # Alternative: Use the fact that the optimal solution should have 
    # |H(k)| approximately constant (flat spectrum)
    # A flat spectrum in frequency corresponds to a delta function in time,
    # but we're constrained to h >= 0 and sum(h) = 1
    # 
    # The closest we can get to a flat spectrum with h >= 0 is a uniform h,
    # which gives H(0) = N, H(k) = 0 for k != 0 (in the ideal case)
    # But for finite N, uniform h gives H(k) = N*delta_{k,0} approximately
    # 
    # Actually, for uniform h = 1/N, H(k) = sum_j (1/N) exp(-2pi i j k/N)
    # H(0) = 1, H(k) = 0 for k != 0 (exactly, by geometric series)
    # So uniform h gives peak |H|^2 = 1, and all other |H(k)|^2 = 0
    # This is already optimal in some sense!
    
    # Wait, let me reconsider. The score is max(conv) / (sum h)^2
    # For uniform h, conv is a triangular function with peak at center
    # conv_peak = sum_j h_j * h_j = sum h^2 = N * (1/N)^2 = 1/N
    # (sum h)^2 = 1
    # So score = 1/N = 1/600 ≈ 0.00167
    # But the seed score is 1.0, so I'm misunderstanding the normalization
    
    # Let me re-read: "normalized maximum of its autoconvolution"
    # The score for all ones is 1.0 (the seed)
    # For h = ones(N), conv = triangular with peak N, sum(h) = N
    # score = N / N^2 = 1/N... but that gives 1/600, not 1.0
    
    # Ah, I see - the score is max(conv)/(sum h)^2, and for all ones:
    # conv_peak = N (since conv[j] = sum_i h[i]*h[j-i], at peak it's N*1*1 = N)
    # sum(h) = N
    # score = N/N^2 = 1/N
    # But the task says seed gives 1.0... 
    
    # Let me just trust the compute_autoconv_max function and work with it
    
    # Given the complexity, let me try a more direct approach:
    # Use scipy.optimize with the frequency-domain insight
    
    # Key insight from the problem: We want to minimize the peak of autoconvolution
    # relative to the total energy. In frequency domain, this is related to
    # making the power spectrum as flat as possible.
    
    # Let's try a simple but effective approach:
    # 1. Start with various initial conditions
    # 2. Use gradient descent on the soft-max objective
    # 3. Polish with local optimization
    
    np.random.seed(42)
    
    # Soft objective for optimization (smooth approximation of max)
    def objective_for_opt(x, T=0.005):
        h = np.maximum(x, 0)
        total = np.sum(h)
        if total < 1e-10:
            return 1e10
        conv = np.convolve(h, h, 'full')
        conv_max = np.max(conv)
        shifted = (conv - conv_max) / T
        shifted = np.clip(shifted, -50, 50)
        soft_max = conv_max + T * np.log(np.sum(np.exp(shifted)))
        return soft_max / (total * total)
    
    # Try multiple initial conditions
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(N))
    
    # 2. Raised cosine
    t = np.linspace(0, 1, N, endpoint=False)
    initializations.append(0.5 + 0.5 * np.cos(2 * np.pi * t))
    initializations[-1] = np.maximum(initializations[-1], 0.01)
    
    # 3. Triangular
    t = np.linspace(-1, 1, N)
    initializations.append(1 - np.abs(t))
    initializations[-1] = np.maximum(initializations[-1], 0.01)
    
    # 4. Gaussian
    t = np.linspace(-1, 1, N)
    initializations.append(np.exp(-t**2 * 2))
    
    # 5. Random
    initializations.append(np.random.rand(N) + 0.1)
    
    # 6. Inverse cosine (peak at edges)
    initializations.append(0.5 - 0.5 * np.cos(2 * np.pi * t))
    initializations[-1] = np.maximum(initializations[-1], 0.01)
    
    # 7. Two-lobe
    initializations.append(np.maximum(0.5 + 0.4 * np.cos(4 * np.pi * t), 0.01))
    
    # 8. Three-lobe
    initializations.append(np.maximum(0.5 + 0.3 * np.cos(6 * np.pi * t), 0.01))
    
    for h0 in initializations:
        if time.time() > deadline - 20:
            break
        
        h0 = np.maximum(h0, 0.001)
        
        # Coarse optimization
        try:
            result = optimize.minimize(
                objective_for_opt, h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_new = np.maximum(result.x, 0)
            score_new = compute_autoconv_max(h_new)
            
            if score_new < best_score:
                best_score = score_new
                best_h = h_new.copy()
                
                # Fine polish
                if time.time() < deadline - 15:
                    result2 = optimize.minimize(
                        objective_for_opt, h_new,
                        method='L-BFGS-B',
                        bounds=[(0, None)] * N,
                        options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    h_polished = np.maximum(result2.x, 0)
                    score_polished = compute_autoconv_max(h_polished)
                    
                    if score_polished < best_score:
                        best_score = score_polished
                        best_h = h_polished.copy()
                        
        except Exception:
            pass
    
    # Final: try to improve with small perturbations
    if time.time() < deadline - 10:
        for _ in range(3):
            h_pert = best_h + 0.005 * np.random.randn(N)
            h_pert = np.maximum(h_pert, 0)
            try:
                result = optimize.minimize(
                    objective_for_opt, h_pert,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                h_new = np.maximum(result.x, 0)
                score_new = compute_autoconv_max(h_new)
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new.copy()
            except Exception:
                pass
    
    return list(best_h)
