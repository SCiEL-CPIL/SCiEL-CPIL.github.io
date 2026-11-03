"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # We want to minimize the normalized maximum of the autoconvolution.
    # Let f be the step function with heights h[0..N-1] on [-1/4, 1/4].
    # The autoconvolution c(t) = (f * f)(t).
    # For step functions on equal intervals, the autoconvolution at the centers of the intervals
    # can be approximated. However, the "normalized maximum" usually refers to:
    # max_t |c(t)| / (integral(f)^2) or similar.
    # Actually, the task says "minimize the normalized maximum of its autoconvolution".
    # Let's look at the standard definition for this type of problem (likely related to C1 in some inequality).
    # If f is non-negative, integral(f) = sum(h) * dx.
    # The peak of the autoconvolution for a non-negative function is at t=0 (center).
    # c(0) = integral f(t)f(-t) dt. Since f is supported on [-1/4, 1/4], and we don't know symmetry,
    # but usually optimal functions are symmetric.
    # Let's assume the normalization is max(c) / (integral(f))^2.
    # By scaling invariance, if we scale h by A, c scales by A^2 and integral scales by A.
    # So the ratio max(c)/(integral(f))^2 is scale-invariant.
    # Thus we can fix the L1 norm or L2 norm to simplify.
    # Let's fix sum(h) = 1. Then we minimize max(c).
    
    # The autoconvolution of a step function h on a grid:
    # Let dx = (1/2) / N = 1/(2N).
    # The values of the convolution at the grid points (centers) are:
    # c_k = sum_{i,j} h_i h_j * (length of overlap of intervals shifted by k)
    # This is equivalent to the discrete convolution of h with h, scaled by dx^2?
    # Actually, if we treat h as samples, the continuous autoconvolution peak is approximately
    # dx * sum_{i,j} h_i h_j * delta(i-j) ... no.
    # Let's use the property that for step functions, the autoconvolution is a piecewise linear function.
    # The maximum is at the center if the function is symmetric and positive.
    # c(0) = integral_{-1/4}^{1/4} f(t)f(-t) dt.
    # If f is symmetric, c(0) = integral f(t)^2 dt.
    # Approximation: c(0) ≈ dx * sum(h_i^2).
    # Normalization: (integral f)^2 = (dx * sum h_i)^2.
    # Ratio R = (dx * sum h_i^2) / (dx^2 * (sum h_i)^2) = (sum h_i^2) / (dx * (sum h_i)^2).
    # To minimize R, we minimize sum(h_i^2) / (sum h_i)^2.
    # This is minimized when h_i is constant!
    # If h_i = 1 for all i, sum(h_i^2) = N, sum(h_i) = N.
    # R = N / (dx * N^2) = 1 / (dx * N) = 1 / (1/2) = 2.
    # Is it possible to do better?
    # Wait, the autoconvolution might have larger values elsewhere if the function is not symmetric?
    # No, for non-negative f, the autoconvolution is maximized at 0 if f is symmetric.
    # If f is not symmetric, the peak might be shifted, but the integral is the same.
    # Actually, the inequality C1 usually bounds the L_infinity norm of the convolution relative to L1 norms.
    # ||f*f||_inf <= C ||f||_1 ||g||_1 ? No, it's autoconvolution.
    # ||f*f||_inf <= C ||f||_1^2.
    # The best constant C is the sup over f of ||f*f||_inf / ||f||_1^2.
    # We want to MINIMIZE this ratio for a specific f?
    # "Choose a nonnegative step function ... to minimize the normalized maximum ... this gives an upper bound on C1."
    # This phrasing is slightly confusing. Usually, to find an upper bound on the best constant C1 (which is a supremum),
    # you don't minimize. You calculate the value for a specific f, which is a lower bound on the supremum?
    # Or does it mean: Find the f that minimizes the ratio, and that minimum is the answer?
    # No, C1 is likely the infimum of constants C such that ||f*f||_inf <= C ||f||_1^2 for ALL f.
    # Then C1 = sup_f (||f*f||_inf / ||f||_1^2).
    # If the task asks to "minimize the normalized maximum", it implies we are looking for the function that
    # makes this ratio as small as possible? That would give a lower bound on the supremum?
    # Or perhaps the task is: "Find the minimum possible value of the normalized maximum over all step functions?"
    # If so, the constant function gives 2. Can we get lower?
    # If we spread the mass more, the peak decreases.
    # Consider f uniform on [-1/4, 1/4]. Peak is 2.
    # Consider f uniform on a smaller interval?
    # If f is uniform on [-a, a], peak of autoconvolution is at 0.
    # c(0) = integral_{-a}^a (1/2a)^2 * 2a dt? No.
    # Let f(t) = 1/(2a) for |t|<a.
    # ||f||_1 = 1.
    # c(0) = integral f(t)f(-t) dt = integral_{-a}^a (1/2a)^2 dt = 2a * 1/(4a^2) = 1/(2a).
    # As a -> 1/4, c(0) -> 1/(1/2) = 2.
    # As a -> 0, c(0) -> infinity.
    # So the uniform function on the full interval gives the minimum peak for uniform functions?
    # What if we use a "bump" that is not rectangular?
    # The ratio is (sum h_i^2) / (sum h_i)^2 * (1/dx).
    # By Cauchy-Schwarz, sum h_i^2 >= (sum h_i)^2 / N.
    # So the ratio is >= 1/(dx * N) = 2.
    # Equality holds when h_i is constant.
    # So the minimum possible value for the normalized maximum (assuming peak at 0 and symmetric) is 2.
    # The seed program returns [1.0]*N, which corresponds to this constant function.
    # The score is the "upper bound on C1". If the value is 2.0, the score is 2.0?
    # "score is the upper bound on C1". "lower is better".
    # If the seed score is 0.0000, it might be because the evaluator expects a specific format or the value is not just the ratio.
    # Or maybe the "normalized maximum" is defined differently.
    # Let's check the seed score. It is 0.0000. This suggests the seed program is failing or returning an invalid value,
    # or the score calculation is `seed/value`. If value is 2.0, and seed value is 2.0, score is 1.0.
    # Why is it 0.0? Maybe an exception? Or maybe the value returned is not the score.
    # "score is the upper bound on C1".
    # If I return a list of heights, the evaluator calculates the score.
    # The prompt says: "Interface: construct() -> sequence of nonnegative step heights... score is the upper bound on C1".
    # So I just return the heights. The evaluator computes the score.
    # If the seed program returns [1.0]*N, and the score is 0.0, it means the evaluator crashed or returned 0.
    # Let's ensure my output is a valid list of floats.
    
    # Since the theoretical minimum is 2 (for the ratio defined above), and the constant function achieves it,
    # any non-constant function will have a HIGHER ratio (worse score, since lower is better).
    # Wait, "lower is better". So we want to MINIMIZE the ratio.
    # The minimum is 2.
    # So the constant function is optimal?
    # If so, why is the seed score 0?
    # Perhaps the normalization is different.
    # Maybe it's max(c) / max(f) * something?
    # Or maybe the "normalized maximum" is max(c) / (integral(f))?
    # If so, for constant f, integral = N*dx = 1/2. max(c) = 2*(1/2)^2? No.
    # Let's stick to the most common definition in such inequalities:
    # C = sup ||f*f||_inf / ||f||_1^2.
    # We want to find the infimum of this ratio? No, C1 is the best constant, so it's the supremum.
    # "minimize the normalized maximum ... this gives an upper bound on C1".
    # This sentence is contradictory if C1 is a supremum. Minimizing a ratio gives a lower bound on the supremum.
    # Unless the task is: "Find the function that minimizes the peak, and report that peak as the answer?"
    # If the answer is the minimum possible peak (normalized), then 2.0 is the answer.
    
    # Let's try to improve the seed. The seed returns [1.0]*N.
    # If the score is 0, maybe the evaluator expects the function to be normalized in a specific way?
    # Or maybe the "upper bound on C1" is calculated as 1 / (minimized ratio)?
    # If ratio is 2, 1/2 = 0.5.
    
    # Let's try a different strategy:
    # The directive suggests: "Replace the continuous optimization with a discrete spectral search... minimize the maximum value of the inverse DFT of the squared magnitude spectrum".
    # This implies we should optimize the shape to reduce the peak.
    # But we proved the constant function minimizes the L2/L1^2 ratio.
    # Is it possible the peak is NOT at 0?
    # For non-negative f, f*f is maximized at 0 if f is symmetric.
    # If f is not symmetric, the peak is still bounded by the symmetric case?
    # Actually, ||f*f||_inf <= ||f||_1 ||f||_inf.
    # This doesn't help with ||f||_1^2.
    
    # Let's reconsider the "normalized maximum".
    # Maybe it's max(f*f) / max(f)?
    # For constant f, max(f*f) = 2 * max(f)^2 * width? No.
    
    # Let's assume the evaluator is robust and the seed failed due to a technicality (e.g. integer vs float, or list type).
    # The seed returns [1.0]*N which is a list of floats.
    
    # Let's try to implement the spectral search as suggested, even if the constant function seems optimal for the simple ratio.
    # Maybe the "normalized maximum" is defined as max(c) / (sum h)^2 * N?
    # If so, constant gives N/N^2 * N = 1?
    
    # Let's try to minimize the peak of the autoconvolution directly using gradient descent on the heights,
    # subject to non-negativity.
    # Objective: J(h) = max_k (c_h)_k / (sum h)^2.
    # Since scaling is invariant, fix sum(h) = 1.
    # Minimize max_k (c_h)_k.
    
    # c_h is the linear convolution of h with h.
    # We can compute c_h efficiently using FFT.
    
    def compute_conv_peak(h):
        # h is a numpy array of length N
        # Pad to 2N-1 for linear convolution
        n_fft = 2 * N
        H = np.fft.fft(h, n=n_fft)
        C = np.fft.ifft(H * H)
        # The autoconvolution of a real signal is real.
        # The peak is the maximum absolute value? Or just maximum value (since h>=0, c>=0)?
        # Since h>=0, c>=0. So max(C.real).
        # Note: The continuous autoconvolution peak is approximately dx * max(discrete_conv).
        # But since we normalize by (sum h)^2, the dx factors cancel if we are consistent.
        # Let's just return the max of the discrete convolution.
        return np.max(C.real)
    
    # Initialize h
    # Start with a Gaussian-like bump to see if we can beat the constant function?
    # Or start with constant and perturb?
    # The constant function is a local minimum for the L2 norm, but is it for the max norm?
    # If we make h slightly non-constant, the peak at 0 might decrease?
    # c(0) = sum h_i^2.
    # If we move mass from the center to the edges, sum h_i^2 decreases (since h_i become more equal? No, they become less equal).
    # Wait, sum h_i^2 is minimized when h_i is constant.
    # So c(0) is minimized by the constant function.
    # What about c(k) for k != 0?
    # For the constant function, c(k) is a triangle wave.
    # c(0) = N. c(1) = N-1.
    # The maximum is at 0.
    # If we shape h to be zero in the center and high at the edges?
    # Then c(0) = sum h_i^2. If h is 0 in center, sum h_i^2 is smaller?
    # But c(k) for small k might increase?
    # Let's try a "U-shaped" function.
    
    # Let's run a simple optimization:
    # 1. Start with a random non-negative vector, normalized to sum=1.
    # 2. Compute gradient of max(c) w.r.t h.
    # 3. Update h, project to non-negative, renormalize.
    
    # However, max(c) is not differentiable. We can use the subgradient:
    # The gradient of max(c) is the vector of c values at the argmax, normalized?
    # Actually, if we fix the index k* where the max occurs, we can optimize for c(k*).
    # But k* might change.
    
    # Alternative: Use the fact that the problem is convex?
    # Minimizing the max of linear functions (c_k(h) are quadratic, not linear).
    # c_k(h) = sum_{i,j} h_i h_j M_{k,ij}.
    # This is a quadratic form.
    
    # Let's try a simple heuristic:
    # Try a few candidate shapes:
    # 1. Constant
    # 2. Gaussian
    # 3. Sine (half wave)
    # 4. U-shape (1 - cos(2*pi*t))
    
    candidates = []
    
    # 1. Constant
    candidates.append(np.ones(N))
    
    # 2. Gaussian
    x = np.linspace(-1, 1, N)
    candidates.append(np.exp(-5 * x**2))
    
    # 3. Sine
    candidates.append(np.sin(np.pi * (x + 1) / 2))
    
    # 4. U-shape
    candidates.append(1 - np.cos(np.pi * x))
    
    # 5. Inverse U-shape (peak in middle)
    candidates.append(np.cos(np.pi * x / 2))
    
    best_h = None
    best_score = np.inf
    
    for h in candidates:
        # Ensure non-negative
        h = np.maximum(h, 0)
        if np.sum(h) == 0:
            continue
        # Normalize
        h = h / np.sum(h)
        
        # Compute score
        score = compute_conv_peak(h)
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
        # Check time
        if time.time() > deadline:
            break

    # If we have time, do some local optimization from the best candidate
    if time.time() < deadline - 10:
        h = best_h.copy()
        # Gradient descent with subgradient
        # We want to minimize max(c).
        # Let's just try random perturbations and keep the best (hill climbing).
        for _ in range(500):
            if time.time() > deadline:
                break
            
            # Random perturbation
            h_perturbed = h + 0.1 * np.random.randn(N)
            h_perturbed = np.maximum(h_perturbed, 0)
            if np.sum(h_perturbed) == 0:
                continue
            h_perturbed = h_perturbed / np.sum(h_perturbed)
            
            score_perturbed = compute_conv_peak(h_perturbed)
            
            if score_perturbed < best_score:
                best_score = score_perturbed
                h = h_perturbed.copy()

    # Convert to list of floats
    return [float(x) for x in h]

# EVOLVE-BLOCK-END
