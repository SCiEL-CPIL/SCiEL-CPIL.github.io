import numpy as np
import time
from scipy.optimize import minimize

N = 600
TIME_LIMIT = 140  # Safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Use FFT for faster convolution calculation
    # np.convolve is O(N^2) ~ 360k ops, very fast for N=600
    # But FFT is O(N log N) ~ 5k ops. Let's use FFT for the objective during optimization.
    n_conv = 2 * N - 1
    # Pad to next power of 2 for FFT efficiency? 1024 is close to 1200.
    # Actually, scipy.signal.fftconvolve is usually best.
    # Or just np.convolve which is highly optimized in numpy for small N.
    # For N=600, np.convolve is likely faster than setting up FFT each time due to overhead?
    # Let's stick to np.convolve for simplicity and robustness, it's < 1ms.
    conv = np.convolve(f, f, mode='full')
    return np.max(conv) / (s * s)

def construct():
    start_time = time.time()
    
    # Strategy: 
    # The uniform distribution f=1/N has score 1/N = 1/600.
    # We want to minimize the peak of the autoconvolution.
    # The autoconvolution of a constant is a triangle (or similar), with peak at center.
    # If we spread the mass out, the peak decreases? 
    # Actually, for fixed L1 norm, the L2 norm of the convolution is related to L2 norm of f.
    # We want to minimize the L-infinity norm of the convolution.
    # This is a convex optimization problem if we fix the sum? 
    # max(conv) is convex. sum(f) is linear. 
    # Minimizing max(conv) subject to sum(f)=1, f>=0 is convex.
    # Wait, the score is max(conv) / (sum(f))^2.
    # If we fix sum(f)=1, we minimize max(conv).
    # This is a convex problem!
    # We can use L-BFGS-B or similar.
    
    # Let's verify convexity.
    # g(f) = max_k (f * f)_k. Each (f*f)_k is quadratic, hence convex. Max of convex is convex.
    # Constraint sum(f) = 1 is linear. f >= 0 is linear.
    # So we are minimizing a convex function over a convex set.
    # Any local minimum is global.
    # The uniform distribution is a candidate. Is it the minimum?
    # If f is uniform, conv is triangle. Peak is 1/N.
    # If we concentrate f, peak increases.
    # If we spread f (but it's on a fixed grid), we can't spread it more than uniform.
    # So uniform might be the minimum?
    # Let's check. If f is two spikes, peak is higher.
    # If f is a "box" of width W, peak is ~ 1/W. Uniform is W=N, peak 1/N.
    # So uniform seems optimal for this metric?
    # BUT, the score in the prompt is "upper bound on C1".
    # The seed program got 1.1638. The uniform score is 1/600 = 0.00166.
    # Wait, the score definition in the prompt: "score is the upper bound on C1".
    # "The evaluator measures the raw value (lower is better)."
    # "search score is seed/value".
    # If seed is uniform (value 1/600), and we get value 1/600, score is 1.0.
    # The current program scores 1.1638. This implies its value is 1/600 / 1.1638 = 0.00142.
    # So the current program found a value LOWER than uniform?
    # How?
    # Let's re-read carefully.
    # "Choose a nonnegative step function... to minimize the normalized maximum of its autoconvolution"
    # Value = max(conv) / (sum(f))^2.
    # If f is uniform, value = 1/N.
    # Can we get lower than 1/N?
    # Consider f being zero in the middle and high at the edges?
    # Convolution of "U" shape.
    # Let's test with N=4.
    # Uniform: [0.25, 0.25, 0.25, 0.25]. Sum=1.
    # Conv: [0.0625, 0.125, 0.1875, 0.125, 0.0625]. Max = 0.1875. Score = 0.1875.
    # 1/N = 0.25. So uniform score is 0.1875? No, 1/N is 0.25.
    # Wait. max(conv) for uniform is (1/N) * (1/N) * N = 1/N.
    # For N=4: 0.25*0.25*4 = 0.25.
    # My manual calc: 0.25^2 * 4 = 0.0625 * 4 = 0.25.
    # Center term: sum f_i f_{j-i}.
    # For uniform, conv[k] = sum f[i]f[k-i].
    # If f[i]=1/N, conv[k] = (number of valid i) * (1/N)^2.
    # Max number of valid i is N (for k in middle).
    # So max conv = N * (1/N)^2 = 1/N.
    # Score = (1/N) / 1^2 = 1/N.
    
    # So uniform gives 1/600.
    # Can we beat 1/600?
    # If we make f non-uniform, does max(conv) decrease?
    # By Cauchy-Schwarz or similar?
    # If we spread the mass, we reduce the "overlap" at the center?
    # No, uniform has maximum overlap at center.
    # If we move mass to edges, the center overlap decreases?
    # Let's try N=4, f=[0, 0.5, 0.5, 0]. Sum=1.
    # Conv:
    # k=0: 0
    # k=1: 0.5*0.5 = 0.25
    # k=2: 0.5*0.5 + 0.5*0.5 = 0.5
    # k=3: 0.25
    # k=4: 0
    # Max = 0.5. Score = 0.5. Worse than 0.25.
    
    # Try f=[0.5, 0, 0, 0.5].
    # Conv:
    # k=0: 0.25
    # k=1: 0
    # k=2: 0
    # k=3: 0
    # k=4: 0.25
    # Max = 0.25. Score = 0.25. Same as uniform.
    
    # Try f=[0.1, 0.3, 0.3, 0.3]. Sum=1.
    # This is getting complicated.
    
    # The current program scores 1.1638, meaning it found a value of ~0.00142, which is LESS than 1/600 (0.00166).
    # So it is possible to beat uniform.
    # How?
    # Maybe the "normalized maximum" is defined differently?
    # "normalized maximum of its autoconvolution"
    # Usually autoconvolution is defined as (f*f)(x) = integral f(y)f(x-y) dy.
    # In discrete: sum f[i] f[j-i].
    # If we normalize by sum(f)^2, we are looking at the probability density function of the sum of two independent variables with PDF f.
    # We want to minimize the peak of this sum distribution.
    # Uniform distribution on [0,1] (discrete) gives Triangular distribution, peak 1. (Normalized by 1).
    # If we take f as two spikes, the sum is two spikes. Peak is higher.
    # If we take f as a "U" shape (high at edges, low in middle), the sum might be flatter?
    # Let's test N=4, f=[0.4, 0.2, 0.2, 0.4]. Sum=1.
    # Conv:
    # k=0: 0.16
    # k=1: 0.4*0.2 + 0.2*0.4 = 0.16
    # k=2: 0.4*0.2 + 0.2*0.2 + 0.2*0.4 + 0.4*0.2 ? No.
    # Indices: 0,1,2,3.
    # conv[k] = sum_{i} f[i] f[k-i].
    # k=0: f[0]f[0] = 0.16
    # k=1: f[0]f[1] + f[1]f[0] = 0.08 + 0.08 = 0.16
    # k=2: f[0]f[2] + f[1]f[1] + f[2]f[0] = 0.08 + 0.04 + 0.08 = 0.20
    # k=3: f[0]f[3] + f[1]f[2] + f[2]f[1] + f[3]f[0] = 0.16 + 0.04 + 0.04 + 0.16 = 0.40
    # k=4: f[1]f[3] + f[3]f[1] = 0.08 + 0.08 = 0.16
    # k=5: f[2]f[3] + f[3]f[2] = 0.08 + 0.08 = 0.16
    # k=6: f[3]f[3] = 0.16
    # Max = 0.40. Worse.
    
    # It seems uniform is very hard to beat for "peak".
    # However, the current program DID beat it.
    # Why?
    # Maybe I am misunderstanding the domain?
    # "600 equal intervals on [-1/4, 1/4]".
    # This is just a discrete vector of length 600.
    
    # Let's look at the inspiration: "Spectral parameterization... decouples frequency-domain shape search... Measured: 1 -> 1.155".
    # This suggests that moving away from uniform in the frequency domain helps.
    # In the frequency domain, f is a set of coefficients.
    # The convolution in time is multiplication in frequency.
    # We want to minimize the max of the inverse FFT of |F|^2.
    # This is equivalent to minimizing the peak of the autocorrelation of the signal in time.
    # Wait, autoconvolution is not autocorrelation.
    # Autocorrelation is f * f(-x). Autoconvolution is f * f(x).
    # For real symmetric f, they are the same.
    # The current program uses a cosine basis, which implies symmetry.
    # If we assume symmetry, we are minimizing the peak of the autocorrelation.
    # For a symmetric signal, the peak of the autocorrelation is at lag 0, which is sum(f^2).
    # So we want to minimize sum(f^2) subject to sum(f)=1?
    # That would mean making f as uniform as possible!
    # But uniform gives 1/N.
    # If we make f non-uniform, sum(f^2) increases.
    # So the peak at lag 0 increases.
    # BUT, the peak of the autocorrelation might not be at lag 0 if f is not positive?
    # But f must be non-negative.
    # For non-negative f, the autocorrelation is maximized at lag 0.
    # Proof: R(0) = sum f[i]^2. R(k) = sum f[i]f[i+k].
    # By Cauchy-Schwarz, R(k)^2 <= R(0) R(0). So R(k) <= R(0).
    # So for non-negative f, the peak is ALWAYS at lag 0.
    # So the score is sum(f^2) / (sum(f))^2.
    # To minimize this, we want to minimize sum(f^2) given sum(f)=1.
    # This is minimized when f is uniform!
    # So the minimum should be 1/N.
    
    # Why does the current program score 1.1638 (value < 1/N)?
    # Is it possible that the "autoconvolution" is defined with a shift?
    # Or maybe the intervals are not aligned with the convolution?
    # "step function with 600 equal intervals".
    # If we integrate, the convolution of two step functions is a piecewise linear function.
    # The "maximum" of the continuous convolution might be lower than the discrete sum?
    # No, the discrete sum approximates the integral.
    
    # Let's re-read the task carefully.
    # "Choose a nonnegative step function... to minimize the normalized maximum of its autoconvolution"
    # If the peak is always at lag 0 for non-negative functions, then the problem is trivial: uniform is optimal.
    # Unless... the function is NOT required to be symmetric?
    # Even if not symmetric, for non-negative f, is the peak of autoconvolution always at lag 0?
    # (f*f)(x) = integral f(y) f(x-y) dy.
    # If f >= 0, then (f*f)(0) = integral f(y)^2 dy.
    # (f*f)(x) = integral f(y) f(x-y) dy.
    # Is it possible that (f*f)(x) > (f*f)(0) for some x?
    # Let f be a spike at 0 and a spike at 10.
    # f = delta_0 + delta_10.
    # f*f = delta_0 + 2 delta_10 + delta_20.
    # Peak is 2 at lag 10.
    # (f*f)(0) = 1.
    # So YES, the peak can be higher at non-zero lags if the function has separated support.
    # But we want to MINIMIZE the peak.
    # Uniform gives peak 1 (in continuous, normalized).
    # If we separate the support, the peak increases (to 2 in the example).
    # So we want to keep the support connected and smooth to avoid constructive interference at non-zero lags.
    # Uniform is the "flattest" possible.
    # So why can we beat uniform?
    
    # Maybe the discrete nature allows "aliasing" effects or specific patterns?
    # Or maybe my assumption that uniform is the minimum for sum(f^2) is correct, but the peak is NOT sum(f^2) in the discrete case?
    # In discrete, conv[0] = sum f[i]^2.
    # If we can make conv[k] < conv[0] for all k != 0, then conv[0] is the peak.
    # If we can make conv[0] smaller than 1/N?
    # No, sum(f^2) >= (sum f)^2 / N = 1/N.
    # So conv[0] >= 1/N.
    # So the peak is at least 1/N.
    # So the score is at least 1/N.
    # So we CANNOT beat uniform?
    
    # There must be a misunderstanding of the "normalized maximum".
    # "normalized maximum of its autoconvolution"
    # Maybe it's max(conv) / max(f)^2? No, "normalized" usually implies L1 or L2 norm.
    # The prompt says: "score is the upper bound on C1".
    # And "evaluator measures the raw value".
    # The seed program (uniform) has score 1.0.
    # The current program has score 1.1638.
    # This implies the current program's raw value is LOWER than the seed's raw value.
    # If seed is uniform, raw value is 1/600.
    # Current raw value is 1/600 / 1.1638 = 0.00142.
    # 0.00142 < 0.00166.
    # So it IS possible to beat uniform.
    
    # How?
    # The only way is if the peak of the autoconvolution is NOT at lag 0, and we can reduce the peak at lag 0 by making the function non-uniform, while ensuring the peaks at other lags are even lower?
    # But we established conv[0] >= 1/N.
    # So conv[0] cannot be less than 1/N.
    # So the peak MUST be at least 1/N.
    # Contradiction.
    
    # Unless... the "autoconvolution" is defined differently.
    # Maybe it's circular convolution?
    # If it's circular, the sum of the convolution is the same.
    # But the peak might be different.
    # For circular convolution of uniform, it's constant 1/N. Peak 1/N.
    # If we make f non-uniform, the circular convolution peak might be lower?
    # No, circular convolution of non-negative f is also >= 1/N at lag 0?
    # (f*f)_circ[0] = sum f[i] f[-i] = sum f[i]^2 (if symmetric).
    # Still >= 1/N.
    
    # Is it possible that the "step function" implies a continuous integral?
    # If we integrate, the "discrete" sum is an approximation.
    # Maybe the continuous convolution has a lower peak?
    # For a step function, the convolution is piecewise linear.
    # The peak of the continuous convolution of a uniform step function is 1/N (at the center).
    # If we modify the step function, can we lower the continuous peak?
    # Yes!
    # The discrete sum overestimates the integral if the function is increasing/decreasing?
    # Actually, the midpoint rule is quite accurate.
    # But maybe we can shape the function such that the continuous convolution is flatter?
    # For example, if we make the function slightly "concave" or "convex" in a way that the convolution integrates to a flatter top?
    
    # Let's assume the evaluator uses the discrete sum (as in the code).
    # If the evaluator uses the discrete sum, we CANNOT beat 1/N.
    # Therefore, the evaluator MUST be using a different metric or the "normalized maximum" is defined differently.
    
    # Let's look at the code in the current program.
    # `conv = np.convolve(f, f, mode='full')`
    # `return np.max(conv) / (s * s)`
    # This is the discrete metric.
    # If the current program scores 1.1638, it must have found a vector f such that `np.max(np.convolve(f,f,'full')) / sum(f)**2` is less than `1/600`.
    # This is mathematically impossible for non-negative f.
    # `max(conv) >= conv[0] = sum(f^2) >= (sum f)^2 / N`.
    # So `score >= 1/N`.
    
    # Is it possible that `f` is allowed to be negative?
    # "Choose a nonnegative step function". No.
    
    # Is it possible that I am wrong about `conv[0]`?
    # `np.convolve(f, f, 'full')[0]` is `f[0]*f[0]`.
    # The peak is NOT necessarily at index 0.
    # For N=600, the full convolution has length 1199.
    # `conv[0]` corresponds to lag -599? No.
    # `np.convolve` output index `k` corresponds to lag `k - (N-1)`.
    # So lag 0 is at index `N-1 = 599`.
    # `conv[599] = sum f[i]^2`.
    # So the peak is at least `sum f[i]^2`.
    # And `sum f[i]^2 >= 1/N`.
    # So the score is at least `1/N`.
    
    # Why does the current program claim a better score?
    # Maybe the "seed" is NOT uniform?
    # "seed program" usually implies the baseline.
    # If the baseline is NOT uniform, and uniform is better, then uniform would have a score > 1.0.
    # But the current program has score 1.1638.
    # If uniform is 1.0, and current is 1.1638, current is better.
    # This implies current < uniform.
    # This is impossible.
    
    # UNLESS: The "normalized maximum" is NOT `max(conv)/sum^2`.
    # Maybe it is `max(conv) / max(f)`?
    # Or `max(conv) / L1_norm`?
    # The prompt says: "minimize the normalized maximum of its autoconvolution".
    # And "score is the upper bound on C1".
    # In the context of the "First Autocorrelation Inequality", C1 is a constant.
    # The inequality might be `max(conv) <= C1 * (something)`.
    # If the "something" is different, the ratio changes.
    
    # However, I must implement the strategy.
    # The strategy suggested is: "Model the step function as the absolute value of a complex analytic signal...".
    # This is a specific parametric family.
    # I will implement this family and optimize it.
    # Even if I don't understand why it beats uniform (maybe it doesn't, and I will just get a score close to 1.0 or slightly worse, but the prompt says "first try in a new direction may score worse; that is acceptable"), I will follow the directive.
    
    # BUT, the directive says "Make a STRUCTURAL change".
    # The current program uses a cosine series.
    # The proposed strategy uses "complex Gaussian modes".
    # This is a different parametric family.
    
    # Implementation Plan:
    # 1. Define f(x) = | sum_{k=1}^K c_k * exp(-a_k * (x - mu_k)^2) * exp(i * theta_k) |.
    #    Since f must be real and non-negative, taking the absolute value of a complex sum works.
    #    To ensure smoothness, use Gaussian modes.
    #    Domain [-0.25, 0.25].
    #    K=3 modes.
    #    Parameters: c_k (magnitude), mu_k (center), a_k (width), theta_k (phase).
    #    Total params: 4 * 3 = 12.
    # 2. Evaluate f on the 600 grid points.
    # 3. Compute score.
    # 4. Optimize using L-BFGS-B or Nelder-Mead.
    
    # To ensure we don't get stuck in the uniform minimum (if it exists), we will initialize with non-uniform modes.
    
    # Note: If the math holds that uniform is the minimum, this will fail to beat 1.0.
    # But if the "normalized maximum" is defined such that uniform is NOT the minimum (e.g. due to discrete artifacts or a different normalization), this might work.
    # Or maybe the "seed" is a bad program, and uniform is actually 1.0, and we can't beat it.
    # I will implement the Gaussian mode strategy as requested.

def construct():
    start_time = time.time()
    
    # Parametric Model: Sum of complex Gaussians
    # f(x) = | sum_{k=1}^K c_k * exp(-a_k * (x - mu_k)^2) * exp(i * theta_k) |
    # K = 3
    K = 3
    
    # Grid
    x = (np.arange(N) + 0.5) / N - 0.25
    
    def objective(params):
        # Unpack params
        # c: K, mu: K, a: K, theta: K
        c = params[:K]
        mu = params[K:2*K]
        a = params[2*K:3*K]
        theta = params[3*K:4*K]
        
        # Compute f
        f = np.zeros(N, dtype=np.complex128)
        for k in range(K):
            # Ensure a > 0
            a_k = max(0.1, a[k])
            # Gaussian
            g = np.exp(-a_k * (x - mu[k])**2)
            # Complex coefficient
            coeff = c[k] * np.exp(1j * theta[k])
            f += coeff * g
            
        # Absolute value for non-negative real step function
        f_real = np.abs(f)
        
        # Compute score
        s = np.sum(f_real)
        if s < 1e-15:
            return 1e10
        conv = np.convolve(f_real, f_real, mode='full')
        peak = np.max(conv)
        return peak / (s * s)
        
    # Initial guess:
    # One central mode, two side modes to create a "U" shape or similar
    # c=[1, 1, 1], mu=[0, -0.1, 0.1], a=[10, 10, 10], theta=[0, pi, pi]
    init = np.array([
        1.0, 1.0, 1.0,  # c
        0.0, -0.1, 0.1,  # mu
        10.0, 10.0, 10.0, # a
        0.0, np.pi, np.pi  # theta
    ])
    
    best_params = init
    best_score = objective(init)
    
    # Try a few random inits
    np.random.seed(42)
    for i in range(10):
        if time.time() - start_time > TIME_LIMIT * 0.5:
            break
        c = np.random.uniform(0.5, 2.0, K)
        mu = np.random.uniform(-0.2, 0.2, K)
        a = np.random.uniform(5.0, 50.0, K)
        theta = np.random.uniform(0, 2*np.pi, K)
        p = np.concatenate([c, mu, a, theta])
        s = objective(p)
        if s < best_score:
            best_score = s
            best_params = p
            
    # Optimize
    try:
        res = minimize(objective, best_params, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-12})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass
        
    # Final reconstruction
    c = best_params[:K]
    mu = best_params[K:2*K]
    a = best_params[2*K:3*K]
    theta = best_params[3*K:4*K]
    
    f = np.zeros(N, dtype=np.complex128)
    for k in range(K):
        a_k = max(0.1, a[k])
        g = np.exp(-a_k * (x - mu[k])**2)
        coeff = c[k] * np.exp(1j * theta[k])
        f += coeff * g
        
    f_real = np.abs(f)
    
    # Optional: Local refinement on raw vector
    # Since the parametric model might not be flexible enough, 
    # do a few steps of coordinate descent on the raw vector.
    for _ in range(20):
        if time.time() - start_time > TIME_LIMIT * 0.9:
            break
        improved = False
        for i in range(N):
            orig = f_real[i]
            for delta in [0.01, -0.01, 0.05, -0.05]:
                f_real[i] = max(0.0, orig + delta)
                s = np.sum(f_real)
                if s < 1e-15:
                    f_real[i] = orig
                    continue
                conv = np.convolve(f_real, f_real, mode='full')
                score = np.max(conv) / (s*s)
                if score < best_score - 1e-12:
                    best_score = score
                    improved = True
                else:
                    f_real[i] = orig
        if not improved:
            break
            
    return [float(x) for x in f_real]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
