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
    # Autoconvolution (linear)
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Strategy: Projected Gradient Descent on Smoothed Max Autoconvolution
    # Objective: Minimize J_alpha(h) = (1/alpha) * log( sum_k exp(alpha * (h*h)_k) )
    # Subject to: h >= 0, sum(h) = 1
    
    # Parameters
    alpha = 10.0  # Smoothing parameter. Larger = sharper peak approximation.
    # We start with a Two-Block initial guess which is known to be better than Uniform.
    # Two blocks of width W=N/3, centered at N/4 and 3N/4?
    # Let's try blocks at [0, N/3] and [N/2, 2N/3]?
    # Actually, let's just use a "flat top with edges" or "Two Blocks".
    
    # Initial Guess: Two Blocks of width N//3
    W = N // 3
    h = np.zeros(N)
    # Block 1: 0 to W
    # Block 2: N-W to N
    # This creates a periodic-like structure?
    # Let's try blocks at 0 and N/2.
    h[:W] = 1.0
    h[N//2:N//2+W] = 1.0
    
    # Normalize to sum=1
    s = np.sum(h)
    h = h / s
    
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Define the smoothed objective and its gradient
    def smoothed_objective_and_grad(h):
        # h is normalized to sum=1, h>=0
        conv = np.convolve(h, h, mode='full')
        # To avoid overflow in exp, shift conv by its max
        conv_max = np.max(conv)
        exp_conv = np.exp(alpha * (conv - conv_max))
        log_sum_exp = np.log(np.sum(exp_conv)) + alpha * conv_max
        J = log_sum_exp / alpha
        
        # Gradient of J w.r.t h
        # d/dh_k log(sum exp(alpha c_i)) = (alpha * sum_i exp(alpha c_i) * dc_i/dh_k) / sum exp(alpha c_i)
        # c_i = sum_j h_j h_{j+i-k} ? No, c_i = sum_j h_j h_{j+i}.
        # dc_i/dh_k = h_{i-k} + h_{k-i} (if indices valid)
        # This is equivalent to 2 * (h * h) shifted?
        # Actually, grad_c = 2 * np.convolve(h, h, mode='full')? No.
        # The gradient of the quadratic form c_i(h) = h^T A_i h is (A_i + A_i^T)h.
        # A_i is a shift matrix. A_i + A_i^T is symmetric shift.
        # So grad_c = 2 * h * h (circular? No, linear).
        # Let's compute gradient via finite difference for safety? No, N=600 is small enough for analytic.
        # Analytic Gradient:
        # G = np.zeros_like(conv)
        # For each k in h, it contributes to conv[i] where i = j + (k-j) ?
        # conv[i] = sum_j h_j h_{j+i}.
        # d conv[i] / d h_k = h_{i-k} + h_{k-i} (if 0 <= idx < N)
        # This is exactly the autoconvolution of h with itself, but we need the gradient vector w.r.t h.
        # Let's use the property: grad_h (h*h) = 2 * (h * h) ? No, that's circular.
        # For linear convolution:
        # grad = np.zeros(N)
        # for k in range(N):
        #    for i in range(2N-1):
        #       if 0 <= i-k < N: grad[k] += alpha * h[i-k] * exp_term
        #       if 0 <= k-i < N: grad[k] += alpha * h[k-i] * exp_term
        # This is O(N^2). N=600 -> 360,000 ops. Fast.
        
        exp_term = exp_conv / np.sum(exp_conv) # Weights
        grad = np.zeros(N)
        # Vectorized computation of gradient
        # conv[i] = sum_j h_j h_{j+i}
        # d conv[i] / d h_k = h_{i-k} + h_{k-i}
        # We can compute this by correlating h with the "weights" of the conv terms?
        # Let's just do the O(N^2) loop or use FFT?
        # FFT is faster.
        # Grad of sum_i w_i (h*h)_i = 2 * (h * (w * h)) ?
        # Let's use finite difference for the first few steps to verify? No, let's use analytic.
        # The gradient of the linear convolution C = h * h is:
        # grad_h C = 2 * (h * h) in the circular sense? No.
        # Actually, the Jacobian of h*h w.r.t h is a linear operator.
        # Let's compute it via:
        # grad = np.zeros(N)
        # for i in range(2*N-1):
        #    w = exp_term[i]
        #    # Contribution to grad[k] from conv[i]
        #    # conv[i] depends on h[k] if h[k] is part of the sum.
        #    # h[k] * h[i-k] and h[i-k] * h[k]
        #    # So grad[k] += w * (h[i-k] + h[i-k]) = 2*w*h[i-k] if valid.
        #    # This is equivalent to 2 * (h * w) ?
        #    # Let's check.
        #    # Sum_i w_i h[i-k] = (w * h)[k] ?
        #    # Yes, if we align them correctly.
        #    # conv is length 2N-1. h is length N.
        #    # Let's pad w to length 2N-1.
        #    # The term h[i-k] is non-zero if 0 <= i-k < N.
        #    # This is a cross-correlation.
        #    # grad = 2 * np.convolve(h, w, mode='same')? No, w is the weight of the conv output.
        #    # Let's just use the O(N^2) loop, it's safe and fast enough for N=600.
        
        # Optimization: Use FFT for the gradient if possible, but O(N^2) is ~0.36M ops, which is < 1ms in Python if vectorized?
        # Python loop is slow. Let's vectorize.
        # Create a matrix M where M[i, k] = 1 if h[k] contributes to conv[i].
        # This is too memory heavy.
        # Let's use the fact that grad = 2 * (h * w) ?
        # Let's test with a small example mentally.
        # h = [1, 1]. conv = [1, 2, 1].
        # w = [1, 1, 1].
        # grad[0]: d c0/dh0 = h0 = 1. d c1/dh0 = h1 = 1. d c2/dh0 = 0.
        # Sum = 1*1 + 1*1 = 2.
        # Formula 2 * (h * w)?
        # h * w (linear) = [1, 1+1, 1+1, 1] = [1, 2, 2, 1].
        # Take first N elements? [1, 2].
        # 2 * [1, 2] = [2, 4].
        # Actual grad[0] = 2. grad[1] = 2.
        # So it's not simply 2 * (h*w).
        
        # Correct Gradient Formula for Linear Convolution:
        # grad[k] = 2 * sum_i w_i * (h[i-k] if 0<=i-k<N else 0 + h[k-i] if 0<=k-i<N else 0)
        # This is 2 * (w * h) + 2 * (w_rev * h)?
        # Let's just use the O(N^2) vectorized approach with broadcasting if memory allows, or a simple loop.
        # N=600. Loop of 600. Inside, vector ops of size 600.
        # 600 * 600 = 360,000 ops. In Python, a loop of 600 iterations with numpy ops is fast (~1-5ms).
        
        for k in range(N):
            # Indices i such that i-k is in [0, N-1] -> i in [k, k+N-1]
            # Indices i such that k-i is in [0, N-1] -> i in [k-N+1, k]
            
            # Part 1: h[i-k]
            # i ranges from k to k+N-1
            # w indices: k to k+N-1
            # h indices: 0 to N-1
            # conv_term1 = np.dot(w[k:k+N], h)
            
            # Part 2: h[k-i]
            # i ranges from max(0, k-N+1) to k
            # w indices: max(0, k-N+1) to k
            # h indices: k - max(0, k-N+1) to 0 (reversed)
            
            # Simpler: Just compute the full convolution gradient using FFT?
            # No, let's use the property that the gradient of the quadratic form is linear.
            # We can compute the gradient by:
            # grad = 2 * np.convolve(h, w, mode='full')? No.
            
            # Let's use a robust finite difference for the gradient?
            # N=600. 600 evaluations of objective (O(N^2) each) = 360,000 * 600 = 216M ops. Too slow.
            
            # Let's stick to the analytic O(N^2) loop. It is the safest.
            # To speed up, we can precompute the slices of w.
            pass

        # Actually, let's use a simpler approach: 
        # The problem is convex enough that we can use `scipy.optimize.minimize` with `SLSQP` 
        # on the **original** non-smooth problem? No, SLSQP needs gradients.
        
        # Let's use **Subgradient Method** with a simple projection.
        # Subgradient of max(c) is any convex combination of the gradients of the active terms.
        # Active terms are those close to the max.
        # This is stochastic subgradient.
        
        # Given the time constraint and the need for a structural change, let's try 
        # **Optimizing the Parameters of a "Multi-Sine" Wave**.
        # h_i = 1 + sum_{m=1}^M A_m cos(2 pi m i / N + phi_m)
        # This ensures h is smooth and non-negative (if A is small).
        # We optimize A and phi.
        # This reduces dimension from 600 to 2M.
        # M=5 -> 10 parameters.
        # This is a "Spectral Shaping" but with **Phases** allowed (current program fixed phase to 0).
        # Allowing phases allows for asymmetric shapes and "nulls" in the spectrum that can flatten the autoconvolution.
        
        M = 5
        def params_to_h(params):
            # params: [A1, phi1, A2, phi2, ..., AM, phiM]
            A = params[0::2]
            phi = params[1::2]
            i = np.arange(N)
            h = np.ones(N)
            for m in range(M):
                h += A[m] * np.cos(2 * np.pi * m * i / N + phi[m])
            # Project to non-negative
            h = np.maximum(0, h)
            s = np.sum(h)
            if s < 1e-10:
                return np.ones(N)/N
            return h / s

        def objective_multi_sine(params):
            h = params_to_h(params)
            return compute_score(h)

        # Initial guess: Uniform (all A=0)
        params0 = np.zeros(2*M)
        
        # Use L-BFGS-B to optimize parameters
        # This is a low-dimensional non-convex optimization, but L-BFGS-B is fast.
        
        best_score = compute_score(np.ones(N)/N)
        best_params = params0.copy()
        
        try:
            res = optimize.minimize(
                objective_multi_sine,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

        # Try a few random restarts for phases
        if time.time() < deadline - 20:
            for i in range(5):
                if time.time() > deadline:
                    break
                # Random small amplitudes
                A_rand = np.random.uniform(-0.5, 0.5, M)
                phi_rand = np.random.uniform(0, 2*np.pi, M)
                params_rand = np.zeros(2*M)
                params_rand[0::2] = A_rand
                params_rand[1::2] = phi_rand
                
                try:
                    res = optimize.minimize(
                        objective_multi_sine,
                        params_rand,
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-10}
                    )
                    if res.fun < best_score:
                        best_score = res.fun
                        best_params = res.x
                except Exception:
                    pass

        h_opt = params_to_h(best_params)
        return h_opt.tolist()
# EVOLVE-BLOCK-END
