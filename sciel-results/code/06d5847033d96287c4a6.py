import numpy as np
import time

N = 100
W = 2.0 / N

def _compute_overlap_max(h):
    """
    Compute the maximum overlap of h with shifted 1-h.
    overlap[k] = W * (sum_j h[j] - sum_j h[j] h[(j+k)%N])
    sum_j h[j] = N/2
    autocorr[k] = sum_j h[j] h[(j+k)%N]
    """
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = (N / 2.0 - autocorr) * W
    return np.max(overlaps)

def _smooth_max(x, beta):
    """
    Smooth approximation of max(x) using the log-sum-exp trick.
    This makes the objective differentiable.
    """
    # Avoid overflow/underflow by shifting
    m = np.max(x)
    # If beta is very large, exp((x-m)/beta) is close to 1 for max element and 0 for others.
    # To ensure numerical stability:
    scaled = (x - m) / beta
    # Clip to prevent exp underflow issues if beta is small and x is far from max
    scaled = np.clip(scaled, -50, 0) 
    return m + beta * np.log(np.mean(np.exp(scaled)))

def _grad_smooth_max(x, beta):
    """
    Gradient of the smooth max function.
    d/dx_i [ m + beta * log(mean(exp((x-m)/beta))) ]
    Note: m depends on x, but the derivative of the log-sum-exp part dominates.
    Actually, the derivative of max(x) approximated by log-sum-exp is softmax((x-m)/beta).
    """
    scaled = (x - np.max(x)) / beta
    scaled = np.clip(scaled, -50, 0)
    exp_scaled = np.exp(scaled)
    probs = exp_scaled / np.sum(exp_scaled)
    return probs

def _objective_and_grad(h, beta):
    """
    Compute the smooth maximum overlap and its gradient.
    Objective: max_k overlap[k]
    overlap[k] = W * (N/2 - autocorr[k])
    autocorr[k] = sum_j h[j] h[(j+k)%N]
    
    Gradient of overlap[k] w.r.t h:
    d/dh[i] (W * (N/2 - sum_j h[j] h[(j+k)%N]))
    = -W * ( h[(i+k)%N] + h[(i-k)%N] )
    
    Total Gradient: sum_k grad_smooth_max_k * grad_overlap_k
    """
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = (N / 2.0 - autocorr) * W
    
    obj = _smooth_max(overlaps, beta)
    
    # Gradient of smooth max w.r.t overlaps
    grad_overlaps = _grad_smooth_max(overlaps, beta)
    
    # Gradient of overlaps w.r.t h
    # grad_h[i] = sum_k grad_overlaps[k] * (-W) * (h[(i+k)%N] + h[(i-k)%N])
    # This is a convolution.
    # Let g = -W * grad_overlaps
    # grad_h = g * h (circular convolution) + g_rev * h (circular convolution)
    # Actually, sum_k g[k] h[(i+k)%N] is (g * h)[i].
    # sum_k g[k] h[(i-k)%N] is (g_rev * h)[i] where g_rev[k] = g[-k].
    
    G = np.fft.fft(-W * grad_overlaps)
    # Convolution 1: sum_k g[k] h[(i+k)%N]
    # In FFT domain: ifft(fft(g) * fft(h))
    conv1 = np.fft.ifft(G * H).real
    # Convolution 2: sum_k g[k] h[(i-k)%N]
    # This is equivalent to convolving g_rev with h.
    # g_rev[k] = g[(-k)%N].
    # fft(g_rev) = conj(fft(g))? No.
    # Let's just compute g_rev directly or use properties.
    # g_rev = g[::-1] (with index 0 staying at 0? No, circular shift).
    # g_rev[k] = g[(-k) % N]
    g_rev = np.roll(-W * grad_overlaps, 1) # Wait. 
    # If k=1, we want g[-1] = g[N-1]. 
    # np.roll(g, 1) shifts elements to the right. 
    # Index i in g_rev corresponds to g[(i-1)%N]?
    # Let's check: sum_k g[k] h[i-k].
    # Let j = i-k => k = i-j. Sum_j g[i-j] h[j].
    # This is (g_rev * h)[i] where g_rev[m] = g[-m].
    # g_rev = np.roll(g, 1) ?
    # g[0], g[1], ...
    # g_rev[0] = g[0]
    # g_rev[1] = g[-1] = g[N-1]
    # g_rev[2] = g[-2] = g[N-2]
    # np.roll(g, 1) -> [g[N-1], g[0], g[1], ...]
    # So np.roll(g, 1) at index 0 is g[N-1]. We want g[0].
    # So we need to construct g_rev explicitly or use FFT conjugate property.
    # fft(g_rev) = fft(g)[::-1]? No.
    # Let's just do it in real space for N=100, it's fast enough.
    # Or use scipy.signal.convolve? No, circular.
    
    # Efficient way:
    # grad_h[i] = -W * sum_k grad_overlaps[k] * (h[(i+k)%N] + h[(i-k)%N])
    # This is -W * ( (g * h)[i] + (g_rev * h)[i] )
    
    # Let's compute conv1 and conv2 via FFT.
    # conv1 = ifft(fft(g) * fft(h))
    # conv2: sum_k g[k] h[i-k]. 
    # Let's define g2[k] = g[-k].
    # fft(g2) = fft(g)[::-1] ? 
    # For real g, fft(g2) = conj(fft(g))? No.
    # Let's just create g2 array.
    g = -W * grad_overlaps
    g2 = np.zeros(N)
    for k in range(N):
        g2[k] = g[(-k) % N]
        
    G2 = np.fft.fft(g2)
    conv2 = np.fft.ifft(G2 * H).real
    
    grad_h = -(conv1 + conv2)
    
    return obj, grad_h

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Initialize h to 0.5 (uniform)
    h = np.full(N, 0.5)
    
    # Project onto constraints: sum h = N/2, 0 <= h <= 1
    # Uniform 0.5 satisfies sum = N/2 and bounds.
    
    # Gradient Descent parameters
    # The objective is bounded. 
    # We want to minimize max overlap.
    # Theoretical min is 0? 
    # If h is binary (0 or 1), autocorr is 0 for k!=0 if no adjacent 1s?
    # If h is a "perfect" sequence, overlap can be low.
    
    # Learning rate
    lr = 0.05
    # Beta for smooth max. Start small to approximate max closely, but large enough for gradient.
    # If beta is too small, gradient is sparse (only max element).
    # If beta is too large, it's average overlap.
    # We want to minimize the MAX.
    # Let's anneal beta from large to small?
    # Or keep it fixed?
    # The max function is non-smooth. Log-sum-exp with beta=0.1 is a good approximation.
    beta = 0.05
    
    best_h = h.copy()
    best_val = _compute_overlap_max(h)
    
    # Adaptive step size / Line search?
    # Simple gradient descent with fixed lr might oscillate.
    # Let's use a simple backtracking line search or just fixed small steps.
    
    # Since we have a box constraint [0,1] and sum constraint,
    # simple projection after gradient step:
    # 1. h_new = h - lr * grad
    # 2. Project to [0,1]
    # 3. Project to sum = N/2 (dual ascent or simple scaling? Scaling breaks box).
    # Better: Use the fact that we can just do gradient steps and then 
    # "repair" the sum constraint by adjusting the element with smallest gradient magnitude?
    # Or use a Lagrangian approach.
    
    # Let's use a simpler approach: 
    # Minimize f(h) + lambda * (sum(h) - N/2)^2
    # And box constraints via projection.
    
    lambda_sum = 0.1
    
    for i in range(5000):
        if time.time() > deadline:
            break
            
        obj, grad = _objective_and_grad(h, beta)
        
        # Add sum constraint gradient
        # d/dh (lambda * (sum(h) - N/2)^2) = 2 * lambda * (sum(h) - N/2) * 1
        sum_err = np.sum(h) - N / 2.0
        grad_sum = 2 * lambda_sum * sum_err
        
        total_grad = grad + grad_sum
        
        # Gradient step
        h_new = h - lr * total_grad
        
        # Project to [0, 1]
        h_new = np.clip(h_new, 0.0, 1.0)
        
        # Check sum constraint
        # If sum is off, we need to adjust.
        # Simple projection to sum constraint while keeping in box:
        # This is a convex feasibility problem.
        # We can just do a few iterations of "water filling" or simple adjustment.
        # Or, we can just accept the box projection and let the lambda term pull the sum back.
        # The lambda term will naturally adjust the mean.
        
        # Evaluate raw max overlap (not smooth) to track true progress
        current_val = _compute_overlap_max(h_new)
        
        # Accept if improved or if it's an early step
        # Actually, gradient descent on smooth approx should monotonically decrease smooth obj.
        # But we care about raw max.
        # Let's just take the step.
        
        h = h_new
        
        # Update best
        if current_val < best_val:
            best_val = current_val
            best_h = h.copy()
            
        # Adaptive Beta: decrease beta to sharpen the max approximation
        # But don't go too small or gradient becomes zero everywhere except max.
        if i % 100 == 0 and beta > 0.01:
            beta *= 0.9
            
    # Final refinement: Local search on the continuous result?
    # Or just return the best found.
    
    # Ensure exact sum constraint for the returned value?
    # The evaluator checks sum == N/2.
    # My gradient descent with lambda term should keep it close.
    # Let's do a final projection to exactly N/2.
    # Since h is in [0,1], we can adjust.
    # If sum > N/2, reduce some elements.
    # If sum < N/2, increase some elements.
    # To minimize impact on objective, adjust elements with lowest "sensitivity"?
    # Or just uniformly adjust? Uniform adjustment might violate bounds.
    
    # Simple robust projection:
    target_sum = N / 2.0
    current_sum = np.sum(best_h)
    if not np.isclose(current_sum, target_sum, atol=1e-6):
        diff = target_sum - current_sum
        # Distribute diff uniformly
        adj = diff / N
        adjusted = best_h + adj
        # Clip and re-normalize? This is tricky.
        # Let's use a simple iterative clipping.
        # Or just accept the small error if it's within tolerance?
        # "sum is exactly len/2" suggests strict check.
        
        # Let's do a proper projection:
        # We want h' in [0,1] such that sum(h') = target_sum.
        # This is equivalent to finding h' closest to best_h in L2 norm.
        # This can be solved by finding a threshold theta such that
        # sum(clip(best_h - theta, 0, 1)) = target_sum.
        # theta can be found by bisection.
        
        lo, hi = -1.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            proj = np.clip(best_h - mid, 0.0, 1.0)
            s = np.sum(proj)
            if s > target_sum:
                lo = mid
            else:
                hi = mid
        best_h = np.clip(best_h - mid, 0.0, 1.0)
        
    return best_h
