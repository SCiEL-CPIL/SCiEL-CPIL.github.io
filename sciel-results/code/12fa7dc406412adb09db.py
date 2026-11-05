"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _value(h):
    """Compute the raw task value (lower is better)."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s * s))

def _objective(x, beta):
    """
    Smooth surrogate objective using log-sum-exp for the max.
    We want to minimize max(conv), so we use a soft-max.
    Also, we need to handle the normalization by sum(h)^2.
    
    Let's define the objective as:
    f(x) = soft_max(conv(x)) / sum(x)^2
    
    To make it differentiable and stable, we can minimize:
    log(soft_max(conv)) - 2*log(|sum(x)|)
    
    However, log-sum-exp is for max. 
    Let M = max(conv). 
    logsumexp(conv) ≈ M + log(sum(exp(conv - M)))
    
    Actually, the task is:
    value = abs(2 * N * max(conv) / sum(h)^2)
    
    We want to minimize this.
    Let's assume sum(h) > 0 for simplicity, or handle sign.
    Since we can scale h, let's fix the scale during optimization to avoid singularity at sum=0.
    Or better, use a parameterization that ensures sum != 0.
    
    Let's try minimizing:
    g(x) = log(soft_max(conv(x))) - 2 * log(abs(sum(x)))
    
    This is equivalent to minimizing soft_max(conv(x)) / sum(x)^2 (up to constants).
    Wait, log(a/b) = log(a) - log(b).
    So minimizing log(soft_max(conv)) - 2*log(sum(x)) is minimizing soft_max(conv)/sum(x)^2.
    
    We need to be careful with the sign of sum(x).
    If sum(x) < 0, sum(x)^2 is still positive.
    But log(abs(sum(x))) is fine.
    
    However, if max(conv) is negative, the value is |...|, so it's -max(conv)/sum^2.
    If max(conv) is positive, it's max(conv)/sum^2.
    
    Let's check the seed: h=ones(400).
    conv is triangular, max is 400 (at center).
    sum is 400.
    value = 2*400*400 / 400^2 = 2.
    
    Can we get max(conv) negative?
    If h has negative values, conv can be negative.
    If max(conv) is negative, then abs(2*N*max(conv)/sum^2) = 2*N*(-max(conv))/sum^2.
    So we want max(conv) to be as close to 0 as possible, ideally negative if possible?
    No, if max(conv) is negative, the numerator is negative, abs makes it positive.
    So we want max(conv) to be small in magnitude?
    Actually, if max(conv) is negative, say -1, then value = 2*N*1/sum^2.
    If max(conv) is positive, say 1, then value = 2*N*1/sum^2.
    So we want max(conv) to be close to 0.
    
    Wait, can max(conv) be 0?
    If h is such that conv is always <= 0, then max(conv) <= 0.
    Then value = 2*N * |max(conv)| / sum^2.
    If we can make max(conv) very negative, the value is large.
    So we want max(conv) to be close to 0 from below? Or just small in magnitude.
    
    Let's use the soft-max surrogate for the max.
    soft_max(z, beta) = (1/beta) * log(sum(exp(beta * z)))
    
    If we minimize soft_max(conv), we are pushing the max down.
    But we also have the denominator sum(h)^2.
    
    Let's try to minimize:
    f(x) = soft_max(conv(x), beta) - 2 * log(abs(sum(x)))
    
    This encourages the max convolution coefficient to be small relative to the square of the sum.
    
    We need to handle the case where sum(x) is close to 0.
    We can add a small penalty or constrain the optimization.
    Alternatively, we can normalize x in the objective to have unit sum, and then scale?
    No, the value is scale invariant?
    value = 2*N*max(conv(h)) / sum(h)^2
    If we scale h by c, conv scales by c^2, sum scales by c.
    value(c*h) = 2*N*c^2*max(conv(h)) / (c*sum(h))^2 = value(h).
    So the value is scale invariant.
    
    This means we can fix the scale of h during optimization to improve numerical stability.
    Let's fix sum(h) = 1.
    Then value = 2*N*max(conv(h)).
    We just need to minimize max(conv(h)) subject to sum(h)=1.
    
    This is a much simpler problem!
    Minimize max(conv(h)) subject to sum(h)=1.
    
    Let's use L-BFGS-B with a constraint sum(h)=1?
    Or re-parameterize: h = base + z, where sum(z)=0.
    Or simply use a penalty for sum(h)-1.
    
    Let's try:
    Objective: soft_max(conv(h), beta) + lambda * (sum(h) - 1)^2
    
    If we can make max(conv(h)) negative, the value is 2*N * |max(conv(h))|.
    So we want max(conv(h)) to be as close to 0 as possible, ideally negative?
    No, if max(conv(h)) is negative, the value is positive.
    We want the value to be small.
    So we want max(conv(h)) to be close to 0.
    
    Wait, if max(conv(h)) is negative, say -1, then value = 2*N*1 = 800.
    If max(conv(h)) is positive, say 1, then value = 2*N*1 = 800.
    If max(conv(h)) is 0.001, value = 0.8.
    
    So we want max(conv(h)) to be close to 0.
    
    Let's refine the objective:
    We want to minimize |max(conv(h))| subject to sum(h)=1.
    
    Using soft-max for max(conv(h)):
    s_max = soft_max(conv(h), beta)
    
    If s_max is close to 0, we are good.
    But soft-max is always positive if beta > 0?
    No, soft-max(z, beta) = (1/beta) log(sum(exp(beta*z))).
    If all z are very negative, soft-max is very negative.
    
    So we can just minimize soft_max(conv(h), beta).
    If the true max is negative, soft-max will be negative.
    If the true max is positive, soft-max will be positive.
    Minimizing soft-max will push the max towards -infinity?
    No, we want it to be close to 0.
    
    Wait, if we minimize soft-max, we are minimizing the max.
    So we are trying to make the max as small (negative) as possible.
    But if the max becomes very negative, the value 2*N*|max| becomes large.
    
    So we have a trade-off.
    We want max(conv) to be close to 0.
    
    Let's try minimizing (soft_max(conv) - target)^2?
    No, that's not smooth at the optimum if target is 0? It is smooth.
    
    Let's try minimizing:
    f(h) = (soft_max(conv(h), beta))^2 + lambda * (sum(h) - 1)^2
    
    This encourages the max convolution coefficient to be close to 0.
    
    Let's test this hypothesis.
    
    """
    conv = np.convolve(x, x)
    # Soft max
    beta = 10.0
    # Shift for numerical stability
    max_conv = np.max(conv)
    if beta * max_conv > 700: # Avoid overflow
        log_sum_exp = max_conv + (1.0/beta) * np.log(np.sum(np.exp(beta * (conv - max_conv))))
    else:
        log_sum_exp = (1.0/beta) * np.log(np.sum(np.exp(beta * conv)))
    
    s_max = log_sum_exp
    sum_h = np.sum(x)
    penalty = 1000.0 * (sum_h - 1.0)**2
    return s_max**2 + penalty

def _objective_grad(x, beta):
    """Gradient of the objective."""
    conv = np.convolve(x, x)
    max_conv = np.max(conv)
    if beta * max_conv > 700:
        exp_conv = np.exp(beta * (conv - max_conv))
        log_sum_exp = max_conv + (1.0/beta) * np.log(np.sum(exp_conv))
    else:
        exp_conv = np.exp(beta * conv)
        log_sum_exp = (1.0/beta) * np.log(np.sum(exp_conv))
    
    s_max = log_sum_exp
    
    # Gradient of s_max w.r.t conv
    # d/dconv_i [ (1/beta) log(sum(exp(beta*conv))) ] = exp(beta*conv_i) / sum(exp(beta*conv))
    if beta * max_conv > 700:
        grad_conv = exp_conv / np.sum(exp_conv)
    else:
        grad_conv = exp_conv / np.sum(exp_conv)
    
    # Gradient of conv w.r.t x
    # conv = np.convolve(x, x)
    # d conv_k / d x_i = x_{k-i} + x_{k-i} ? No.
    # conv_k = sum_j x_j x_{k-j}
    # d conv_k / d x_i = x_{k-i} + x_{k-i} = 2 x_{k-i} (if k-i is in range)
    # Actually, np.convolve(x, x) full mode:
    # conv[k] = sum_{j=0}^{N-1} x[j] x[k-j]
    # d conv[k] / d x[i] = x[k-i] + x[k-i] ?
    # Let's check.
    # If we change x[i] by delta, conv[k] changes by x[k-i]*delta + x[k-i]*delta?
    # No, the term x[j] x[k-j] involves x[i] if j=i or k-j=i.
    # If j=i, term is x[i] x[k-i]. Derivative w.r.t x[i] is x[k-i].
    # If k-j=i, i.e., j=k-i, term is x[k-i] x[i]. Derivative w.r.t x[i] is x[k-i].
    # So d conv[k] / d x[i] = x[k-i] + x[k-i] = 2 x[k-i] (if indices are valid).
    # Note: x[k-i] is 0 if k-i is out of [0, N-1].
    
    # We can compute grad_x = 2 * np.convolve(grad_conv, x, mode='full')?
    # Let's verify.
    # grad_x[i] = sum_k grad_conv[k] * d conv[k] / d x[i]
    # = sum_k grad_conv[k] * 2 x[k-i]
    # = 2 * sum_k grad_conv[k] x[k-i]
    # This is 2 * (grad_conv * x)[i] in convolution sense?
    # (a * b)[i] = sum_k a[k] b[i-k]
    # Here we have sum_k grad_conv[k] x[k-i].
    # Let j = k-i => k = i+j.
    # sum_j grad_conv[i+j] x[j].
    # This is the cross-correlation or convolution with reversed indices?
    # np.convolve(a, b)[i] = sum_j a[j] b[i-j].
    # If we do np.convolve(grad_conv, x)[i] = sum_j grad_conv[j] x[i-j].
    # Let k = i-j => j = i-k.
    # sum_k grad_conv[i-k] x[k].
    # This is not quite sum_k grad_conv[k] x[k-i].
    
    # Let's just use the direct definition for gradient calculation to be safe, or use a known formula.
    # Actually, gradient of sum_k c_k conv[k] w.r.t x[i] is 2 * (c * x)[i] where * is correlation?
    # Let's use: grad_x = 2 * np.convolve(grad_conv, x, mode='full')[N-1 : 2*N-1] ?
    # Let's test with small N.
    
    # For now, let's approximate or use a simpler gradient if possible.
    # Or just use finite differences for the gradient in L-BFGS-B? No, that's slow.
    # Let's derive carefully.
    # f(x) = g(conv(x))
    # grad_x = J^T grad_conv
    # J_{k,i} = d conv[k] / d x[i] = x[k-i] + x[k-i] (if valid)
    # So grad_x[i] = sum_k grad_conv[k] (x[k-i] + x[k-i] if valid)
    # = 2 * sum_k grad_conv[k] x[k-i]
    
    # Let's compute this efficiently.
    # We can use FFT or just loop if N=400 is small enough?
    # N=400, conv length 799.
    # Looping over i (400) and summing over k (799) is 400*799 ~ 300k ops. Very fast.
    
    grad_x = np.zeros(N)
    for i in range(N):
        # sum_k grad_conv[k] x[k-i]
        # k-i must be in [0, N-1] => i <= k <= i+N-1
        k_min = max(0, i)
        k_max = min(len(grad_conv)-1, i+N-1)
        if k_min <= k_max:
            # x[k-i] for k in [k_min, k_max]
            # indices of x: k-i
            # k_min-i to k_max-i
            idx_start = k_min - i
            idx_end = k_max - i + 1
            grad_x[i] = 2.0 * np.dot(grad_conv[k_min:k_max+1], x[idx_start:idx_end])
    
    # Gradient of penalty: lambda * (sum(x)-1)^2
    # d/dx_i = 2 * lambda * (sum(x)-1)
    lambda_pen = 1000.0
    sum_h = np.sum(x)
    grad_penalty = 2.0 * lambda_pen * (sum_h - 1.0)
    
    # Gradient of s_max^2: 2 * s_max * grad_s_max
    grad_obj = 2.0 * s_max * grad_x + grad_penalty
    
    return grad_obj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = None
    best_val = np.inf
    
    # Initial guesses
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Random
    np.random.seed(42)
    for _ in range(5):
        x = np.random.randn(N)
        x = x / np.sum(x)
        inits.append(x)
        
    # 3. Structured: Bumps
    for center in [N//4, N//2, 3*N//4]:
        x = np.zeros(N)
        x[center-10:center+10] = 1.0
        x = x / np.sum(x)
        inits.append(x)
        
    # 4. Fourier modes
    for freq in [1, 2, 5]:
        x = np.cos(2 * np.pi * freq * np.arange(N) / N)
        x = x / np.sum(x)
        inits.append(x)
        
    for x0 in inits:
        if time.time() > deadline:
            break
            
        try:
            # Optimize
            res = minimize(
                _objective, 
                x0, 
                method='L-BFGS-B', 
                jac=_objective_grad, 
                options={'maxiter': 1000, 'maxfun': 10000}
            )
            
            h_opt = res.x
            val = _value(h_opt)
            
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
                
        except Exception as e:
            continue
    
    # Final polish: exact objective minimization using the best h as start
    if best_h is not None and time.time() < deadline - 5.0:
        try:
            # Use a higher beta for sharper max approximation
            # Or just use the exact objective with subgradient? No, L-BFGS needs gradient.
            # Let's try a few more iterations with the best h.
            
            # We can also try to refine with a local search on the exact objective?
            # Since the exact objective is non-smooth, gradient methods might struggle.
            # But L-BFGS-B with the smooth surrogate is a good approximation.
            
            # Let's try to increase beta in the objective for the final polish.
            # We can't easily change beta in the function without modifying it.
            # Let's just run minimize again with the best h.
            
            res = minimize(
                _objective, 
                best_h, 
                method='L-BFGS-B', 
                jac=_objective_grad, 
                options={'maxiter': 500, 'maxfun': 5000}
            )
            
            h_polish = res.x
            val_polish = _value(h_polish)
            
            if val_polish < best_val:
                best_val = val_polish
                best_h = h_polish.copy()
                
        except Exception:
            pass
    
    if best_h is None:
        return np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
