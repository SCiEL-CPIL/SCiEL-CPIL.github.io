"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv(x):
    """Compute linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    size2 = 1
    while size2 < size:
        size2 *= 2
    F = fft(x, size2)
    A = ifft(F * np.conj(F)).real
    return A[:size]

def _score(x):
    """Compute max(|autoconv|) / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    A = _autoconv(x)
    m = np.max(np.abs(A))
    return m / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    # Initialize with a smooth sinusoidal profile
    # This distributes autoconvolution energy across lags rather than concentrating at 0
    x = np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False))
    
    # Normalize to a reasonable magnitude
    x = x * 10.0
    
    best_x = x.copy()
    best_score = _score(best_x)
    
    # Gradient Descent on the continuous signal
    # Objective: J(x) = max(|A(x)|) / (sum(x))^2
    # We approximate the gradient of max(|A|) using the subgradient at the peak
    # and normalize by the integral term.
    
    learning_rate = 0.5
    momentum = np.zeros_like(x)
    beta = 0.9
    
    for step in range(20000):
        if time.time() > deadline - 5.0:
            break
        
        s = np.sum(x)
        if abs(s) < 1e-6:
            # If sum is near zero, the score is undefined/huge. Push away from zero.
            # Add a small bias to ensure sum is non-zero
            x += 0.1 * np.sign(np.sum(x) + 0.01)
            s = np.sum(x)
            
        A = _autoconv(x)
        abs_A = np.abs(A)
        max_val = np.max(abs_A)
        score = max_val / (s * s)
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
            
        # Compute gradient of J(x)
        # J(x) = M / S^2 where M = max(|A|) and S = sum(x)
        # dJ/dx = (1/S^2) * dM/dx - 2*M/S^3 * dS/dx
        # dS/dx = 1 (vector of ones)
        
        # dM/dx: The gradient of the max absolute value of the autoconvolution.
        # A[k] = sum_i x[i] x[i+k]
        # dA[k]/dx[j] = x[j+k] + x[j-k] (with appropriate bounds)
        # We only care about the lag k* where |A[k*]| is maximum.
        
        k_star = np.argmax(abs_A)
        
        # Compute gradient contribution from the peak at k_star
        # grad_A[k_star] is a vector g where g[j] = x[j+k_star] + x[j-k_star]
        # We need to handle boundaries.
        
        g = np.zeros_like(x)
        # x[j+k] for j in [0, N-1-k]
        if k_star < N:
            g[:N-k_star] += x[k_star:]
            # x[j-k] for j in [k, N-1]
            g[k_star:] += x[:N-k_star]
            
        # Sign of the max value
        sign_peak = np.sign(A[k_star])
        
        # Gradient of M = |A[k_star]| wrt x is sign_peak * g
        grad_M = sign_peak * g
        
        # Gradient of S^2 = 2*S*sum(x) -> derivative is 2*S*1
        # Term 1: (1/S^2) * grad_M
        term1 = grad_M / (s * s)
        
        # Term 2: -2*M/S^3 * 1
        term2 = -2 * max_val / (s**3) * np.ones_like(x)
        
        grad_J = term1 + term2
        
        # Normalize gradient to prevent divergence
        norm_grad = np.linalg.norm(grad_J)
        if norm_grad > 0:
            grad_J = grad_J / norm_grad
        
        # Update with momentum
        momentum = beta * momentum + (1 - beta) * grad_J
        x -= learning_rate * momentum
        
        # Constrain x to a bounded interval to keep it "step-like"
        # We want to encourage values near +/- some level, but allow intermediate values during optimization.
        # Let's clip to [-10, 10]
        x = np.clip(x, -10.0, 10.0)
        
        # Periodic annealing of learning rate
        if step % 500 == 0:
            learning_rate *= 0.95
            if learning_rate < 0.01:
                learning_rate = 0.01

    # Final quantization
    # The problem asks for "signed step heights". The seed used +/- 1.
    # However, the interface says "sequence of signed step heights".
    # The score is max(|autoconv|) / (sum)^2.
    # If we scale the entire sequence by a constant c, autoconv scales by c^2 and sum scales by c, so sum^2 scales by c^2.
    # The ratio is invariant to global scaling!
    # Therefore, we can choose any scale.
    # But "step functions" usually implies discrete levels.
    # Let's try to map the continuous x to a set of discrete levels.
    # The seed used +/- 1.
    # Let's try to quantize to the nearest integer, or +/- 1, or a few levels.
    
    # Option 1: Binary +/- 1 based on sign
    x_bin = np.sign(x)
    # Ensure no zeros (if any, set to 1)
    x_bin[x_bin == 0] = 1.0
    
    # Option 2: Keep as integers (round)
    x_int = np.round(x)
    
    # Option 3: Keep as is (continuous)
    # The evaluator might expect specific types? "sequence of signed step heights".
    # The seed returned floats.
    # Let's evaluate the scores of these variants and pick the best.
    
    candidates = [
        x_bin,
        x_int,
        x,
        np.sign(x) * 10.0, # Large amplitude binary
        np.clip(x, -5, 5),
        np.clip(x, -1, 1)
    ]
    
    best_candidate = None
    best_candidate_score = np.inf
    
    for c in candidates:
        sc = _score(c)
        if sc < best_candidate_score:
            best_candidate_score = sc
            best_candidate = c
            
    # If the continuous version is best, we can try to refine it further with a few more steps
    if best_candidate is x and best_candidate_score < best_score:
         # It improved, so we keep it.
         pass
         
    return best_candidate

# EVOLVE-BLOCK-END
