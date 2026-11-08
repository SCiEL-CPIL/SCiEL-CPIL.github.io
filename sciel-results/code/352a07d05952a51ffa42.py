"""
Strategy: Frequency-Domain Minimax Optimization with Soft Constraints.

The task is to minimize the maximum overlap of h with shifted 1-h.
Let g(x) = 1 - h(x).
Overlap at shift k: O_k = sum_i h_i g_{i+k}.
In the frequency domain, the circular cross-correlation C_k = sum h_i g_{i+k} 
can be computed via IFFT(FFT(h) * conj(FFT(g))).
We want to minimize max_k C_k.

The current approach uses discrete binary search (Simulated Annealing).
This new approach uses continuous optimization in the time domain but 
guided by the frequency domain structure, or more directly, optimizes 
the spectral envelope.

Actually, a more direct "different design family" is to use a 
continuous relaxation with gradient descent, specifically targeting 
the minimax objective. The objective is non-smooth due to max(), 
so we use a soft-max approximation (log-sum-exp) to make it differentiable.

Constraints:
1. sum(h) = N/2
2. 0 <= h_i <= 1

We will use projected gradient descent or a penalty method.
Given the "different strategy" requirement, let's try optimizing the 
DFT coefficients directly? No, time-domain projection is easier for 
box constraints.

Let's use a "Spectral Minimax" approach:
1. Parameterize h as a continuous vector.
2. Objective: L = log-sum-exp(beta * C) / beta, where C is the vector of overlaps.
3. Gradient descent with projection onto the feasible set {h | sum(h)=N/2, 0<=h<=1}.
4. To encourage a binary-like solution (which is often optimal for such integer-constrained 
   relaxations), we can add a small entropy penalty or just rely on the geometry.

However, the prompt suggests: "Model the problem as a minimax optimization over the 
discrete Fourier transform (DFT) magnitude spectrum... allowing direct minimization 
of the spectral envelope".

Let's stick to a robust continuous optimization using Adam optimizer on the 
log-sum-exp of the overlaps, with strict projection to maintain constraints.
This is fundamentally different from discrete SA: it's continuous, gradient-based, 
and uses a smooth approximation of the max operator.
"""

import numpy as np
import time

N = 100

def _project(h):
    """
    Project h onto the set {x | sum(x) = N/2, 0 <= x_i <= 1}.
    This is done by finding a threshold t such that sum(clamp(h_i - t, 0, 1)) = N/2.
    """
    target_sum = N / 2.0
    
    # If h is already close, just clip and adjust
    # We need to solve sum(min(max(h_i - t, 0), 1)) = target_sum
    
    # Sort h to find the threshold efficiently if needed, but N=100 is small.
    # Simple bisection on t.
    # min possible sum is 0 (if t is large), max is N (if t is small).
    
    low = np.min(h) - 1.0
    high = np.max(h) + 1.0
    
    # Check if target is reachable
    # If target_sum < 0 or > N, infeasible, but we assume valid input.
    
    for _ in range(50):
        t = (low + high) / 2.0
        h_proj = np.clip(h - t, 0.0, 1.0)
        s = np.sum(h_proj)
        if s > target_sum:
            low = t
        else:
            high = t
            
    t = (low + high) / 2.0
    h_proj = np.clip(h - t, 0.0, 1.0)
    return h_proj

def _compute_overlaps(h):
    """
    Compute the vector of circular overlaps O_k = sum_i h_i (1 - h_{i+k}).
    """
    g = 1.0 - h
    H = np.fft.fft(h)
    G = np.fft.fft(g)
    # Cross correlation C_k = sum h_i g_{i+k}
    # This is IFFT(FFT(h) * conj(FFT(g)))? 
    # Let's verify.
    # Correlation C_k = sum_x h(x) g(x+k).
    # FFT property: FFT(correlation) = FFT(h) * conj(FFT(g))? 
    # Actually, circular cross-correlation c[k] = sum_n h[n] g[n+k].
    # DFT(c) = H * conj(G)? 
    # Let's test with small N.
    # h=[1,0], g=[0,1].
    # c[0] = 1*0 + 0*1 = 0.
    # c[1] = 1*1 + 0*0 = 1.
    # H=[1,1], G=[1,-1]. conj(G)=[1,-1]. H*conj(G)=[1,-1]. IFFT=[0,1]. Correct.
    
    C = np.real(np.fft.ifft(H * np.conj(G)))
    return C

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initialize
    # Start with a random feasible point
    np.random.seed(42) # Deterministic for reproducibility in this block, or random?
    # Let's use random restarts.
    
    best_h = None
    best_score = float('inf')
    
    # Multiple restarts to escape local minima
    num_restarts = 10
    time_per_restart = (deadline - start_time) / num_restarts
    
    for r in range(num_restarts):
        # Initialize h
        # Start with a noisy binary-like sequence
        h = np.random.rand(N)
        h = _project(h)
        
        # Adam optimizer state
        m = np.zeros(N)
        v = np.zeros(N)
        t = 0
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        lr = 0.05
        
        # Softmax beta for log-sum-exp
        # Start with lower beta (smoother) and increase (sharper)
        beta_start = 10.0
        beta_end = 100.0
        
        max_iters = 500
        
        for i in range(max_iters):
            if time.time() > start_time + r * time_per_restart + time_per_restart * 0.9:
                break
            
            # Current overlaps
            C = _compute_overlaps(h)
            
            # Objective: log-sum-exp(beta * C) / beta
            # To avoid overflow, subtract max
            C_max = np.max(C)
            beta_C = beta_start * C
            beta_C_shifted = beta_C - np.max(beta_C)
            LSE = np.log(np.sum(np.exp(beta_C_shifted))) + np.max(beta_C)
            obj = LSE / beta_start
            
            # Gradient of LSE w.r.t C:
            # d/dC_j (1/beta * log(sum exp(beta * C))) 
            # = (1/beta) * (beta * exp(beta * C_j) / sum exp(beta * C))
            # = exp(beta * C_j) / sum exp(beta * C)
            # This is the softmax distribution over C.
            
            weights = np.exp(beta_C_shifted)
            weights /= np.sum(weights)
            
            # Gradient of C w.r.t h
            # C_k = sum_i h_i (1 - h_{i+k})
            # dC_k/dh_m = delta_{m,k} (1 - h_k) + h_m * (-delta_{m, k})? No.
            # C_k = sum_i h_i - sum_i h_i h_{i+k}
            # dC_k/dh_m = 1 - (delta_{m,k} h_{m+k} + h_m delta_{m, k})? 
            # Let's derive carefully.
            # C_k = sum_i h_i (1 - h_{i+k})
            # Term involving h_m:
            # 1. h_m appears in the first sum: h_m * (1 - h_{m+k})
            # 2. h_m appears in the second part of the product for some i such that i+k = m => i = m-k.
            #    Term: h_{m-k} * (1 - h_m)
            
            # So dC_k/dh_m = (1 - h_{m+k}) - h_{m-k}
            
            # Total gradient G_h = sum_k weights[k] * dC_k/dh_m
            # G_h[m] = sum_k weights[k] * (1 - h[(m+k)%N] - h[(m-k)%N])
            
            # We can compute this efficiently using FFT or just loop if N is small.
            # N=100, so loop is fine, or use correlation.
            # Let W be weights.
            # G_h = W * 1 - corr(W, h) - corr(W, h_reversed)?
            # Actually, let's just use the definition.
            # G_h[m] = sum_k W_k - sum_k W_k h_{m+k} - sum_k W_k h_{m-k}
            # sum_k W_k = 1 (since weights are softmax)
            # sum_k W_k h_{m+k} is the circular convolution (W * h)[m]?
            # Conv: (W*h)[m] = sum_k W_k h_{m-k}.
            # Correlation: sum_k W_k h_{m+k} = (W * h_flipped)[m]?
            
            # Let's use FFT for speed and correctness.
            # A = sum_k W_k h_{m+k}
            # This is cross-correlation of W and h.
            # FFT(A) = FFT(W) * conj(FFT(h))
            # B = sum_k W_k h_{m-k}
            # This is convolution of W and h.
            # FFT(B) = FFT(W) * FFT(h)
            
            W_fft = np.fft.fft(weights)
            H_fft = np.fft.fft(h)
            
            A = np.real(np.fft.ifft(W_fft * np.conj(H_fft))) # sum W_k h_{m+k}
            B = np.real(np.fft.ifft(W_fft * H_fft))           # sum W_k h_{m-k}
            
            grad_C = 1.0 - A - B
            
            # Gradient of objective w.r.t h
            grad_h = grad_C
            
            # Adam update
            t += 1
            m = beta1 * m + (1 - beta1) * grad_h
            v = beta2 * v + (1 - beta2) * (grad_h ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            h_new = h - lr * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Project to feasible set
            h_new = _project(h_new)
            
            # Update h
            h = h_new
            
            # Track best
            # We need the actual max overlap, not the smooth approximation
            C_current = _compute_overlaps(h)
            current_max = np.max(C_current)
            
            if current_max < best_score:
                best_score = current_max
                best_h = h.copy()
                
            # Anneal beta
            progress = i / max_iters
            beta_start = beta_start + (beta_end - beta_start) * (progress ** 2)

    # Final polish: If best_h is not strictly binary, we might try to snap to binary?
    # But the problem allows continuous [0,1].
    # The score is based on the raw value.
    # Ensure sum is exactly N/2 (projection does this, but floating point might drift slightly)
    # _project guarantees sum is close to N/2.
    
    if best_h is None:
        # Fallback
        best_h = np.full(N, 0.5)
        
    # Final check and correction
    # The evaluator requires sum to be exactly len/2? "sum is exactly len/2"
    # Due to floating point, we might need to adjust one element.
    current_sum = np.sum(best_h)
    diff = N / 2.0 - current_sum
    # Distribute diff to the element with the most "room"
    # Or just add to the first element, clipping if necessary.
    # Better: add diff to the element closest to 0.5?
    # Simple: add to index 0, clip, and re-normalize? 
    # Since N=100, diff is likely tiny.
    best_h[0] += diff
    # Clip and re-project if it went out of bounds
    if best_h[0] < 0 or best_h[0] > 1:
         best_h = _project(best_h)
         
    return best_h
