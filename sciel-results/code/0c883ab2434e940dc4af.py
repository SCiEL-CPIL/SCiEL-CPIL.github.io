import numpy as np
import time

N = 600

def compute_score(h):
    h = np.asarray(h)
    if np.any(h < 0):
        return float('inf')
    sum_h = np.sum(h)
    if sum_h == 0:
        return float('inf')
    # Compute autoconvolution
    # c_k for k from -(N-1) to N-1
    # Using FFT for speed
    h_pad = np.zeros(2 * N)
    h_pad[:N] = h
    fft_h = np.fft.fft(h_pad)
    fft_conv = fft_h * fft_h
    conv = np.fft.ifft(fft_conv).real
    # conv[0] corresponds to k=0 (c_0)
    # conv[N-1] corresponds to k=-(N-1)
    # conv[N] corresponds to k=N? No, linear conv length is 2N-1.
    # Indices: 0 to 2N-2.
    # c_k for k>=0 is conv[k].
    # c_k for k<0 is conv[2N-1+k]?
    # Actually, np.fft.ifft of fft_h^2 gives circular convolution if not padded enough.
    # We padded to 2N, so linear conv is contained.
    # The max is at k=0, which is conv[0].
    max_val = np.max(conv)
    return max_val / (sum_h ** 2)

def construct():
    start_time = time.time()
    deadline = 140.0  # 20s margin
    
    best_h = np.ones(N)
    best_score = compute_score(best_h)
    
    # Try a few random initializations and coordinate descent
    # Since uniform is likely optimal for max/sum^2, we just verify.
    # But to be safe, run a few steps of optimization.
    
    def optimize(h, max_iters=100):
        h = h.copy()
        h = np.maximum(h, 1e-9)
        for i in range(max_iters):
            if time.time() > deadline:
                break
            # Compute gradient of score with respect to h_j
            # This is complex. Let's use a simple heuristic: 
            # Perturb each coordinate and keep if it improves.
            # With N=600, 600 perturbations per iter is fast.
            improved = False
            for j in range(N):
                if time.time() > deadline:
                    break
                orig = h[j]
                # Try increasing and decreasing
                # Step size relative to mean
                step = 0.01 * np.mean(h)
                
                h[j] = orig + step
                s1 = compute_score(h)
                h[j] = orig - step
                s2 = compute_score(h)
                h[j] = orig
                
                if s1 < best_score:
                    h[j] = orig + step
                    best_score = s1
                    improved = True
                elif s2 < best_score:
                    h[j] = orig - step
                    best_score = s2
                    improved = True
            if not improved:
                break
        return h, best_score

    # Initialize with uniform
    h_curr = np.ones(N)
    s_curr = compute_score(h_curr)
    best_h = h_curr
    best_score = s_curr
    
    # Try 2-block structure
    h_2block = np.zeros(N)
    h_2block[:N//2] = 1.0
    h_2block[N//2:] = 1.0 # This is just uniform again.
    # Real 2 block:
    h_2block = np.zeros(N)
    h_2block[:N//4] = 1.0
    h_2block[3*N//4:] = 1.0
    
    h_opt, s_opt = optimize(h_2block, max_iters=5)
    if s_opt < best_score:
        best_score = s_opt
        best_h = h_opt
    
    # Try random sparse
    h_rand = np.random.rand(N)
    h_opt, s_opt = optimize(h_rand, max_iters=5)
    if s_opt < best_score:
        best_score = s_opt
        best_h = h_opt

    # Final check on uniform
    h_uniform = np.ones(N)
    s_uniform = compute_score(h_uniform)
    if s_uniform < best_score:
        best_score = s_uniform
        best_h = h_uniform

    return list(best_h)
