import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    # Safety margin: 10 seconds before the 120s limit
    deadline = start_time + 110.0
    
    # Objective: minimize value = abs(2 * N * max(conv(h,h)) / sum(h)^2)
    # Due to scale invariance, fix sum(h) = 1.
    # Then value = 2 * N * max(conv(h,h)).
    # We want to minimize max(conv(h,h)).
    # 
    # Key Insight from "Different Strategy":
    # Instead of optimizing h directly with non-smooth max, 
    # we can use a Smoothed Max (Softmax) approximation and 
    # perform Gradient Descent on h, but with a crucial structural twist:
    # We initialize h based on the "eigenfunction" of the convolution operator 
    # that minimizes the peak-to-sum ratio.
    #
    # Actually, a more "Materially Different" approach is to use 
    # **Alternating Direction Method of Multipliers (ADMM)** or 
    # **Convex Optimization on the Toeplitz Matrix**.
    #
    # Let's stick to a robust **Gradient Descent on Softmax** but with 
    # **Adaptive Temperature** and **Multiple Restarts** from structured 
    # initializations (eigenfunctions of circulant approximations).
    #
    # The current program uses a subgradient method on the exact max, 
    # which is non-smooth and prone to getting stuck.
    # This new program uses a smooth surrogate (Softmax) which allows 
    # for more stable gradient steps, and crucially, explores a wider 
    # range of "shapes" via structured initializations derived from 
    # discrete cosine/sine transforms.
    
    def compute_value_exact(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        c = np.convolve(h, h)
        return abs(2.0 * N * np.max(c) / (s * s))

    def softmax_max(c, T):
        c_max = np.max(c)
        c_shifted = c - c_max
        lse = c_max + T * np.log(np.sum(np.exp(c_shifted / T)))
        # Gradient of lse w.r.t c is softmax(c/T)
        p = np.exp(c_shifted / T)
        p = p / np.sum(p)
        return lse, p

    def gradient_softmax(h, T):
        c = np.convolve(h, h)
        lse, p = softmax_max(c, T)
        
        # Gradient of lse w.r.t h
        # grad_h (p^T c) = 2 * conv(p, h) [sliced appropriately]
        # As derived before, grad = 2 * (p * h) where * is convolution.
        # We need the slice corresponding to h indices 0..N-1.
        # conv(p, h) has length 3N-2.
        # The term for h[j] in the gradient is sum_k p[k] (h[k-j] + h[j-k]).
        # This is equivalent to 2 * (p * h)[j] IF p is symmetric or if we handle boundaries.
        # Let's use the explicit vectorized computation for safety and correctness.
        
        # p has length 2N-1.
        # We compute g = 2 * ifft(fft(p) * fft(h)) ? No, that's full conv.
        # Let's use the property:
        # grad[j] = sum_k p[k] h[k-j] + sum_k p[k] h[j-k]
        
        # Part 2: sum_k p[k] h[j-k] is conv(p, h)[j].
        # Part 1: sum_k p[k] h[k-j].
        
        # Efficient computation using FFT for the full convolution of p and h
        # and slicing.
        
        # However, for N=400, a direct O(N^2) loop in Python is ~160k ops/iter.
        # 1000 iters = 160M ops. Might be ~10-20s. Acceptable.
        # But let's try to vectorize better.
        
        # g1[j] = sum_{k} p[k] h[k-j]
        # This is cross-correlation.
        
        # Let's just use the fact that for the purpose of descent, 
        # we can approximate the gradient by 2 * convolve(p, h) centered.
        # Let's verify the slice.
        # conv(p, h) length 3N-2.
        # Index j in h corresponds to the "center" of the overlap?
        
        # Let's use a simpler, robust gradient:
        # Finite difference is too slow.
        # Let's use the analytic gradient with explicit sum, vectorized by rows.
        
        # p is (2N-1,). h is (N,).
        # We want a matrix M where M[k, j] = h[k-j] if valid else 0.
        # Then g1 = p @ M.
        # Similarly for g2.
        
        # Constructing M is O(N^2) memory and time.
        # N=400, M is 400x400. 160k floats. 1.2 MB. Fast.
        
        # Indices for h[k-j]: 0 <= k-j < N => j <= k < j+N.
        # For a fixed k, valid j are max(0, k-N+1) <= j <= min(N-1, k).
        
        # Let's build the gradient using convolution properties more carefully.
        # grad_j = sum_k p_k h_{k-j} + sum_k p_k h_{j-k}
        
        # Let's compute conv_full = np.convolve(p, h, mode='full')
        # conv_full[m] = sum_i p_i h_{m-i}
        
        # Term 2: sum_k p_k h_{j-k}.
        # Let i=k. sum_i p_i h_{j-i}. This is conv_full[j].
        
        # Term 1: sum_k p_k h_{k-j}.
        # Let i = k-j. k = i+j.
        # sum_i p_{i+j} h_i.
        # This is not a standard slice of conv_full.
        
        # However, if we reverse p, let p_rev[i] = p[-1-i] ?
        
        # Let's just use the explicit loop for the gradient, it's safer.
        # We can optimize it by noting that p is sparse? No.
        
        # Alternative: Use the fact that grad = 2 * np.convolve(p, h, mode='same') 
        # is often a good approximation for the "centered" part, but boundaries matter.
        
        # Let's use a "soft" gradient:
        # g = 2 * np.convolve(p, h, mode='full')
        # The relevant part for h is the middle section.
        # For h of length N and p of length 2N-1, the "valid" convolution 
        # (where p and h overlap significantly) is in the middle.
        
        # Actually, let's just use the explicit sum with numpy broadcasting, 
        # which is fast enough for N=400.
        
        # p has shape (2N-1,)
        # h has shape (N,)
        
        # Create index arrays
        k_idx = np.arange(2*N-1)[:, None] # (2N-1, 1)
        j_idx = np.arange(N)[None, :]     # (1, N)
        
        # h[k-j] valid if 0 <= k-j < N
        diff1 = k_idx - j_idx
        mask1 = (diff1 >= 0) & (diff1 < N)
        
        # h[j-k] valid if 0 <= j-k < N
        diff2 = j_idx - k_idx
        mask2 = (diff2 >= 0) & (diff2 < N)
        
        # We need to sum p[k] * h[k-j] over k for each j.
        # This is (p * h_shifted) where h is shifted.
        
        # Let's do it via matrix multiplication.
        # M1[j, k] = h[k-j] if valid else 0.
        # This is a Toeplitz-like matrix.
        
        # Instead of building M, let's use the fact that:
        # g1 = np.correlate(p, h, mode='full') ?
        # np.correlate(p, h, 'full')[m] = sum_k p[k] h[k+m] ?
        # Definition: (a * b)[n] = sum_m a[m] b[n-m].
        # np.correlate(a, b, 'full')[n] = sum_m a[m] b[m+n].
        
        # We want sum_k p[k] h[k-j].
        # Let n = -j. sum_k p[k] h[k + (-j)] = np.correlate(p, h, 'full')[-j].
        # But 'full' output indices are 0 to len(p)+len(h)-2.
        # The offset is len(h)-1.
        # So index in output = -j + (N-1).
        
        # Let's test this hypothesis.
        # g1[j] = np.correlate(p, h, mode='full')[N-1-j]
        
        # And g2[j] = sum_k p[k] h[j-k] = conv(p, h)[j] = np.convolve(p, h)[j].
        # np.convolve(p, h)[j] = sum_i p[i] h[j-i].
        
        # So grad = (1/T) * (g1 + g2)
        
        corr = np.correlate(p, h, mode='full')
        conv = np.convolve(p, h, mode='full')
        
        # Slice for g1: indices N-1-N+1 to N-1-0 => 0 to N-1.
        # Wait, j goes 0 to N-1.
        # Index = N-1-j.
        # If j=0, idx=N-1.
        # If j=N-1, idx=0.
        # So g1_vec = corr[N-1::-1] ? No, corr is length 3N-2.
        # The valid part for correlation is usually centered.
        
        # Let's just use the explicit sum for the first 10 iterations to verify, 
        # then switch to the fast method if it matches.
        # For now, let's use a simpler, robust method: 
        # **Subgradient Descent on the Exact Max** with **Line Search**.
        # This avoids the complexity of the softmax gradient slicing.
        # The current program failed because it used a fixed small LR and 
        # didn't do line search.
        
        return None # Placeholder

    # Strategy: 
    # 1. Generate diverse initializations (Sine, Cosine, Ramp, Random, Comb).
    # 2. For each, perform Gradient Descent on the **Smoothed Max** (Softmax).
    # 3. Use **Backtracking Line Search** to ensure decrease.
    # 4. This is structurally different from the current "Subgradient on Exact Max" 
    #    by using a smooth approximation and adaptive step size.
    
    def softmax_obj(h, T):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e10, np.zeros(N)
        c = np.convolve(h, h)
        c_max = np.max(c)
        c_shifted = c - c_max
        lse = c_max + T * np.log(np.sum(np.exp(c_shifted / T)))
        val = 2.0 * N * lse / (s * s)
        
        # Gradient of val w.r.t h
        # val = 2 N lse s^-2
        # grad = 2 N ( (1/s^2) grad_lse - 2 lse s^-3 grad_s )
        # grad_s = ones(N)
        
        # grad_lse = (1/T) * (g1 + g2) as derived.
        p = np.exp(c_shifted / T)
        p = p / np.sum(p)
        
        # Compute grad_lse
        # Using the correlation/convolution trick
        # g1[j] = sum_k p[k] h[k-j]
        # g2[j] = sum_k p[k] h[j-k]
        
        # Let's use the explicit sum for safety, N=400 is small.
        # We can vectorize the sum over k for each j.
        
        # p is (2N-1,). h is (N,).
        # We want a vector g of length N.
        
        # Let's use the property that g = 2 * ifft(fft(p) * fft(h)) is NOT correct for this boundary.
        # Let's use the explicit loop but with numpy arrays for speed.
        
        grad_lse = np.zeros(N)
        
        # Precompute p
        # For each j, grad_lse[j] = sum_k p[k] * (h[k-j] if valid + h[j-k] if valid)
        
        # We can split this into two convolutions if we pad p and h correctly.
        # But let's just do the O(N^2) sum, it's ~160k ops.
        # 5000 iters * 160k = 800M. In Python, this is ~50-100s. Too slow.
        
        # We MUST use the fast method.
        # g2 = conv(p, h)[0:N] ?
        # conv(p, h)[j] = sum_i p[i] h[j-i].
        # This matches g2 exactly.
        
        # g1 = sum_k p[k] h[k-j].
        # This is np.correlate(p, h, mode='full')[N-1-j] ?
        # Let's verify:
        # np.correlate(p, h, 'full')[m] = sum_k p[k] h[k + (m - (N-1))].
        # We want h[k-j]. So m - (N-1) = -j => m = N-1-j.
        # So g1[j] = corr[N-1-j].
        
        corr = np.correlate(p, h, mode='full')
        conv = np.convolve(p, h, mode='full')
        
        # Extract g1
        # j ranges 0..N-1.
        # Indices in corr: N-1, N-2, ..., 0.
        g1_vec = corr[N-1::-1] # Length N
        
        # Extract g2
        # j ranges 0..N-1.
        # Indices in conv: 0, 1, ..., N-1.
        g2_vec = conv[0:N]
        
        grad_lse = (g1_vec + g2_vec) / T
        
        # Total gradient
        grad_val = (2.0 * N / (s * s)) * grad_lse - (4.0 * N * lse / (s ** 3)) * np.ones(N)
        
        return val, grad_val

    def line_search(h, grad, val, T, alpha, c1=1e-4, max_steps=20):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return h, val
        for _ in range(max_steps):
            h_new = h - alpha * grad
            # Normalize to sum=1 to stay on manifold? 
            # Or just let it drift? 
            # The objective is scale invariant. 
            # If we don't normalize, the gradient includes the scale term.
            # Let's normalize to sum=1 to simplify the landscape.
            s_new = np.sum(h_new)
            if abs(s_new) < 1e-12:
                alpha *= 0.5
                continue
            h_new = h_new / s_new
            
            val_new, _ = softmax_obj(h_new, T)
            if val_new <= val - c1 * alpha * np.dot(grad, h - h_new): # Simplified Wolfe
                return h_new, val_new
            alpha *= 0.5
        return h, val

    # Initializations
    x = np.linspace(0, 2*np.pi, N)
    inits = [
        np.ones(N),
        np.sin(x),
        np.cos(x),
        np.sin(2*x) + 2, # Positive bias
        np.arange(N, dtype=np.float64) / N, # Ramp
        np.random.default_rng(42).normal(0, 1, N),
        np.random.default_rng(43).normal(0, 1, N),
        # Comb
        np.array([1 if i % 4 == 0 else -0.5 for i in range(N)]),
    ]
    
    h_best = inits[0]
    val_best = compute_value_exact(h_best)
    
    for h_init in inits:
        h = h_init.copy()
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        h = h / s
        
        T = 1.0
        alpha = 0.1
        current_val, current_grad = softmax_obj(h, T)
        
        for _ in range(200):
            if time.time() > deadline:
                break
                
            # Adaptive Temperature: decrease T as we converge
            # T = max(0.01, T * 0.99)
            
            h, current_val = line_search(h, current_grad, current_val, T, alpha)
            
            # Check exact value
            exact_val = compute_value_exact(h)
            if exact_val < val_best:
                val_best = exact_val
                h_best = h.copy()
            
            # Update gradient
            _, current_grad = softmax_obj(h, T)
            
            # Decay alpha
            alpha *= 0.95
            
            if time.time() > deadline:
                break
    
    # Final polish with exact subgradient on the best found
    h = h_best.copy()
    s = np.sum(h)
    h = h / s
    
    for _ in range(50):
        if time.time() > deadline:
            break
        c = np.convolve(h, h)
        k_max = np.argmax(c)
        
        grad = np.zeros(N)
        for j in range(N):
            val = 0.0
            if 0 <= k_max - j < N:
                val += h[k_max - j]
            if 0 <= j - k_max < N:
                val += h[j - k_max]
            grad[j] = val
        
        # Line search on exact max
        alpha = 0.01
        val_curr = compute_value_exact(h)
        for _ in range(10):
            h_try = h - alpha * grad
            s_try = np.sum(h_try)
            if abs(s_try) < 1e-12:
                alpha *= 0.5
                continue
            h_try = h_try / s_try
            val_try = compute_value_exact(h_try)
            if val_try < val_curr:
                h = h_try
                val_curr = val_try
                break
            alpha *= 0.5
        
        if val_curr < val_best:
            val_best = val_curr
            h_best = h.copy()

    return h_best
