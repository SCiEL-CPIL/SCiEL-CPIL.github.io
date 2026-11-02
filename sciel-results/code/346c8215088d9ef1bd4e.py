"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_val = 0.0
    best_h = np.ones(N)
    
    # Pre-compute indices
    idx = np.arange(N)
    
    # We will search over different sparsities (support sizes)
    # For N=50, a Sidon set or random sparse set can keep max autoconv low.
    # If support size is K, max autoconv peak is at most K (if perfectly orthogonal) 
    # but usually lower for random sets.
    # Ratio ~ (Sum h^2) / (Sum h * MaxConv)
    
    # Strategy:
    # 1. Try several random supports of size K (e.g., 10, 20, 30, 40)
    # 2. For each support, optimize amplitudes h_i > 0 using gradient ascent
    #    on the log-amplitude or directly, maximizing the ratio.
    #    Let f(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    #    We can maximize log(f(h)) or just f(h).
    #    Gradient ascent with step size adaptation.
    
    support_sizes = [10, 15, 20, 25, 30, 35, 40, 45]
    
    # Shuffle support sizes to ensure variety if time runs out
    np.random.seed(42)
    np.random.shuffle(support_sizes)
    
    for K in support_sizes:
        if time.time() > deadline:
            break
            
        # Try multiple random supports for this K
        trials_per_K = 5
        for t in range(trials_per_K):
            if time.time() > deadline:
                break
                
            # Select random support
            support = np.random.choice(N, K, replace=False)
            support.sort()
            
            # Initialize amplitudes on support
            # Start with uniform, then optimize
            h_full = np.zeros(N)
            h_full[support] = 1.0
            
            # Optimize amplitudes
            # We work in log-space to keep positivity, or just clip.
            # Let's use direct gradient ascent on h > 0.
            # f(h) = sum(h^2) / (sum(h) * max_conv(h))
            
            # To speed up, we can precompute the convolution matrix structure?
            # No, just use np.convolve for speed.
            
            # Gradient ascent parameters
            lr = 0.1
            momentum = 0.9
            v = np.zeros(N)
            
            # We only update h on support
            h = h_full.copy()
            
            # Pre-calculate which indices are in support for fast indexing
            in_support = np.zeros(N, dtype=bool)
            in_support[support] = True
            support_idx = support
            
            # Initial value
            conv = np.convolve(h, h, mode='full')
            max_conv = np.max(conv)
            l1 = np.sum(h)
            l2_sq = np.sum(h**2)
            val = l2_sq / (l1 * max_conv)
            
            if val > best_val:
                best_val = val
                best_h = h.copy()
                
            # Gradient ascent loop
            # We want to maximize R = A / (B * C) where A=sum h^2, B=sum h, C=max conv
            # dR/dh_i = (2h_i * B * C - A * (dB/dh_i * C + B * dC/dh_i)) / (B^2 * C^2)
            # dB/dh_i = 1
            # dC/dh_i is tricky because C = max_j (h*h)_j.
            # Let j* be the argmax of (h*h).
            # (h*h)_j = sum_k h_k h_{j-k} (with appropriate bounds)
            # d(h*h)_j* / dh_i = sum_k (delta_{k,i} h_{j*-k} + h_k delta_{j*-k, i})
            #                 = h_{j*-i} + h_{j*-(j*-i)} = h_{j*-i} + h_i (if indices valid)
            
            for step in range(500):
                if time.time() > deadline:
                    break
                    
                conv = np.convolve(h, h, mode='full')
                # Find max conv index
                # Note: np.convolve full mode has length 2N-1. Index 0 corresponds to shift 0?
                # Actually, for arrays of length N, full conv index m corresponds to sum_{k} h_k h_{m-k}
                # where indices are valid.
                max_idx = np.argmax(conv)
                max_conv = conv[max_idx]
                
                l1 = np.sum(h)
                l2_sq = np.sum(h**2)
                
                # If max_conv is 0, gradient is ill-defined, but h>=0 and not all 0
                if max_conv < 1e-9:
                    break
                    
                # Calculate gradient for all i in support
                # We need dC/dh_i for i in support
                # C = (h*h)_{max_idx}
                # dC/dh_i = h_{max_idx - i} + h_i (if max_idx - i is in [0, N-1])
                # Note: h is zero outside support.
                
                grad = np.zeros(N)
                
                # Vectorized calculation of dC/dh_i for i in support
                # i ranges over support_idx
                # j_star = max_idx
                # term1 = h[j_star - i]
                # term2 = h[i]
                
                # We need to be careful with indices.
                # h is length N.
                # j_star is in [0, 2N-2].
                # i is in [0, N-1].
                # j_star - i is in [-(N-1), 2N-2].
                # h[k] is 0 if k < 0 or k >= N.
                
                i_vals = support_idx
                j_star = max_idx
                
                # Calculate h[j_star - i] for all i in i_vals
                # valid indices for h are 0..N-1
                # Let k = j_star - i
                k_vals = j_star - i_vals
                valid_k = (k_vals >= 0) & (k_vals < N)
                
                dC_di = np.zeros(K)
                # term2 is h[i]
                dC_di[:] = h[i_vals]
                
                # term1 is h[k] where k = j_star - i
                # Only add if k is valid
                if np.any(valid_k):
                    dC_di[valid_k] += h[k_vals[valid_k]]
                
                # Now compute full gradient for i in support
                # R = A / (B * C)
                # dR/dh_i = (2 h_i B C - A (C + B dC_di)) / (B^2 C^2)
                #          = (2 h_i B C - A C - A B dC_di) / (B^2 C^2)
                
                num = 2 * h[i_vals] * l1 * max_conv - l2_sq * max_conv - l2_sq * l1 * dC_di
                den = l1**2 * max_conv**2
                
                grad_vals = num / den
                
                # Update momentum
                v[support_idx] = momentum * v[support_idx] + (1 - momentum) * grad_vals
                
                # Update h
                # We need to ensure h stays positive.
                # Since we are maximizing, and h=0 is a boundary, 
                # if gradient is negative, we might go to 0.
                # Let's use a small epsilon or clip.
                new_h_vals = h[support_idx] + lr * v[support_idx]
                new_h_vals = np.maximum(new_h_vals, 0.0)
                
                h[support_idx] = new_h_vals
                
                # Optional: normalize to prevent overflow/underflow?
                # The ratio is scale invariant! 
                # R(c*h) = (c^2 A) / (c B * c C) = R(h)
                # So we can normalize h to sum to 1 or max to 1 to keep numbers stable.
                if np.sum(h) > 1e6 or np.sum(h) < 1e-6:
                    h = h / np.sum(h)
                    v = v / np.sum(h) # Scale momentum too? Or just reset?
                    # Actually, if we scale h, the gradient direction is same, but magnitude changes.
                    # Since R is scale invariant, we can fix sum(h)=1.
                    # Let's just normalize h to sum=1 after update.
                    h = h / np.sum(h)
                    # Recalculate v? No, v is gradient direction.
                    # If we constrain sum(h)=1, we should project gradient.
                    # But for now, let's just let it float and normalize periodically.
                
                # Check improvement
                conv_check = np.convolve(h, h, mode='full')
                max_conv_check = np.max(conv_check)
                l1_check = np.sum(h)
                l2_sq_check = np.sum(h**2)
                if max_conv_check > 1e-9 and l1_check > 1e-9:
                    val_check = l2_sq_check / (l1_check * max_conv_check)
                    if val_check > best_val:
                        best_val = val_check
                        best_h = h.copy()

    return list(best_h)
# EVOLVE-BLOCK-END
