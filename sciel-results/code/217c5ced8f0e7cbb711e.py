import numpy as np
import time

N = 400
DX = 0.5 / N

def _linear_acf(h):
    """
    Compute the linear autocorrelation of h using FFT.
    Returns array of length 2*N - 1, centered at N-1.
    """
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    power = np.abs(H) ** 2
    acf_full = np.fft.irfft(power, n=fft_size)
    acf = np.empty(2 * n - 1, dtype=np.float64)
    acf[n-1] = acf_full[0]
    for k in range(1, n):
        acf[n-1+k] = acf_full[k]
        acf[n-1-k] = acf_full[fft_size - k]
    return acf

def _score(h):
    """
    Calculate the upper bound on C3, which is DX * max_abs_acf.
    """
    acf = _linear_acf(h)
    n = len(h)
    center = n - 1
    right = acf[center+1:]
    left = acf[:center]
    max_abs = max(np.max(np.abs(right)), np.max(np.abs(left)))
    return float(DX * max_abs)

def _psl_value(acf, n):
    """
    Extract the Peak Sidelobe Level from the ACF array.
    """
    if n <= 1:
        return 0.0
    center = n - 1
    right = acf[center+1:]
    left = acf[:center]
    return float(max(np.max(np.abs(right)), np.max(np.abs(left))))

def _compute_acf(h):
    return _linear_acf(h)

def _generate_candidates():
    """
    Generate a diverse set of initial candidates.
    """
    candidates = []
    rng = np.random.default_rng(42)
    
    # 1. Random binary
    for _ in range(50):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # 2. Thue-Morse
    tm = np.array([1.0 if bin(i).count('1') % 2 == 0 else -1.0 for i in range(N)])
    candidates.append(tm)
    candidates.append(-tm)
    
    # 3. Legendre sequences (using primes)
    primes = [409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499]
    for p in primes:
        if p > N:
            seq = np.zeros(N)
            for i in range(1, N):
                # Legendre symbol
                ls = pow(i, (p-1)//2, p)
                seq[i] = 1.0 if ls == 1 else -1.0
            seq[0] = 1.0
            candidates.append(seq)
            candidates.append(-seq)
            
    # 4. M-sequences (LFSR)
    # Polynomials for degree 9 (max length 511)
    polys = [0x0130, 0x0160, 0x0190, 0x01C0, 0x0230, 0x0260]
    for poly in polys:
        degree = 9
        m = (1 << degree) - 1
        state = 1
        seq = np.zeros(N)
        for i in range(N):
            seq[i] = 1.0 if (state & 1) else -1.0
            lsb = state & 1
            state >>= 1
            if lsb:
                state ^= poly
            state &= m
        candidates.append(seq)
        candidates.append(-seq)
        
    # 5. Chirp-like (quadratic phase)
    t = np.arange(N)
    for a in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        phase = 2 * np.pi * a * t * t / N
        candidates.append(np.sign(np.cos(phase)))
        candidates.append(np.sign(np.sin(phase)))
        
    # 6. Low-pass filtered noise (Haar inspiration)
    # Generate Gaussian noise, filter, threshold
    for _ in range(20):
        noise = rng.standard_normal(N)
        # Simple low pass: moving average
        kernel = np.ones(5) / 5.0
        filtered = np.convolve(noise, kernel, mode='same')
        candidates.append(np.sign(filtered))
        
    return candidates

def _continuous_optimize(h_init, steps=2000, lr=0.1, beta=10.0, time_budget=5.0):
    """
    Optimize a continuous version of h to minimize a soft-max of ACF peaks.
    Then project to {-1, 1}.
    """
    start_time = time.time()
    h = h_init.astype(np.float64).copy()
    n = len(h)
    
    # Precompute FFT size
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
        
    best_h = h.copy()
    best_score = _score(np.sign(h))
    
    for step in range(steps):
        if time.time() - start_time > time_budget:
            break
            
        # Compute ACF
        H = np.fft.rfft(h, n=fft_size)
        power = np.abs(H) ** 2
        acf_full = np.fft.irfft(power, n=fft_size)
        acf = np.empty(2 * n - 1, dtype=np.float64)
        acf[n-1] = acf_full[0]
        for k in range(1, n):
            acf[n-1+k] = acf_full[k]
            acf[n-1-k] = acf_full[fft_size - k]
            
        # Sidelobes
        right = acf[n:]
        left = acf[:n-1]
        sidelobes = np.concatenate([left, right])
        abs_sl = np.abs(sidelobes)
        
        # Soft max objective: (1/beta) * log(sum(exp(beta * |sl|)))
        # Gradient of soft max w.r.t h
        # d/dh (1/beta log sum exp(beta |sl_k|)) = sum (exp(beta |sl_k|) / sum exp) * sign(sl_k) * d/dh sl_k
        # d/dh_j sl_k = h[j-k] + h[j+k] (with bounds)
        
        # Weights for each lag
        max_abs = np.max(abs_sl)
        if max_abs > 1e-9:
            weights = np.exp(beta * (abs_sl - max_abs)) # Shift for stability
            sum_weights = np.sum(weights)
            if sum_weights > 0:
                weights /= sum_weights
            else:
                weights = np.ones_like(abs_sl) / len(abs_sl)
        else:
            weights = np.ones_like(abs_sl) / len(abs_sl)
            
        signs = np.sign(sidelobes)
        # If 0, sign is 0, but derivative of |x| at 0 is undefined. 
        # In practice, subgradient 0 is fine.
        
        # Compute gradient
        grad = np.zeros(n)
        # For each lag k, contribution to grad[j] is weight_k * sign_k * (h[j-k] + h[j+k])
        # This is a convolution-like operation.
        # Let's do it efficiently.
        # grad[j] = sum_k w_k s_k (h[j-k] + h[j+k])
        # = sum_k w_k s_k h[j-k] + sum_k w_k s_k h[j+k]
        
        # Create the kernel for the first term
        # Kernel W[j-k] -> W is indexed by k. 
        # Let's construct an array K of length 2n-1 representing the combined weights.
        # Actually, simpler to just loop k if n is small, but n=400, loops are slow in Python.
        # Let's use FFT for the gradient computation too?
        # grad = ifft( fft(h) * fft(kernel) )?
        # The term is sum_k c_k h[j-k]. This is convolution of h with c.
        # c[k] for k>0 is w_k s_k. c[-k] is w_k s_k.
        # So c is symmetric if s_k is symmetric? No, s_k = sign(ACF[k]). ACF is symmetric, so s_k is symmetric.
        # So c is symmetric.
        
        c = np.zeros(2*n-1)
        # Map sidelobes to indices
        # left: indices 0 to n-2 correspond to lags -(n-1) to -1
        # right: indices n to 2n-2 correspond to lags 1 to n-1
        # Wait, my acf array:
        # acf[0] is lag -(n-1)
        # acf[n-1] is lag 0
        # acf[2n-2] is lag n-1
        
        # left part of acf: acf[0:n-1] -> lags -1 to -(n-1)
        # right part of acf: acf[n:2n-1] -> lags 1 to n-1
        
        # Let's re-extract with correct signs
        # lag k > 0: acf[n-1+k]
        # lag -k: acf[n-1-k]
        
        # We need w_k for k=1..n-1.
        # w_k corresponds to abs_sl at index k-1 in the concatenated array?
        # My concatenated array: [left, right]
        # left: acf[0], ..., acf[n-2] -> lags -(n-1), ..., -1
        # right: acf[n], ..., acf[2n-2] -> lags 1, ..., n-1
        
        # Let's just build the kernel directly
        kernel = np.zeros(2*n-1)
        # For k in 1..n-1:
        # weight w_k, sign s_k
        # Contribution to grad[j] from lag k: w_k s_k (h[j-k] + h[j+k])
        # This is equivalent to convolving h with a kernel K where K[k] = w_k s_k and K[-k] = w_k s_k.
        # In 0-indexed array of size 2n-1:
        # Index n-1+k corresponds to lag k.
        # Index n-1-k corresponds to lag -k.
        
        for k in range(1, n):
            # Get weight and sign for this lag
            # In my 'sidelobes' array:
            # left part is acf[0:n-1]. Index i in left corresponds to lag -(n-1-i).
            # right part is acf[n:2n-1]. Index i in right corresponds to lag (i+1).
            
            # It's easier to just recompute abs_sl and signs from acf directly to avoid mapping errors
            pass
            
        # Let's redo the gradient calculation clearly
        # We have acf array.
        # We want grad[j] = sum_{k=1}^{n-1} w_k s_k (h[j-k] + h[j+k])
        # where w_k = exp(beta |acf[k]|) / Z, s_k = sign(acf[k])
        
        # Construct kernel K of length 2n-1
        K = np.zeros(2*n-1)
        for k in range(1, n):
            val = acf[n-1+k]
            abs_val = abs(val)
            # We need to map this to the weight.
            # We pre-computed weights for the concatenated array.
            # Let's just recompute weights from acf to be safe.
            
        # Recompute weights from acf
        abs_acf = np.abs(acf)
        abs_acf[n-1] = 0 # Exclude main lobe
        max_abs = np.max(abs_acf)
        if max_abs > 1e-9:
            weights = np.exp(beta * (abs_acf - max_abs))
            sum_w = np.sum(weights)
            if sum_w > 0:
                weights /= sum_w
            else:
                weights = np.ones_like(abs_acf) / (2*n-1)
        else:
            weights = np.ones_like(abs_acf) / (2*n-1)
            
        signs = np.sign(acf)
        
        # Kernel for convolution
        # grad = ifft( fft(h) * fft(K) )?
        # The operation is sum_k K[k] h[j-k]. This is cross-correlation or convolution?
        # sum_k K[k] h[j-k] is convolution (h * K)[j] if K is indexed by k.
        # But our K is centered at n-1.
        
        # Let's use the property:
        # grad[j] = sum_k weights[k] * signs[k] * (h[j-k] + h[j+k])
        # Let C[k] = weights[k] * signs[k].
        # Note C is symmetric because ACF is symmetric and weight depends on abs.
        # So C[k] = C[-k].
        # grad[j] = sum_k C[k] h[j-k] + sum_k C[k] h[j+k]
        # = sum_k C[k] h[j-k] + sum_m C[-m] h[j-m] (let m=-k)
        # = 2 * sum_k C[k] h[j-k] (if C is symmetric)
        
        # So we just need to compute the convolution of h with C, then multiply by 2.
        # But C is defined on lags -n+1..n-1.
        
        # Let's create a kernel array of size fft_size for FFT convolution.
        kernel_fft = np.zeros(fft_size, dtype=np.complex128)
        # Place C in the kernel
        # C[k] is at index k in lag space.
        # In FFT array, lag k is at index k (if k>0) or fft_size-k (if k<0).
        # But standard FFT convolution assumes kernel starts at index 0.
        # Let's shift C so that lag 0 is at index 0? No, easier to use linear convolution logic or just loop if n is small.
        # N=400. Looping 400 times is fast.
        
        grad = np.zeros(n)
        for k in range(1, n):
            c_val = weights[n-1+k] * signs[n-1+k]
            if c_val == 0: continue
            # h[j-k] and h[j+k]
            # h[j-k] is valid for j >= k
            # h[j+k] is valid for j < n-k
            # grad[j] += c_val * (h[j-k] + h[j+k])
            
            # Vectorized:
            # term1: h[k:] added to grad[:n-k]
            # term2: h[:n-k] added to grad[k:]
            
            grad[:n-k] += c_val * h[k:]
            grad[k:] += c_val * h[:n-k]
            
        # Update h
        h -= lr * grad
        
        # Project to [-1, 1] or just keep continuous?
        # If we keep continuous, the "sign" projection at the end might jump.
        # Better to project to [-1, 1] to keep values bounded, then final sign.
        h = np.clip(h, -1.0, 1.0)
        
        # Check score of projected sign
        h_sign = np.sign(h)
        h_sign[h_sign == 0] = 1.0
        s = _score(h_sign)
        if s < best_score:
            best_score = s
            best_h = h_sign.copy()
            
    return best_h

def _local_search(h, time_budget=10.0):
    """
    Fast local search on binary sequence.
    """
    start_time = time.time()
    h = h.astype(np.float64).copy()
    n = len(h)
    acf = _linear_acf(h)
    current_psl = _psl_value(acf, n)
    best_h = h.copy()
    best_psl = current_psl
    rng = np.random.default_rng(123)
    
    # Precompute indices for fast access
    for _ in range(10000): # Max iterations
        if time.time() - start_time > time_budget:
            break
            
        # Try a random flip
        i = int(rng.integers(n))
        
        # Calculate change in ACF
        # Delta ACF[k] = -2 h[i] (h[i-k] + h[i+k])
        # We only care if this reduces the max absolute ACF.
        
        # Instead of full recompute, check if the flipped lags were the max.
        # Simple approach: Recompute ACF. N=400, FFT is fast.
        h[i] *= -1
        new_acf = _linear_acf(h)
        new_psl = _psl_value(new_acf, n)
        
        if new_psl < current_psl:
            current_psl = new_psl
            acf = new_acf
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()
        else:
            # Revert
            h[i] *= -1
            # acf is still the old one, no need to recompute if we revert
            
    return best_h, best_psl

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin for 150s limit
    
    best_h = None
    best_score = float('inf')
    
    candidates = _generate_candidates()
    
    # Score all candidates
    scored = []
    for h in candidates:
        s = _score(h)
        scored.append((s, h))
    scored.sort(key=lambda x: x[0])
    
    # Take top candidates for refinement
    top_k = min(20, len(scored))
    
    for i in range(top_k):
        if time.time() > deadline - 20:
            break
            
        s, h = scored[i]
        
        # 1. Continuous optimization
        time_left = deadline - time.time() - 5.0
        if time_left > 5.0:
            h_cont = _continuous_optimize(h, steps=1000, lr=0.05, beta=20.0, time_budget=min(5.0, time_left))
            s_cont = _score(h_cont)
            if s_cont < best_score:
                best_score = s_cont
                best_h = h_cont.copy()
        
        # 2. Local search on the best of (original, continuous)
        h_to_polish = h_cont if s_cont < s else h
        s_to_polish = min(s_cont, s)
        
        time_left = deadline - time.time() - 2.0
        if time_left > 1.0:
            h_ls, s_ls = _local_search(h_to_polish, time_budget=min(3.0, time_left))
            if s_ls < best_score:
                best_score = s_ls
                best_h = h_ls.copy()
        
        # Update best_score for this candidate if local search improved it
        # (Already done above)
        
    # If no improvement, return best candidate
    if best_h is None:
        best_h = scored[0][1]
        
    return best_h
