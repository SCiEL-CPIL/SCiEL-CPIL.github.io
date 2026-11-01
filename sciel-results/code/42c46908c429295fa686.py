import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    n_fft = 1 << (2 * N - 1).bit_length()
    
    def compute_acf(h):
        F = np.fft.fft(h, n=n_fft)
        power = np.abs(F) ** 2
        acf = np.fft.ifft(power).real
        # Take the first N lags (0 to N-1)
        return acf[:N]

    def get_max_sidelobe(acf):
        # The peak at lag 0 is N (since sum h^2 = N for binary).
        # We want to minimize the maximum of the other lags.
        # Note: acf is symmetric for real inputs, but we only computed 0..N-1.
        # For N=400, lags 1..399 are the positive lags.
        # The negative lags -1..-399 are symmetric to 1..399.
        # So max(acf[1:]) is the max sidelobe.
        return np.max(acf[1:])

    # Strategy:
    # 1. Generate initial candidates using "flat spectrum" initialization.
    #    If |F[k]| is constant, then autocorrelation is a delta function (ideal).
    #    We can generate random phases, take ifft, and threshold to +/- 1.
    # 2. Refine using a custom bit-flip heuristic.
    
    rng = np.random.RandomState(42)
    
    best_h = None
    best_val = float('inf')
    
    # Pre-compute indices for shifts? 
    # For bit-flip heuristic:
    # Flipping h[i] from s to -s (change -2s) changes acf[k] by:
    # delta_acf[k] = -2s * (h[i+k] + h[i-k]) for k != 0
    # delta_acf[0] = 0 (since h[i]^2 is always 1)
    
    # We want to minimize max(acf[1:])
    # A greedy approach: find the current peak lag K.
    # Find an index i and sign s such that flipping h[i] reduces acf[K] significantly
    # without increasing other peaks too much.
    
    # Faster heuristic:
    # Iterate through lags. If a lag k is the current max, try to flip bits to reduce it.
    # Specifically, if acf[k] is high, we want to reduce the sum h[j]h[j+k].
    # Flipping h[i] changes terms involving i and i+k.
    
    # Let's use a simpler, robust local search:
    # 1. Compute acf.
    # 2. Identify the worst lag k* (excluding 0).
    # 3. For each position i, calculate the change in acf[k*] if we flip h[i].
    #    Change in acf[k*] due to flipping h[i]:
    #    Old contribution: h[i]h[i+k*] + h[i-k*]h[i]
    #    New contribution: -h[i]h[i+k*] - h[i-k*](-h[i]) = -h[i]h[i+k*] + h[i-k*]h[i]
    #    Wait, flipping h[i] to -h[i].
    #    Term 1: h[i]h[i+k*] becomes (-h[i])h[i+k*] = -h[i]h[i+k*]. Change: -2 h[i]h[i+k*].
    #    Term 2: h[i-k*]h[i] becomes h[i-k*](-h[i]) = -h[i-k*]h[i]. Change: -2 h[i-k*]h[i].
    #    Total change in acf[k*] = -2 h[i] (h[i+k*] + h[i-k*]).
    # 4. Pick the flip that gives the most negative change (largest reduction).
    # 5. Check if this flip increases the global max sidelobe. If not, accept.
    #    To be efficient, we can just check the new max.
    
    def local_search(h, max_iters=5000):
        h = h.copy()
        acf = compute_acf(h)
        current_max = get_max_sidelobe(acf)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            # Find worst lag
            k_star = np.argmax(acf[1:]) + 1
            
            # Calculate potential improvement for each flip at index i
            # change_acf_k[i] = -2 * h[i] * (h[(i+k_star)%N] + h[(i-k_star)%N])
            h_plus = np.roll(h, -k_star)
            h_minus = np.roll(h, k_star)
            changes = -2.0 * h * (h_plus + h_minus)
            
            # We want the most negative change (largest reduction)
            i_best = np.argmin(changes)
            delta = changes[i_best]
            
            if delta >= 0:
                # No flip reduces the peak lag k_star
                # Try random flips to escape? Or stop.
                # Let's try a few random flips to escape local minima
                escaped = False
                for _ in range(5):
                    i_rand = rng.randint(N)
                    h_new = h.copy()
                    h_new[i_rand] *= -1
                    acf_new = compute_acf(h_new)
                    new_max = get_max_sidelobe(acf_new)
                    if new_max < current_max:
                        h = h_new
                        acf = acf_new
                        current_max = new_max
                        escaped = True
                        break
                if not escaped:
                    break
                else:
                    continue
            
            # Apply the best flip
            h[i_best] *= -1
            acf = compute_acf(h)
            new_max = get_max_sidelobe(acf)
            
            # Accept if it doesn't increase the global max
            # Note: reducing the peak lag might increase another lag.
            # We only accept if the new global max is <= current max
            if new_max <= current_max:
                current_max = new_max
            else:
                # Revert
                h[i_best] *= -1
                # If we can't improve the peak lag without worsening the global max,
                # try to reduce a different high lag?
                # For simplicity, stop if we can't improve the global max via the peak lag.
                # Or try the second best lag.
                # Let's just break to avoid infinite loops in this simple heuristic.
                # A more sophisticated approach would try all top K lags.
                break
                
        return h, current_max

    # Generate initial candidates
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Flat spectrum initialization
        # Random phases
        phases = rng.uniform(0, 2 * np.pi, n_fft)
        F = np.exp(1j * phases)
        # Ensure conjugate symmetry for real signal
        F[0] = 1.0
        F[n_fft // 2] = 1.0 if n_fft % 2 == 0 else 1.0
        # Make it Hermitian
        F_conj = np.conj(F[::-1])
        F = (F + F_conj) / 2.0 # Rough symmetrization, might not be perfect
        # Better: generate random real signal, take FFT, set magnitude to 1, keep phase?
        # No, that's circular.
        
        # Alternative: Just random binary sequence is a valid baseline.
        # But the prompt asks for "inverse FFT of a target low-peak magnitude spectrum".
        # Let's try: h = sign(ifft(fft(random_binary)))? No.
        
        # Let's stick to the prompt's inspiration:
        # "initializing the 400-bit sequence from the inverse FFT of a target low-peak magnitude spectrum"
        # Target magnitude: Flat (all 1s).
        # Phase: Random.
        # h_cont = ifft(1 * exp(i * random_phases))
        # h_bin = sign(h_cont)
        
        # Ensure real:
        # F[k] = exp(i * theta_k)
        # theta_0 = 0
        # theta_{N-k} = -theta_k
        thetas = rng.uniform(-np.pi, np.pi, n_fft)
        thetas[0] = 0
        # Enforce symmetry for real ifft
        # For even n_fft:
        # F[k] = conj(F[n_fft-k])
        # theta[k] = -theta[n_fft-k]
        # We can just generate theta[1..n_fft//2-1] and mirror.
        # theta[0] = 0
        # theta[n_fft//2] = 0 (if even)
        
        theta_half = rng.uniform(-np.pi, np.pi, n_fft // 2)
        thetas[1:n_fft//2] = theta_half
        thetas[n_fft//2+1:] = -theta_half[::-1]
        if n_fft % 2 == 0:
            thetas[n_fft//2] = 0
            
        F = np.exp(1j * thetas)
        h_cont = np.fft.ifft(F).real
        h_bin = np.sign(h_cont)
        h_bin[h_bin == 0] = 1 # Handle zeros
        
        # Refine
        h_opt, val = local_search(h_bin)
        
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
            
        # Also try random binary as a fallback/comparison in early iterations
        if r < 5:
            h_rand = rng.choice([1, -1], N)
            h_opt_r, val_r = local_search(h_rand)
            if val_r < best_val:
                best_val = val_r
                best_h = h_opt_r.copy()

    # If no good result, return a random one
    if best_h is None:
        best_h = rng.choice([1, -1], N)
        
    return best_h
