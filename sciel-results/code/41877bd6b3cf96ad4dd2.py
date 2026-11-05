import time
import math
import random
import numpy as np

def _max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    signs: list or numpy array of +1/-1.
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    max_abs = 0
    
    # Iterate over step d
    for d in range(1, n + 1):
        # For each residue class r in 0..d-1
        for r in range(min(d, n)):
            # Extract subsequence: indices r, r+d, r+2d, ...
            sub = signs_arr[r::d]
            m = len(sub)
            if m <= 1:
                if 1 > max_abs:
                    max_abs = 1
                continue
            
            # Calculate max subarray sum and min subarray sum
            cum = np.cumsum(sub)
            cum_ext = np.concatenate(([0], cum))
            
            min_prefix = np.minimum.accumulate(cum_ext)
            max_prefix = np.maximum.accumulate(cum_ext)
            
            max_sub = np.max(cum_ext[1:] - min_prefix[:-1])
            min_sub = np.min(cum_ext[1:] - max_prefix[:-1])
            
            current_max = max(abs(max_sub), abs(min_sub))
            if current_max > max_abs:
                max_abs = current_max
                
    return max_abs

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 256
    best_signs = None
    best_disc = float('inf')
    
    # Helper to evaluate and update best
    def evaluate_and_update(signs, label=""):
        nonlocal best_signs, best_disc
        disc = _max_ap_discrepancy(signs)
        if disc < best_disc:
            best_disc = disc
            best_signs = signs[:]
        return disc
    
    # 1. Generate Initial Candidates
    candidates = []
    
    # Thue-Morse
    tm = []
    for i in range(n):
        cnt = bin(i).count('1')
        tm.append(1 if cnt % 2 == 0 else -1)
    candidates.append(tm)
    
    # Walsh-Hadamard (Rademacher functions)
    for k in [1, 2, 3, 4]:
        wh = []
        for i in range(n):
            wh.append(1 if (i >> k) & 1 == 0 else -1)
        candidates.append(wh)
        
    # Random
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        rng = random.Random(seed)
        rand = [rng.choice((-1, 1)) for _ in range(n)]
        candidates.append(rand)
        
    # Sine wave
    for freq in [0.5, 0.7, 1.0, 1.5, 2.0]:
        sine = [1 if math.sin(2 * math.pi * freq * i / n) >= 0 else -1 for i in range(n)]
        candidates.append(sine)
        
    # Evaluate initial candidates
    for c in candidates:
        if time.time() > deadline - 20:
            break
        evaluate_and_update(c)
        
    if best_signs is None:
        return [1] * n  # Fallback

    # 2. Spectral Band-Limiting Optimization
    # Strategy: 
    # - Start with the best candidate found so far.
    # - Convert to float array.
    # - Perform iterative projection:
    #   a) Compute DFT.
    #   b) Soft-threshold high frequency components (suppress variance).
    #   c) Inverse DFT.
    #   d) Hard threshold to +/-1 signs.
    #   e) Evaluate.
    # - Use multiple "bandwidths" (cutoff frequencies) and thresholds.
    
    base_signs = np.array(best_signs, dtype=np.float64)
    
    # Try different spectral profiles
    # A lower cutoff means more smoothing (lower freq only)
    # A higher threshold means more aggressive suppression
    
    profiles = []
    for cutoff in [4, 8, 12, 16, 24, 32]:
        for thresh in [0.5, 1.0, 2.0, 3.0]:
            profiles.append((cutoff, thresh))
            
    # Add some "sharp" profiles to preserve structure
    profiles.append((64, 0.1))
    profiles.append((128, 0.0)) # No change essentially
    
    rng = random.Random(123)
    
    current_signs = base_signs.copy()
    
    # Iterative refinement
    # We will iterate over profiles, applying the spectral filter to the current best
    # and potentially doing a few local search steps if it improves.
    
    # To maximize exploration within time, we will run this process for several "generations"
    # starting from different initial points (best candidate + random perturbations)
    
    num_generations = 10
    gen_signs_list = [base_signs.copy()]
    for _ in range(num_generations - 1):
        # Perturb the best sign sequence slightly
        perturbed = base_signs.copy()
        for _ in range(5):
            idx = rng.randint(0, n - 1)
            perturbed[idx] *= -1
        gen_signs_list.append(perturbed)
        
    for gen_idx, start_sig in enumerate(gen_signs_list):
        if time.time() > deadline - 5:
            break
            
        cur = start_sig.copy()
        
        # Iterate through spectral profiles
        for cutoff, thresh in profiles:
            if time.time() > deadline - 5:
                break
                
            # Apply spectral filter
            # 1. DFT
            fft_vals = np.fft.fft(cur)
            
            # 2. Soft thresholding
            # We want to keep low frequencies strong, reduce high frequencies.
            # Simple approach: multiply high freq components by a decay factor or zero them out.
            # Let's use a window: 1.0 for k < cutoff, linear decay to 0 for k < cutoff + 10.
            mask = np.ones(n)
            decay_start = cutoff
            decay_end = cutoff + 10
            for k in range(decay_start, min(decay_end, n)):
                # Linear decay from 1 to 0
                alpha = 1.0 - (k - decay_start) / (decay_end - decay_start)
                mask[k] = alpha
                # Also mirror for negative frequencies (n-k)
                mask[n-k] = alpha
            
            # Apply threshold: if magnitude < thresh, set to 0? 
            # Or just scale? The prompt says "soft-thresholding".
            # Standard soft threshold: sign(x) * max(|x| - thresh, 0)
            # But we want to preserve the phase. 
            # Let's apply a gain based on frequency and magnitude.
            
            # Actually, let's stick to the "band-limiting" idea primarily.
            # Suppress high frequencies to reduce local variance.
            filtered_fft = fft_vals * mask
            
            # Additional soft threshold on magnitude to remove noise
            magnitudes = np.abs(filtered_fft)
            signs_fft = np.sign(filtered_fft)
            # Soft threshold: new_mag = max(mag - thresh, 0)
            new_mags = np.maximum(magnitudes - thresh, 0)
            filtered_fft = signs_fft * new_mags
            
            # 3. Inverse DFT
            # Ensure real part
            inv_fft = np.real(np.fft.ifft(filtered_fft))
            
            # 4. Hard threshold to +/-1
            # Normalize to avoid bias
            # If all zeros, keep previous
            if np.allclose(inv_fft, 0):
                continue
                
            # Center the signal? The signs are +/-1. The mean should be close to 0.
            # Let's just threshold at 0.
            new_signs_float = np.sign(inv_fft)
            # If zero, assign random or keep old? Keep old is safer for continuity.
            new_signs_float[new_signs_float == 0] = cur[new_signs_float == 0]
            
            # Convert to int list for evaluation
            new_signs_int = new_signs_float.astype(int)
            
            # Evaluate
            disc = _max_ap_discrepancy(new_signs_int)
            
            if disc < best_disc:
                best_disc = disc
                best_signs = new_signs_int[:]
                cur = new_signs_float.copy()
            else:
                # If it didn't improve globally, maybe it's a good local step?
                # For simplicity and speed, we only accept if it improves the global best.
                # This makes the search greedy in the spectral space.
                pass
            
    # 3. Final Local Search Polish
    # Use the best spectral result as a starting point for a quick local search
    # to fix any remaining high-discrepancy APs.
    
    if time.time() < deadline - 5:
        signs = np.array(best_signs, dtype=np.int8)
        rng2 = random.Random(999)
        
        # Quick hill climbing
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for _ in range(50): # Try 50 random flips per pass
                if time.time() > deadline - 2:
                    break
                idx = rng2.randint(0, n - 1)
                old_val = signs[idx]
                signs[idx] = -old_val
                
                disc = _max_ap_discrepancy(signs)
                if disc < best_disc:
                    best_disc = disc
                    best_signs = signs.copy()
                    improved = True
                else:
                    signs[idx] = old_val
                    if not improved:
                        # If we haven't improved in this batch, break to save time?
                        # Or continue? Let's break if we've tried enough.
                        pass
                        
    return best_signs

# Verification
if __name__ == "__main__":
    signs = construct()
    print(len(signs))
    print(_max_ap_discrepancy(signs))
