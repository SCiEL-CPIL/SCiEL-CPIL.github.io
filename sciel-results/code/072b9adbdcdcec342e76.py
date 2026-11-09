import numpy as np
import math
import time
import random

def construct():
    start_time = time.time()
    time_limit = 118.0  # Leave 2s buffer for safety, but try to use more time
    
    # Helper to evaluate score efficiently using FFT
    def evaluate(U):
        if not U or 0 not in U:
            return -np.inf
        
        U = sorted(set(U))
        M = max(U)
        if M == 0:
            return 1.0
        
        # Determine FFT size
        # We need to compute sumset (0..2M) and diffset (-M..M)
        # Linear convolution of two length M+1 vectors requires size >= 2*(M+1)-1 = 2M+1
        # To be safe and fast, use next power of 2 >= 2*M + 1
        n_fft = 1
        while n_fft < 2 * M + 1:
            n_fft *= 2
        
        v = np.zeros(n_fft, dtype=np.float64)
        for u in U:
            v[u] = 1.0
            
        v_fft = np.fft.fft(v)
        
        # Sumset: |U+U|
        # Convolution v * v
        sum_conv = np.fft.ifft(v_fft * v_fft).real
        # Valid indices for sum are 0 to 2M
        # Due to floating point errors, we check > 0.5 (since min non-zero count is 1)
        sumset_size = np.sum(sum_conv[:2*M+1] > 0.5)
        
        # Diffset: |U-U|
        # Autocorrelation: v * conj(v) -> ifft gives correlation
        # Correlation at lag k is sum v[i] v[i+k]
        # For real signals, autocorrelation is symmetric.
        # The peak is at index 0 (lag 0).
        # Positive lags 1..M are in [1:M+1]
        # Negative lags -1..-M are in [n_fft-1 : n_fft-M-1 : -1]
        
        auto_corr = np.fft.ifft(v_fft * np.conj(v_fft)).real
        
        # Count non-zero lags
        # Lag 0 is always |U| > 0
        pos_lags = auto_corr[1:M+1]
        neg_lags = auto_corr[n_fft-1:n_fft-M-1:-1]
        
        pos_count = np.sum(pos_lags > 0.5)
        neg_count = np.sum(neg_lags > 0.5)
        
        diffset_size = pos_count + neg_count + 1
        
        if sumset_size == 0:
            return -np.inf
            
        ratio = diffset_size / sumset_size
        if ratio <= 1:
            # Score will be <= 1, but we still return it
            pass
            
        denom = math.log(2 * M + 1)
        if denom == 0:
            return 1.0
            
        score = 1 + math.log(ratio) / denom
        return score

    best_overall = -np.inf
    best_U_overall = [0, 1, 3]
    
    # Strategy:
    # 1. Try a few different initial structures (geometric, random sparse, etc.)
    # 2. Use Simulated Annealing to refine
    # 3. Allow variable size N and max value M
    
    # Initial candidates
    candidates = []
    
    # Candidate 1: Small set
    candidates.append([0, 1, 3, 7])
    
    # Candidate 2: Geometric progression (powers of 2)
    U_geo = [0]
    curr = 1
    for i in range(10):
        U_geo.append(curr)
        curr *= 2
    candidates.append(U_geo[:10])
    
    # Candidate 3: Random sparse set
    U_rand = [0]
    curr = 1
    for i in range(15):
        curr += random.randint(1, 5)
        U_rand.append(curr)
    candidates.append(U_rand)
    
    # Candidate 4: Larger random set
    U_rand2 = [0]
    curr = 1
    for i in range(25):
        curr += random.randint(1, 3)
        U_rand2.append(curr)
    candidates.append(U_rand2)
    
    # Evaluate initial candidates
    for U_init in candidates:
        score = evaluate(U_init)
        if score > best_overall:
            best_overall = score
            best_U_overall = U_init.copy()
    
    # Run SA on the best candidate and potentially others if time permits
    # We will run SA on multiple tracks or just refine the best one?
    # Refining the best one is usually better. But maybe different basins?
    # Let's run SA on the top 2 initial candidates.
    
    tracks = []
    scored_cands = [(evaluate(U), U) for U in candidates]
    scored_cands.sort(key=lambda x: x[0], reverse=True)
    
    for score, U in scored_cands[:2]:
        tracks.append((score, U.copy()))
    
    # Time allocation per track
    # Simple: split remaining time among tracks
    # Or just run one track until time runs out?
    # Given the variability, let's run the best track for most of the time,
    # and if time remains, try others.
    
    # Let's implement a single SA loop that can be interrupted by time.
    # We'll iterate through tracks, giving each a portion of the time.
    
    total_time = time_limit
    time_per_track = total_time / len(tracks) if tracks else total_limit
    
    for idx, (cur_score, U) in enumerate(tracks):
        track_start = time.time()
        track_deadline = track_start + time_per_track
        
        T = 1.0
        T_min = 0.001
        cooling_rate = 0.995 # Slower cooling for more exploration
        
        while T > T_min:
            if time.time() > track_deadline or time.time() > start_time + time_limit:
                break
            
            U_temp = U.copy()
            M = max(U_temp) if U_temp else 0
            N = len(U_temp)
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Shift an element
                if N > 1:
                    idx_elem = random.randint(1, N-1)
                    # Shift size proportional to current max or random small
                    shift_max = max(5, M // 10)
                    shift = random.randint(-shift_max, shift_max)
                    new_val = U_temp[idx_elem] + shift
                    if new_val >= 0 and new_val not in U_temp:
                        U_temp[idx_elem] = new_val
                        U_temp = sorted(U_temp)
                    else:
                        continue
            elif move_type < 0.7:
                # Add an element
                if N < 100: # Cap size to keep M reasonable
                    # Add near existing elements or extend range
                    if random.random() < 0.5:
                        # Extend range
                        new_val = M + random.randint(1, max(5, M//10))
                    else:
                        # Fill gap
                        if M > 0:
                            new_val = random.randint(1, M)
                        else:
                            new_val = 1
                    
                    if new_val not in U_temp:
                        U_temp.append(new_val)
                        U_temp = sorted(U_temp)
                    else:
                        continue
            else:
                # Remove an element
                if N > 1:
                    idx_elem = random.randint(1, N-1)
                    U_temp.pop(idx_elem)
            
            new_score = evaluate(U_temp)
            
            if new_score > cur_score:
                U = U_temp
                cur_score = new_score
                if cur_score > best_overall:
                    best_overall = cur_score
                    best_U_overall = U.copy()
            else:
                delta = cur_score - new_score
                if delta < 0:
                    prob = 1.0
                else:
                    prob = math.exp(-delta / T)
                
                if random.random() < prob:
                    U = U_temp
                    cur_score = new_score
            
            T *= cooling_rate
            
            # Optional: restart if stuck?
            # Simple restart logic: if no improvement for many steps, reset T?
            # For now, keep it simple.

    if best_overall > -np.inf:
        return best_U_overall
    return [0, 1, 3]
