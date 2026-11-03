<<<<<<< SEARCH
def block_simulated_annealing(h, start_time, max_time):
    """Block-level simulated annealing with peak-directed perturbations."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    n_fft = 2 * N
    
    # Precompute current FFT for faster evaluation
    current_fft = np.fft.rfft(current_h, n=n_fft)
    current_conv = np.fft.irfft(current_fft * current_fft, n=n_fft)[:2*N-1]
    current_val = np.max(current_conv) / (N * N)
    
    # Cooling schedule
    T_start = 0.15
    T_end = 0.0005
    T_current = T_start
    
    # Block sizes to try
    block_sizes = [4, 8, 16, 32, 64, 128]
    
    no_improve = 0
    max_no_improve = 120
    total_steps = 0
    
    while time.time() - start_time < max_time:
        total_steps += 1
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        # Determine peak location for directed perturbations
        peak_idx = np.argmax(current_conv)
        
        # Pick a random block, biased toward peak region
        bs = block_sizes[np.random.randint(len(block_sizes))]
        n_blocks = N // bs
        if n_blocks < 2:
            bs = 4
            n_blocks = N // bs
        
        # 30% chance to pick a block near the peak
        if np.random.random() < 0.3:
            # Map conv peak index to block index
            # conv has length 2*N-1, peak_idx maps to time offset
            time_offset = peak_idx - (N - 1)
            center_block = max(0, min(n_blocks - 1, (time_offset + N//2) // bs))
            block_idx = max(0, min(n_blocks - 1, center_block + np.random.randint(-2, 3)))
        else:
            block_idx = np.random.randint(0, n_blocks)
        
        start_idx = block_idx * bs
        end_idx = start_idx + bs
        
        # Generate perturbation
        mode = np.random.randint(5)
        
        h_new = current_h.copy()
        if mode == 0:
            # Scale block
            scale = np.random.uniform(0.5, 1.5)
            h_new[start_idx:end_idx] *= scale
        elif mode == 1:
            # Shift mass within block
            half = bs // 2
            delta = np.random.uniform(-0.3, 0.3) * np.mean(current_h[start_idx:end_idx])
            h_new[start_idx:start_idx+half] -= delta
            h_new[start_idx+half:end_idx] += delta
        elif mode == 2:
            # Add/remove Gaussian bump
            center = start_idx + bs // 2
            sigma = max(2, bs // 4)
            x = np.arange(start_idx, end_idx)
            bump = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
            amp = np.random.uniform(-0.25, 0.25) * np.mean(current_h)
            h_new[start_idx:end_idx] += amp * bump
        elif mode == 3:
            # Smooth the block with neighbors (low-pass)
            if start_idx >= 2 and end_idx <= N - 2:
                left = np.mean(current_h[start_idx-2:start_idx])
                right = np.mean(current_h[end_idx:end_idx+2])
                h_new[start_idx:end_idx] = 0.7 * current_h[start_idx:end_idx] + 0.15 * left + 0.15 * right
            else:
                h_new[start_idx:end_idx] *= np.random.uniform(0.8, 1.2)
        else:
            # Random reweight: multiply by a smooth random factor
            x = np.linspace(0, 1, bs)
            factor = 1.0 + np.random.uniform(-0.3, 0.3) * np.cos(np.pi * x)
            h_new[start_idx:end_idx] *= factor
        
        h_new = np.maximum(h_new, 0.0)
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            continue
        h_new = h_new * (N / s_new)
        
        # Fast evaluation using FFT
        new_fft = np.fft.rfft(h_new, n=n_fft)
        new_conv = np.fft.irfft(new_fft * new_fft, n=n_fft)[:2*N-1]
        val_new = np.max(new_conv) / (N * N)
        
        # Accept if better, or with probability exp(-(val_new - val_cur)/T)
        if val_new < current_val or np.random.random() < np.exp(-(val_new - current_val) / max(T_current, 1e-10)):
            current_h = h_new
            current_fft = new_fft
            current_conv = new_conv
            current_val = val_new
            no_improve = 0
        else:
            no_improve += 1
        
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
            no_improve = 0
        
        # Cool down with geometric schedule
        cooling_rate = (T_end / T_start) ** (1.0 / max(800, total_steps))
        T_current = max(T_current * cooling_rate, T_end)
        
        if no_improve > max_no_improve:
            # Reset to best and restart with higher temperature
            current_h = best_h.copy()
            current_fft = np.fft.rfft(current_h, n=n_fft)
            current_conv = np.fft.irfft(current_fft * current_fft, n=n_fft)[:2*N-1]
            current_val = np.max(current_conv) / (N * N)
            T_current = T_start * 0.6
            no_improve = 0
    
    return best_h, best_val
=======
def block_simulated_annealing(h, start_time, max_time):
    """Block-level simulated annealing with peak-directed perturbations
    and adaptive temperature scheduling."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    n_fft = 2 * N
    conv_len = 2 * N - 1
    
    # Precompute current FFT for faster evaluation
    current_fft = np.fft.rfft(current_h, n=n_fft)
    current_conv = np.fft.irfft(current_fft * current_fft, n=n_fft)[:conv_len]
    current_val = np.max(current_conv) / (N * N)
    
    # Cooling schedule - faster cooling for more effective search
    T_start = 0.12
    T_end = 0.0003
    T_current = T_start
    
    # Block sizes to try (weighted toward medium sizes for better balance)
    block_sizes = [4, 8, 8, 16, 16, 32, 32, 64, 128]
    
    no_improve = 0
    max_no_improve = 100
    total_steps = 0
    best_val_history = [best_val]
    
    while time.time() - start_time < max_time:
        total_steps += 1
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        # Determine peak location for directed perturbations
        peak_idx = np.argmax(current_conv)
        
        # Pick a random block, biased toward peak region
        bs = block_sizes[np.random.randint(len(block_sizes))]
        n_blocks = N // bs
        if n_blocks < 2:
            bs = 4
            n_blocks = N // bs
        
        # 35% chance to pick a block near the peak (increased bias)
        if np.random.random() < 0.35:
            # Map conv peak index to block index
            time_offset = peak_idx - (N - 1)
            center_block = max(0, min(n_blocks - 1, (time_offset + N//2) // bs))
            block_idx = max(0, min(n_blocks - 1, center_block + np.random.randint(-3, 4)))
        else:
            block_idx = np.random.randint(0, n_blocks)
        
        start_idx = block_idx * bs
        end_idx = start_idx + bs
        
        # Generate perturbation - 6 modes for more diversity
        mode = np.random.randint(6)
        
        h_new = current_h.copy()
        block_mean = np.mean(current_h[start_idx:end_idx])
        
        if mode == 0:
            # Scale block (asymmetric: more likely to reduce peak regions)
            if peak_idx > 0 and peak_idx < conv_len:
                # If block is near peak, bias toward reduction
                dist_to_peak = abs(start_idx + bs//2 - (peak_idx - (N-1) + N//2))
                if dist_to_peak < N // 4:
                    scale = np.random.uniform(0.4, 1.3)
                else:
                    scale = np.random.uniform(0.5, 1.5)
            else:
                scale = np.random.uniform(0.5, 1.5)
            h_new[start_idx:end_idx] *= scale
        elif mode == 1:
            # Shift mass within block
            half = bs // 2
            delta = np.random.uniform(-0.3, 0.3) * block_mean
            h_new[start_idx:start_idx+half] -= delta
            h_new[start_idx+half:end_idx] += delta
        elif mode == 2:
            # Add/remove Gaussian bump
            center = start_idx + bs // 2
            sigma = max(2, bs // 4)
            x = np.arange(start_idx, end_idx)
            bump = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
            amp = np.random.uniform(-0.3, 0.3) * block_mean
            h_new[start_idx:end_idx] += amp * bump
        elif mode == 3:
            # Smooth the block with neighbors (low-pass)
            if start_idx >= 2 and end_idx <= N - 2:
                left = np.mean(current_h[start_idx-2:start_idx])
                right = np.mean(current_h[end_idx:end_idx+2])
                blend = np.random.uniform(0.1, 0.3)
                h_new[start_idx:end_idx] = (1 - 2*blend) * current_h[start_idx:end_idx] + blend * left + blend * right
            else:
                h_new[start_idx:end_idx] *= np.random.uniform(0.8, 1.2)
        elif mode == 4:
            # Random reweight: multiply by a smooth random factor
            x = np.linspace(0, 1, bs)
            factor = 1.0 + np.random.uniform(-0.35, 0.35) * np.cos(np.pi * x)
            h_new[start_idx:end_idx] *= factor
        else:
            # Swap mass between this block and a nearby block
            if start_idx > 0:
                src_start = max(0, start_idx - bs)
                src_end = start_idx
                swap_amt = np.random.uniform(-0.2, 0.2) * block_mean
                h_new[start_idx:end_idx] -= swap_amt
                h_new[src_start:src_end] += swap_amt
            elif end_idx < N:
                dst_start = end_idx
                dst_end = min(N, end_idx + bs)
                swap_amt = np.random.uniform(-0.2, 0.2) * block_mean
                h_new[start_idx:end_idx] += swap_amt
                h_new[dst_start:dst_end] -= swap_amt
            else:
                h_new[start_idx:end_idx] *= np.random.uniform(0.7, 1.3)
        
        h_new = np.maximum(h_new, 0.0)
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            continue
        h_new = h_new * (N / s_new)
        
        # Fast evaluation using FFT
        new_fft = np.fft.rfft(h_new, n=n_fft)
        new_conv = np.fft.irfft(new_fft * new_fft, n=n_fft)[:conv_len]
        val_new = np.max(new_conv) / (N * N)
        
        # Accept if better, or with probability exp(-(val_new - val_cur)/T)
        delta_val = val_new - current_val
        if delta_val < 0 or np.random.random() < np.exp(-delta_val / max(T_current, 1e-10)):
            current_h = h_new
            current_fft = new_fft
            current_conv = new_conv
            current_val = val_new
            no_improve = 0
        else:
            no_improve += 1
        
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
            no_improve = 0
            best_val_history.append(best_val)
        
        # Adaptive cooling: cool faster when improving, slower when stuck
        if no_improve < 20:
            cool_factor = 0.995
        elif no_improve < 50:
            cool_factor = 0.998
        else:
            cool_factor = 0.9995
        
        T_current = max(T_current * cool_factor, T_end)
        
        if no_improve > max_no_improve:
            # Reset to best and restart with moderate temperature
            current_h = best_h.copy()
            current_fft = np.fft.rfft(current_h, n=n_fft)
            current_conv = np.fft.irfft(current_fft * current_fft, n=n_fft)[:conv_len]
            current_val = np.max(current_conv) / (N * N)
            T_current = max(T_start * 0.5, T_current * 2)
            no_improve = 0
    
    return best_h, best_val
>>>>>>> REPLACE
