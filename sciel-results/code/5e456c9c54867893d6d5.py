import numpy as np
import math
import time
import random

def construct():
    # Start with a simple seed
    best_U = [0, 1, 3]
    best_score = -np.inf
    
    # Time limit
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    def evaluate(U):
        if not U:
            return -np.inf
        if 0 not in U:
            return -np.inf
        U = sorted(set(U))
        M = max(U)
        if M == 0:
            return 1.0
        
        # Determine FFT size
        # We need size >= 2*M + 1 for linear convolution/correlation
        # Next power of 2 is standard for FFT efficiency
        max_sum = 2 * M
        n_fft = 1
        while n_fft < max_sum + 1:
            n_fft *= 2
        
        v = np.zeros(n_fft, dtype=np.float64)
        # Vectorized assignment is faster than loop for large N, but N is small.
        # Direct assignment is fine.
        for u in U:
            v[u] = 1.0
            
        v_fft = np.fft.fft(v)
        
        # Sumset: |U+U|
        sum_conv = np.fft.ifft(v_fft * v_fft).real
        # Indices 0 to 2*M
        sumset_size = np.sum(sum_conv[:2*M+1] > 1e-9)
        
        # Difference set: |U-U|
        # Autocorrelation
        auto_corr = np.fft.ifft(v_fft * np.conj(v_fft)).real
        # Lags 1 to M
        pos_lags = auto_corr[1:M+1]
        # Lags -1 to -M (stored at n_fft-1 to n_fft-M)
        neg_lags = auto_corr[n_fft-1:n_fft-M-1:-1]
        
        pos_count = np.sum(pos_lags > 1e-9)
        neg_count = np.sum(neg_lags > 1e-9)
        diffset_size = pos_count + neg_count + 1
        
        if sumset_size == 0:
            return -np.inf
            
        ratio = diffset_size / sumset_size
        denom = math.log(2 * M + 1)
        if denom == 0:
            return 1.0
            
        score = 1 + math.log(ratio) / denom
        return score

    best_overall = -np.inf
    best_U_overall = [0, 1, 3]
    
    # Multi-start strategy
    # We will run SA for a few different initial configurations
    
    def run_sa(initial_U, time_budget):
        U = sorted(set(initial_U))
        if 0 not in U:
            U.append(0)
            U = sorted(U)
        
        current_score = evaluate(U)
        if current_score > best_overall:
            best_overall = current_score
            best_U_overall = U.copy()
            
        T = 1.0
        T_min = 1e-4
        cooling_rate = 0.995
        
        end_time = time.time() + time_budget
        
        while T > T_min and time.time() < end_time:
            # Perturbation
            move_type = random.random()
            U_temp = U.copy()
            valid_move = False
            
            if move_type < 0.4:
                # Shift an element (keep 0 fixed)
                if len(U_temp) > 1:
                    idx = random.randint(1, len(U_temp) - 1)
                    M = max(U_temp)
                    # Larger shifts to explore space
                    shift = random.randint(-max(1, M//10), max(1, M//10))
                    new_val = U_temp[idx] + shift
                    if new_val >= 0 and new_val not in U_temp:
                        U_temp[idx] = new_val
                        U_temp = sorted(U_temp)
                        valid_move = True
            elif move_type < 0.6:
                # Add element
                M = max(U_temp) if U_temp else 0
                # Add near existing or random in range
                if random.random() < 0.5:
                    new_val = random.randint(1, M + 10)
                else:
                    new_val = random.randint(1, M * 2 + 10)
                if new_val not in U_temp:
                    U_temp.append(new_val)
                    valid_move = True
            elif move_type < 0.8:
                # Remove element (keep 0)
                if len(U_temp) > 1:
                    idx = random.randint(1, len(U_temp) - 1)
                    U_temp.pop(idx)
                    valid_move = True
            else:
                # Replace element with random new value
                if len(U_temp) > 1:
                    idx = random.randint(1, len(U_temp) - 1)
                    M = max(U_temp)
                    new_val = random.randint(1, M + 20)
                    if new_val not in U_temp:
                        U_temp[idx] = new_val
                        U_temp = sorted(U_temp)
                        valid_move = True
            
            if not valid_move:
                # If move was invalid, try a small shift or skip
                # To avoid infinite loops on invalid moves, just skip this iteration
                # But we must update T? No, SA usually waits.
                # For efficiency, just continue to next iteration without cooling?
                # Or cool slightly.
                T *= cooling_rate
                continue
            
            new_score = evaluate(U_temp)
            
            if new_score > current_score:
                U = U_temp
                current_score = new_score
                if current_score > best_overall:
                    best_overall = current_score
                    best_U_overall = U.copy()
            else:
                delta = current_score - new_score
                if delta < 0:
                    prob = 1.0
                else:
                    prob = math.exp(-delta / T)
                if random.random() < prob:
                    U = U_temp
                    current_score = new_score
            
            T *= cooling_rate
            
        return U, current_score

    # Define several initial seeds
    seeds = [
        [0, 1, 3, 7, 15, 31], # Geometric-ish
        [0, 1, 2, 4, 8, 16, 32], # Powers of 2
        [0, 1, 3, 4, 6, 9], # Random small
        [0, 1, 2, 3, 5, 8, 13, 21], # Fibonacci-ish
        [0, 1, 10, 11, 20, 21, 30, 31], # Blocks
    ]
    
    # Also add random seeds
    for _ in range(5):
        N = random.randint(5, 15)
        U_seed = [0]
        for _ in range(N-1):
            U_seed.append(random.randint(1, 50))
        U_seed = sorted(set(U_seed))
        seeds.append(U_seed)
        
    # Allocate time
    # Give more time to later seeds if early ones are promising?
    # Or just split time evenly.
    # Let's run each for a fixed chunk, then a final long run on the best.
    
    num_seeds = len(seeds)
    # Reserve 30s for the final polish
    polish_time = 30.0
    remaining_time = time_limit - polish_time
    time_per_seed = remaining_time / num_seeds
    
    for i, seed in enumerate(seeds):
        if time.time() > time_limit - polish_time:
            break
            
        # Run SA
        U_res, score_res = run_sa(seed, time_per_seed)
        
        # Check if we improved
        # best_overall is updated inside run_sa
        
    # Final polish on the best found so far
    if best_U_overall:
        # Run SA on the best found with the remaining time
        run_sa(best_U_overall, time_limit - (time.time() - start_time))
        
    if best_overall > -np.inf:
        return best_U_overall
    return [0, 1, 3]
