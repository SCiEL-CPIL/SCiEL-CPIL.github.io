import numpy as np
import time

N = 400
DEADLINE = 115.0  # Safety margin below 150s evaluator limit

def _compute_r3(h):
    """
    Compute R3[k] = sum_i h[i] h[i+k] h[i+2k] for k = 0 .. N//2
    Using FFT for initialization speed.
    """
    h_float = h.astype(np.float64)
    size = 2 * N + 1
    h_fft = np.fft.fft(h_float, n=size)
    r3_fft = h_fft * h_fft * h_fft
    r3_full = np.fft.ifft(r3_fft).real
    return r3_full[:N+1]

def _score(h):
    r3 = _compute_r3(h)
    return np.max(np.abs(r3))

def _evaluate_block_flip(h, start, length):
    """
    Evaluate the change in score if we flip a block of signs from h[start] to h[start+length-1].
    Returns (new_score, delta_r3) where delta_r3 is the change in the R3 array.
    """
    # Create a copy of h with the block flipped
    h_new = h.copy()
    h_new[start:start+length] *= -1.0
    
    # Compute new R3
    new_r3 = _compute_r3(h_new)
    new_score = np.max(np.abs(new_r3))
    
    return new_score, new_r3

def _simulated_annealing(h, start_temp, end_temp, cooling_rate, steps):
    """
    Simulated annealing with block flips.
    """
    current_h = h.copy()
    current_score = _score(current_h)
    best_h = current_h.copy()
    best_score = current_score
    
    n_lags = N // 2 + 1
    
    for step in range(steps):
        # Temperature
        temp = start_temp * (cooling_rate ** step)
        
        # Choose a random block flip
        # Block size: random between 1 and max_block
        max_block = min(32, N // 4)
        block_len = np.random.randint(1, max_block + 1)
        start = np.random.randint(0, N - block_len + 1)
        
        # Evaluate the flip
        new_score, new_r3 = _evaluate_block_flip(current_h, start, block_len)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0 or np.random.random() < np.exp(-delta / (temp + 1e-9)):
            current_h[start:start+block_len] *= -1.0
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
        
        # Time check
        if time.time() - start_temp > DEADLINE:
            break
    
    return best_h, best_score

def construct():
    start_time = time.time()
    
    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE
    
    best_h = np.random.choice([-1.0, 1.0], size=N)
    best_score = _score(best_h)
    
    # Try a few different initializations
    for init in range(5):
        if not elapsed_ok():
            break
        
        if init == 0:
            h = np.random.choice([-1.0, 1.0], size=N)
        elif init == 1:
            # Sine-based initialization
            t = np.linspace(0, 2 * np.pi, N)
            h = np.sign(np.sin(t))
            h[h == 0] = 1.0
        elif init == 2:
            # Random with bias
            h = np.random.choice([-1.0, 1.0], size=N, p=[0.5, 0.5])
        elif init == 3:
            # Block-based initialization
            block_size = 4
            n_blocks = N // block_size
            blocks = np.random.choice([-1.0, 1.0], size=n_blocks)
            h = np.repeat(blocks, block_size)
        else:
            h = np.random.choice([-1.0, 1.0], size=N)
        
        score = _score(h)
        
        # Run simulated annealing
        h_sa, score_sa = _simulated_annealing(h, start_temp=10.0, end_temp=0.1, cooling_rate=0.999, steps=5000)
        
        if score_sa < best_score:
            best_score = score_sa
            best_h = h_sa.copy()
        
        if not elapsed_ok():
            break
    
    return best_h

if __name__ == "__main__":
    h = construct()
    print("Constructed:", h[:20])
    print("Score:", _score(h))
