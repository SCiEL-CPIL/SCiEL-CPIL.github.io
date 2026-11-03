"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_overlaps(h):
    """Compute overlap at each shift: sum_i h[i]*(1-h[(i+k)%N]) for k=1..N-1"""
    # overlap_k = sum h[i] - sum h[i]*h[(i+k)%N] = N/2 - autocorr[k]
    # We want to minimize max overlap = N/2 - min autocorr
    # So minimize max overlap <=> maximize min autocorr
    # Compute autocorrelation via FFT
    h_ext = np.concatenate([h, np.zeros(N)])
    H = np.fft.rfft(h_ext)
    acf = np.fft.irfft(H * np.conj(H))[:N]
    # acf[0] = sum h^2, acf[k] = sum h[i]*h[(i+k)%N]
    # Overlap at shift k (k>=1) = N/2 - acf[k]
    overlaps = N / 2.0 - acf[1:]
    return overlaps, acf

def _smooth_max(values, beta):
    """Log-sum-exp approximation of max"""
    m = np.max(values)
    if m == 0:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (values - m)))) / beta

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = None
    best_score = np.inf
    
    def evaluate(h):
        nonlocal best_h, best_score
        # Project h to [0,1] and fix sum
        h = np.clip(h, 0, 1)
        # Adjust to make sum exactly N/2
        diff = N / 2.0 - np.sum(h)
        if abs(diff) > 1e-12:
            # Distribute adjustment proportionally to avoid clipping issues
            room_up = 1.0 - h  # how much can go up
            room_down = h  # how much can go down
            if diff > 0:
                # Need to add diff total
                total_room = np.sum(room_up)
                if total_room < diff:
                    return None  # infeasible
                h = h + diff * room_up / total_room
            else:
                total_room = np.sum(room_down)
                if total_room < -diff:
                    return None
                h = h + diff * room_down / total_room
            h = np.clip(h, 0, 1)
            # Final correction for floating point
            h += (N / 2.0 - np.sum(h)) / N
            h = np.clip(h, 0, 1)
        
        overlaps, acf = _compute_overlaps(h)
        score = np.max(overlaps)
        if score < best_score:
            best_score = score
            best_h = h.copy()
        return h, score
    
    # Initialize with various starting points
    np.random.seed(42)
    
    # Try several random initializations
    initializations = []
    # Uniform
    initializations.append(np.full(N, 0.5))
    # Random
    for _ in range(5):
        h_init = np.random.uniform(0, 1, N)
        initializations.append(h_init)
    # Binary-like
    for _ in range(3):
        h_init = np.random.choice([0, 1], N, p=[0.5, 0.5]).astype(float)
        initializations.append(h_init)
    
    for h_init in initializations:
        if time.time() > deadline - 5:
            break
        
        h, _ = evaluate(h_init)
        if h is None:
            continue
        
        # Projected gradient descent with smooth max
        beta_start = 10.0
        beta_end = 1000.0
        n_iter = 500
        lr = 0.1
        
        for it in range(n_iter):
            if time.time() > deadline - 5:
                break
            
            # Anneal beta
            t = it / n_iter
            beta = beta_start * (beta_end / beta_start) ** t
            
            overlaps, acf = _compute_overlaps(h)
            score = _smooth_max(overlaps, beta)
            
            # Gradient of smooth max w.r.t. overlaps
            exp_vals = np.exp(beta * (overlaps - np.max(overlaps)))
            weights = exp_vals / np.sum(exp_vals)
            
            # d(overlap_k)/d(h[i]) for overlap_k = sum_j h[j]*(1-h[(j+k)%N])
            # = sum_j [delta_{i,j}*(1-h[(j+k)%N]) + h[j]*(-delta_{i,(j+k)%N})]
            # = (1-h[(i+k)%N]) - h[(i-k)%N]
            # So gradient w.r.t. h[i] = sum_k weights[k] * [(1-h[(i+k)%N]) - h[(i-k)%N]]
            
            grad = np.zeros(N)
            for k in range(1, N):
                wk = weights[k-1]
                # (i+k) mod N and (i-k) mod N
                grad += wk * ((1 - np.roll(h, -k)) - np.roll(h, k))
            
            # Projected gradient step
            h_new = h - lr * grad
            h_new = np.clip(h_new, 0, 1)
            
            # Fix sum
            diff = N / 2.0 - np.sum(h_new)
            if abs(diff) > 1e-10:
                h_new += diff / N
                h_new = np.clip(h_new, 0, 1)
                # Iterate to converge sum
                for _ in range(10):
                    diff = N / 2.0 - np.sum(h_new)
                    if abs(diff) < 1e-12:
                        break
                    h_new += diff / N
                    h_new = np.clip(h_new, 0, 1)
            
            h = h_new
            evaluate(h)
        
        # Final polish with high beta
        if time.time() > deadline - 3:
            break
        
        beta = 5000.0
        for it in range(100):
            if time.time() > deadline - 2:
                break
            
            overlaps, acf = _compute_overlaps(h)
            score = _smooth_max(overlaps, beta)
            
            exp_vals = np.exp(beta * (overlaps - np.max(overlaps)))
            weights = exp_vals / np.sum(exp_vals)
            
            grad = np.zeros(N)
            for k in range(1, N):
                wk = weights[k-1]
                grad += wk * ((1 - np.roll(h, -k)) - np.roll(h, k))
            
            h_new = h - 0.01 * grad
            h_new = np.clip(h_new, 0, 1)
            
            diff = N / 2.0 - np.sum(h_new)
            if abs(diff) > 1e-10:
                h_new += diff / N
                h_new = np.clip(h_new, 0, 1)
                for _ in range(10):
                    diff = N / 2.0 - np.sum(h_new)
                    if abs(diff) < 1e-12:
                        break
                    h_new += diff / N
                    h_new = np.clip(h_new, 0, 1)
            
            h = h_new
            evaluate(h)
    
    if best_h is None:
        return np.full(N, 0.5)
    
    # Final verification and correction
    best_h = np.clip(best_h, 0, 1)
    diff = N / 2.0 - np.sum(best_h)
    best_h += diff / N
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
