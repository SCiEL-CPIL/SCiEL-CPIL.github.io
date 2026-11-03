import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    # Initialize with a random sequence of -1 and 1
    np.random.seed(123)
    x = np.random.choice([-1, 1], size=N)
    
    # Initial Correlation and Sum
    r = np.correlate(x, x, mode='full')
    S = np.sum(x)
    
    def get_score(r_vec, sum_val):
        if sum_val == 0:
            return float('inf')
        max_abs_r = np.max(np.abs(r_vec))
        return max_abs_r / (sum_val * sum_val)
    
    current_score = get_score(r, S)
    best_x = x.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T_start = 5.0
    T_end = 0.001
    cooling_rate = 0.99995 # Slower cooling for more exploration
    T = T_start
    
    # Pre-allocate buffers for incremental update
    size_corr = 2 * N - 1
    lags = np.arange(-(N-1), N)
    idx1 = np.empty(size_corr, dtype=int)
    idx2 = np.empty(size_corr, dtype=int)
    term1 = np.zeros(size_corr)
    term2 = np.zeros(size_corr)
    delta_r = np.zeros(size_corr)
    r_buf = np.zeros(size_corr)
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer for safety
    
    while time.time() - start_time < time_limit:
        # Pick random index to flip
        i = np.random.randint(N)
        xi = x[i]
        
        # Calculate indices for the update
        idx1[:] = i + lags
        idx2[:] = i - lags
        
        # Masks for valid indices
        mask1 = (idx1 >= 0) & (idx1 < N)
        mask2 = (idx2 >= 0) & (idx2 < N)
        
        # Compute terms
        term1.fill(0)
        term2.fill(0)
        
        # Efficiently update terms using boolean indexing
        term1[mask1] = x[idx1[mask1]]
        term2[mask2] = x[idx2[mask2]]
        
        # Calculate delta R
        # delta_r = -2 * xi * (term1 + term2)
        delta_r[:] = -2.0 * xi * (term1 + term2)
        
        # New R in buffer
        r_buf[:] = r
        r_buf += delta_r
        
        S_candidate = S - 2 * xi
        
        if S_candidate == 0:
            score_candidate = float('inf')
        else:
            score_candidate = np.max(np.abs(r_buf)) / (S_candidate * S_candidate)
        
        # Simulated Annealing Decision
        if score_candidate < current_score:
            # Always accept improvement
            x[i] = -xi
            r[:] = r_buf
            S = S_candidate
            current_score = score_candidate
            
            if score_candidate < best_score:
                best_score = score_candidate
                best_x = x.copy()
        else:
            # Accept worse move with probability exp(-delta / T)
            delta = score_candidate - current_score
            if delta < 0:
                accept = True
            else:
                # Avoid overflow in exp
                if delta / T > 500:
                    prob = 0.0
                else:
                    prob = np.exp(-delta / T)
                accept = (np.random.random() < prob)
            
            if accept:
                x[i] = -xi
                r[:] = r_buf
                S = S_candidate
                current_score = score_candidate
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
    return best_x
# EVOLVE-BLOCK-END
