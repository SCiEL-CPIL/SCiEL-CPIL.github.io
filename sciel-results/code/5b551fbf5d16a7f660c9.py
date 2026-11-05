import numpy as np

N = 400

def construct():
    # Simulated Annealing approach
    # Objective: minimize V = abs(2 * N * max(conv(h,h)) / sum(h)**2)
    
    def compute_val(h):
        S = np.sum(h)
        if np.abs(S) < 1e-9:
            return np.inf
        c = np.convolve(h, h)
        m = np.max(c)
        return abs(2 * N * m / (S * S))

    best_h = None
    best_val = np.inf
    
    # Number of restarts
    n_restarts = 20
    
    for r in range(n_restarts):
        # Random initialization
        h = np.random.randn(N)
        S = np.sum(h)
        if np.abs(S) < 1e-9:
            h[0] = 1
            S = np.sum(h)
        h = h / S
        
        current_val = compute_val(h)
        if current_val < best_val:
            best_val = current_val
            best_h = h.copy()
            
        # Simulated Annealing parameters
        T = 0.1
        T_min = 0.001
        cooling_rate = 0.999
        n_steps = 2000
        
        for step in range(n_steps):
            # Generate candidate
            # Perturb a random subset of indices
            n_perturb = np.random.randint(1, 5)
            indices = np.random.choice(N, n_perturb, replace=False)
            h_new = h.copy()
            h_new[indices] += np.random.randn(n_perturb) * 0.1
            
            # Normalize
            S_new = np.sum(h_new)
            if np.abs(S_new) < 1e-9:
                continue
            h_new = h_new / S_new
            
            # Evaluate
            new_val = compute_val(h_new)
            
            # Acceptance criterion
            if new_val < current_val:
                h = h_new
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
            else:
                delta = new_val - current_val
                # Avoid overflow in exp
                if delta / T < 500:
                    if np.random.rand() < np.exp(-delta / T):
                        h = h_new
                        current_val = new_val
            
            T *= cooling_rate
            if T < T_min:
                T = T_min

    return best_h
