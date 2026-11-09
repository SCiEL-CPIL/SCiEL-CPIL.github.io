"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Function to compute the score: max overlap of h with shifted (1-h)
    def get_score(h):
        g = 1.0 - h
        # Cross correlation: sum h[i] * g[i+k]
        # np.correlate(h, g, mode='full') computes this for all shifts
        corr = np.correlate(h, g, mode='full')
        return np.max(corr)
    
    def sa_run(h_init, time_budget):
        """Run simulated annealing starting from h_init for given time budget."""
        h = h_init.copy()
        current_score = get_score(h)
        best_h = h.copy()
        best_score = current_score
        
        # Simulated Annealing parameters
        T = 1.0
        T_min = 0.001
        alpha = 0.99995  # Slower cooling for more exploration
        
        t_start = time.time()
        while time.time() - t_start < time_budget:
            # Pick two random indices to swap
            i = np.random.randint(0, N)
            j = np.random.randint(0, N)
            
            # Skip if same index or same value (swap does nothing)
            if i == j or h[i] == h[j]:
                continue
                
            # Perform swap
            h[i], h[j] = h[j], h[i]
            
            new_score = get_score(h)
            
            # Accept if better or with probability exp(-delta/T)
            delta = new_score - current_score
            if delta < 0 or (T > 0 and np.random.rand() < np.exp(-delta / T)):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
            else:
                # Revert swap
                h[i], h[j] = h[j], h[i]
                
            T *= alpha
            if T < T_min:
                T = T_min
        
        return best_h, best_score
    
    # Generate multiple diverse initializations
    def make_binary_init(seed_val=None):
        """Create a binary sequence with exactly N/2 ones."""
        if seed_val is not None:
            rng = np.random.RandomState(seed_val)
        else:
            rng = np.random
        indices = rng.choice(N, N // 2, replace=False)
        h = np.zeros(N, dtype=np.float64)
        h[indices] = 1.0
        return h
    
    # Structured initialization: alternating pattern
    h_alt = np.zeros(N, dtype=np.float64)
    h_alt[0::2] = 1.0  # 1,0,1,0,...
    
    # Thue-Morse-like initialization (balanced sequence)
    def thue_morse(n):
        """Generate Thue-Morse sequence of length n."""
        t = [0]
        while len(t) < n:
            t += [1 - x for x in t]
        return t[:n]
    
    h_tm = np.array(thue_morse(N), dtype=np.float64)
    # Ensure exactly N/2 ones
    num_ones = np.sum(h_tm)
    if num_ones != N // 2:
        # Adjust to have exactly N/2 ones
        indices = np.where(h_tm == 1)[0]
        zeros = np.where(h_tm == 0)[0]
        if num_ones > N // 2:
            # Remove some ones
            remove = np.random.choice(indices, num_ones - N // 2, replace=False)
            h_tm[remove] = 0
        else:
            # Add some ones
            add = np.random.choice(zeros, N // 2 - num_ones, replace=False)
            h_tm[add] = 1
    
    # Random initialization
    h_rand = make_binary_init()
    
    # Run SA with multiple initializations and keep the best
    inits = [h_alt, h_tm, h_rand]
    best_h = None
    best_score = np.inf
    
    # Allocate time: 70% for main SA runs, 30% for polish
    main_time = 80.0
    polish_time = 15.0
    
    # Run SA on each initialization
    for h_init in inits:
        if time.time() > deadline - polish_time:
            break
        remaining = min(main_time, deadline - polish_time - time.time())
        if remaining <= 0:
            break
        h_sol, score = sa_run(h_init, remaining)
        if score < best_score:
            best_score = score
            best_h = h_sol
    
    # Polish phase: fine-tune the best solution
    if best_h is not None and time.time() < deadline - 2.0:
        h_polish = best_h.copy()
        current_score = get_score(h_polish)
        T = 0.05
        T_min = 0.001
        alpha = 0.9999
        t_start = time.time()
        polish_deadline = time.time() + polish_time
        
        while time.time() < polish_deadline:
            i = np.random.randint(0, N)
            j = np.random.randint(0, N)
            
            if i == j or h_polish[i] == h_polish[j]:
                continue
                
            h_polish[i], h_polish[j] = h_polish[j], h_polish[i]
            new_score = get_score(h_polish)
            
            delta = new_score - current_score
            if delta < 0 or (T > 0 and np.random.rand() < np.exp(-delta / T)):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = h_polish.copy()
            else:
                h_polish[i], h_polish[j] = h_polish[j], h_polish[i]
                
            T *= alpha
            if T < T_min:
                T = T_min
    
    return best_h
# EVOLVE-BLOCK-END
