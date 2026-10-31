import numpy as np
import time

def construct():
    N = 400
    deadline = time.time() + 110  # 10s margin
    
    best_x = np.ones(N)
    best_max_side = np.inf
    best_J = np.inf
    
    def calc_metrics(x):
        # C[0] is lag 1, ..., C[N-2] is lag N-1
        c_full = np.correlate(x, x, mode='full')
        # Center is N-1
        c_side = np.concatenate([c_full[:N-1], c_full[N:]])
        # Actually c_full[N-1] is lag 0.
        # Lags are -(N-1) ... -1, 0, 1 ... N-1.
        # c_full indices:
        # 0: lag -(N-1)
        # N-1: lag 0
        # 2N-2: lag N-1
        # Symmetric, so we can take c_full[N:] which is lags 1..N-1.
        c_pos = c_full[N:] 
        max_side = np.max(np.abs(c_pos))
        J = np.sum(c_pos**2)
        return max_side, J, c_pos

    # Initial best
    max_s, J_s, _ = calc_metrics(best_x)
    best_max_side = max_s
    best_J = J_s
    
    while time.time() < deadline:
        x = np.random.choice([-1, 1], N)
        max_s, J, c_pos = calc_metrics(x)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_delta_J = 0
            best_i = -1
            best_new_c = None
            best_new_J = J
            best_new_max = max_s
            
            # Evaluate all flips
            for i in range(N):
                # Calculate delta_C
                delta_C = np.zeros(N-1)
                
                # Part 1: k < N-i
                # k = 1 ... N-1-i
                # x indices: i+1 ... N-1
                if N - i > 1:
                    delta_C[:N-1-i] += -2 * x[i+1:]
                
                # Part 2: k <= i
                # k = 1 ... i
                # x indices: i-1 ... 0 (reversed)
                if i > 0:
                    delta_C[:i] += -2 * x[i-1::-1]
                
                # Calculate delta_J
                # Delta J = sum( (C+D)^2 - C^2 ) = sum( 2 C D + D^2 )
                dJ = 2 * np.dot(c_pos, delta_C) + np.dot(delta_C, delta_C)
                
                if dJ < best_delta_J:
                    best_delta_J = dJ
                    best_i = i
                    # We don't need to update c_pos immediately, just store the best candidate
                    # But we need to check if it improves Max Side Lobe too?
                    # The prompt minimizes Max Abs Autoconv.
                    # Minimizing J is a proxy.
                    # We should prefer moves that reduce Max Side Lobe.
                    # Let's compute new max side for the best J candidate at the end of the loop?
                    # Or just accept if J decreases?
                    # If J decreases, Max Side Lobe usually decreases or stays same.
                    # To be safe, let's track the best Max Side Lobe directly.
                    
                    # However, calculating new max side for every i is expensive (O(N) each, total O(N^2)).
                    # We already do O(N) for dJ.
                    # Let's just minimize J.
                    pass
            
            if best_i != -1:
                # Apply flip
                x[best_i] *= -1
                # Recalculate c_pos and J
                max_s, J, c_pos = calc_metrics(x)
                improved = True
                
                if max_s < best_max_side:
                    best_max_side = max_s
                    best_x = x.copy()
                    best_J = J

    return best_x
