import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    np.random.seed(42)
    
    best_overall_s = None
    best_overall_E = np.inf
    
    # Multiple restarts
    max_restarts = 20
    restarts = 0
    
    while time.time() < deadline and restarts < max_restarts:
        restarts += 1
        s = np.random.choice([-1, 1], size=N)
        
        def compute_R(s):
            R = np.zeros(N)
            for k in range(1, N):
                R[k] = np.dot(s[:N-k], s[k:])
            return R
        
        R = compute_R(s)
        E = np.sum(R[1:]**2)
        
        best_s = s.copy()
        best_E = E
        
        # Simulated annealing
        T = 5.0
        cooling = 0.9995
        min_T = 0.001
        
        while time.time() < deadline - 2 and T > min_T:
            i = np.random.randint(N)
            old_s_i = s[i]
            
            dR = np.zeros(N)
            # j = i: affects R(k) for k where i+k < N
            for k in range(1, N - i):
                dR[k] += -2 * old_s_i * s[i + k]
            # j = i - k: affects R(k) for k where i-k >= 0
            for k in range(1, i + 1):
                dR[k] += -2 * old_s_i * s[i - k]
            
            dE = np.sum(2 * R * dR + dR**2)
            
            if dE < 0 or np.random.rand() < np.exp(-dE / T):
                s[i] = -s[i]
                R += dR
                E += dE
                if E < best_E:
                    best_E = E
                    best_s = s.copy()
            
            T *= cooling
        
        if best_E < best_overall_E:
            best_overall_E = best_E
            best_overall_s = best_s.copy()
    
    if best_overall_s is None:
        return np.ones(N)
    return best_overall_s
