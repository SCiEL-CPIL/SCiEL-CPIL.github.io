import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def score(a):
        conv = np.convolve(a, a, mode='full')
        return np.max(np.abs(conv))
    
    best_a = np.ones(N) / N
    best_score = score(best_a)
    
    np.random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Initialize: uniform + random perturbation
        a = np.ones(N) / N + np.random.randn(N) * 0.1
        s = a.sum()
        if abs(s) < 1e-10:
            continue
        a /= s
        cur_score = score(a)
        
        if cur_score < best_score:
            best_score = cur_score
            best_a = a.copy()
        
        # Simulated annealing
        T = 0.001
        T_min = 1e-8
        cooling = 0.999
        
        while time.time() < deadline and T > T_min:
            a_trial = a.copy()
            num_perturb = np.random.randint(1, 30)
            indices = np.random.choice(N, num_perturb, replace=False)
            a_trial[indices] += np.random.randn(num_perturb) * 0.05
            
            s = a_trial.sum()
            if abs(s) < 1e-10:
                continue
            a_trial /= s
            
            s_trial = score(a_trial)
            
            if s_trial < cur_score or np.random.rand() < np.exp(-(s_trial - cur_score) / T):
                a = a_trial
                cur_score = s_trial
                if cur_score < best_score:
                    best_score = cur_score
                    best_a = a.copy()
            
            T *= cooling
    
    return best_a
