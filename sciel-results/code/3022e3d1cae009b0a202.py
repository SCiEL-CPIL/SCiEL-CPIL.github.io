import numpy as np
import time

N = 400

def compute_R(x):
    R = np.zeros(N)
    for k in range(N):
        R[k] = np.dot(x[:N-k], x[k:N])
    return R

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_score = np.inf
    best_x = np.ones(N, dtype=np.int8)
    
    # Precompute index arrays for delta calculation
    # idx1[i] are the k values for the "forward" part (i+k < N)
    # idx2[i] are the k values for the "backward" part (i-k >= 0)
    idx1 = [None] * N
    idx2 = [None] * N
    for i in range(N):
        if N - i > 1:
            idx1[i] = np.arange(1, N - i)
        else:
            idx1[i] = np.array([], dtype=int)
        if i > 0:
            idx2[i] = np.arange(1, i + 1)
        else:
            idx2[i] = np.array([], dtype=int)
            
    candidates = []
    candidates.append(np.ones(N, dtype=np.int8))
    candidates.append(np.array([1, -1] * (N//2), dtype=np.int8))
    
    # Thue-Morse
    tm = np.zeros(N, dtype=np.int8)
    for i in range(N):
        c = bin(i).count('1')
        tm[i] = 1 if c % 2 == 0 else -1
    candidates.append(tm)
    
    # Random
    np.random.seed(42)
    candidates.append(np.random.choice([1, -1], N).astype(np.int8))
    
    for x_init in candidates:
        if time.time() > deadline:
            break
            
        x = x_init.copy()
        R = compute_R(x)
        score = R[0]**3 + 2.0 * np.sum(R[1:]**3)
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
            
        # Simulated Annealing
        T = 500.0
        cooling = 0.999
        min_T = 0.5
        steps = 0
        max_steps = 2000
        
        while steps < max_steps:
            if time.time() > deadline:
                break
                
            i = np.random.randint(N)
            
            # Calculate Delta Score
            delta_s = 0.0
            
            # Forward part
            if len(idx1[i]) > 0:
                ks = idx1[i]
                # x[i+1 : N] corresponds to x[i+k] for k=1..N-1-i
                neighbors = x[i+1:N]
                deltas = -2.0 * x[i] * neighbors
                old_R = R[ks]
                new_R = old_R + deltas
                delta_s += 2.0 * np.sum(new_R**3 - old_R**3)
                
            # Backward part
            if len(idx2[i]) > 0:
                ks = idx2[i]
                # x[:i] reversed corresponds to x[i-k] for k=1..i
                neighbors = x[:i][::-1]
                deltas = -2.0 * x[i] * neighbors
                old_R = R[ks]
                new_R = old_R + deltas
                delta_s += 2.0 * np.sum(new_R**3 - old_R**3)
                
            # Metropolis Acceptance
            if delta_s < 0 or np.random.random() < np.exp(-delta_s / T):
                # Accept
                x[i] *= -1
                
                # Update R
                if len(idx1[i]) > 0:
                    ks = idx1[i]
                    neighbors = x[i+1:N]
                    # x[i] is now flipped. 
                    # Delta was calculated as -2 * x_old * x_neighbor.
                    # x_old = -x_new.
                    # Delta = 2 * x_new * x_neighbor.
                    R[ks] += 2.0 * x[i] * neighbors
                    
                if len(idx2[i]) > 0:
                    ks = idx2[i]
                    neighbors = x[:i][::-1]
                    R[ks] += 2.0 * x[i] * neighbors
                    
                score += delta_s
                
                if score < best_score:
                    best_score = score
                    best_x = x.copy()
                    
            T *= cooling
            if T < min_T:
                break
            steps += 1
            
    return best_x
