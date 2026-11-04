import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def thue_morse(n):
        seq = [1]
        while len(seq) < n:
            seq += [-x for x in seq]
        return np.array(seq[:n], dtype=np.int8)

    def get_score(sig):
        # Compute aperiodic autocorrelation
        # np.correlate is O(N^2)
        ac = np.correlate(sig, sig, mode='full')
        return np.max(np.abs(ac))

    best_sigma = thue_morse(N)
    best_score = get_score(best_sigma)
    
    # We will use a combination of local search and random restarts
    # To make local search faster, we can precompute parts, but N=400 is small.
    
    # Strategy:
    # 1. Local search on current sigma.
    # 2. If stuck, random restart.
    
    sigma = best_sigma.copy()
    
    while time.time() < deadline:
        # Local Search Phase
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_delta = 0
            best_idx = -1
            current_score = get_score(sigma)
            
            for i in range(N):
                sigma[i] = -sigma[i]
                new_score = get_score(sigma)
                delta = current_score - new_score
                if delta > best_delta:
                    best_delta = delta
                    best_idx = i
                sigma[i] = -sigma[i] # revert
                
            if best_idx != -1:
                sigma[best_idx] = -sigma[best_idx]
                new_score = get_score(sigma)
                if new_score < best_score:
                    best_score = new_score
                    best_sigma = sigma.copy()
                improved = True
            else:
                # Stuck in local minimum
                break
        
        # If we are stuck and have time, do a random restart
        if time.time() < deadline - 2.0:
            # Generate a new random start or perturbed Thue-Morse
            # Let's try a random sequence
            sigma = np.random.choice([-1, 1], size=N).astype(np.int8)
            current_score = get_score(sigma)
            
            # Quick local search on this new start
            for _ in range(50): # Limited local search steps
                if time.time() > deadline - 1.0:
                    break
                improved = False
                best_delta = 0
                best_idx = -1
                c_score = get_score(sigma)
                
                # Sample a subset of indices to speed up restart local search
                # or just do full pass if fast enough
                for i in range(N):
                    sigma[i] = -sigma[i]
                    n_score = get_score(sigma)
                    delta = c_score - n_score
                    if delta > best_delta:
                        best_delta = delta
                        best_idx = i
                    sigma[i] = -sigma[i]
                    
                if best_idx != -1:
                    sigma[best_idx] = -sigma[best_idx]
                    improved = True
                else:
                    break
                    
            current_score = get_score(sigma)
            if current_score < best_score:
                best_score = current_score
                best_sigma = sigma.copy()
            else:
                # If we didn't improve, we can just keep the best found so far.
                # The loop will continue, but we might want to vary the restart strategy.
                pass

    return best_sigma
