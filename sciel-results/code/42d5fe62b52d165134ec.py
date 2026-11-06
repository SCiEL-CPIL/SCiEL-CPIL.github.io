import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin
    
    N = 1000000
    N_MIN = 10000
    
    # Sieve for Largest Prime Factor
    # We need LPF for n and n+1, so up to N+1
    lpf = np.zeros(N + 2, dtype=np.int32)
    
    for i in range(2, N + 2):
        if lpf[i] == 0:
            lpf[i::i] = i
            
    # Convert to list for faster scalar access in the tight loop
    lpf_list = lpf.tolist()
    
    best_n = N_MIN
    best_score = float('inf')
    
    for n in range(N_MIN, N + 1):
        score = max(lpf_list[n], lpf_list[n+1])
        if score < best_score:
            best_score = score
            best_n = n
            if best_score == 5:
                # 5 is the theoretical minimum for n >= 10000
                # (3 is impossible, 4 is not prime)
                return best_n
                
    return best_n
