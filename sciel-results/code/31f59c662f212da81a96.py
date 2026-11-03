import numpy as np
import time

def generate_m_sequence(length, poly):
    n = len(poly)
    taps = poly
    reg = [1] * n
    seq = []
    for _ in range(length):
        seq.append(reg[0])
        fb = 0
        for t in taps:
            fb ^= reg[t]
        reg = reg[1:] + [fb]
    return np.array(seq) * 2 - 1

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    N = 400
    
    # Initial sequence
    m_seq = generate_m_sequence(511, [9, 4, 0])
    s = m_seq[:N].copy()
    
    def get_score(seq):
        ac = np.correlate(seq, seq, mode='full')
        return np.max(np.abs(ac))
        
    current_val = get_score(s)
    best_val = current_val
    best_s = s.copy()
    
    stagnation_counter = 0
    
    while time.time() < deadline:
        improved = False
        indices = np.random.permutation(N)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            old_val = s[idx]
            s[idx] = -old_val
            
            new_val = get_score(s)
            
            if new_val < best_val:
                best_val = new_val
                best_s = s.copy()
                improved = True
                stagnation_counter = 0
            else:
                s[idx] = old_val
        
        if not improved:
            stagnation_counter += 1
            # If stuck for 10 full passes, try a random restart
            if stagnation_counter >= 10:
                # Random restart
                s = np.random.choice([-1, 1], size=N)
                current_val = get_score(s)
                if current_val < best_val:
                    best_val = current_val
                    best_s = s.copy()
                    stagnation_counter = 0
                else:
                    # Keep best_s, but reset s to best_s to continue from best
                    s = best_s.copy()
                    stagnation_counter = 0

    return best_s
