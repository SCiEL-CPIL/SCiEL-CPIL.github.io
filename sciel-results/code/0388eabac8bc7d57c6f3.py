import numpy as np

def construct():
    target = 123457
    lpf = np.zeros(target + 1, dtype=np.int32)
    lpf[1] = 1
    
    for i in range(2, target + 1):
        if lpf[i] == 0:
            lpf[i::i] = i
            
    # Scan
    best_a = 1
    best_b = target - 1
    best_score = max(lpf[best_a], lpf[best_b])
    
    half_target = target // 2
    for a in range(2, half_target + 1):
        b = target - a
        score = max(lpf[a], lpf[b])
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
        if best_score == 2: # Theoretical min for even sum? No, sum is odd. Min is 3?
            break
            
    return [int(best_a), int(best_b)]
