import time

def construct():
    target = 123457
    deadline = time.time() + 80
    
    # Sieve for Largest Prime Factor
    lpf = [0] * (target + 1)
    lpf[1] = 1
    for i in range(2, target + 1):
        if lpf[i] == 0: # i is prime
            for j in range(i, target + 1, i):
                lpf[j] = i
                
    best_a = 1
    best_b = target - 1
    best_score = max(lpf[best_a], lpf[best_b])
    
    half_target = target // 2
    for a in range(2, half_target + 1):
        if time.time() > deadline:
            break
        b = target - a
        score = max(lpf[a], lpf[b])
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
            
    return [best_a, best_b]
