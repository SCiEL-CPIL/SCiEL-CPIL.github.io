import time

def construct():
    S = 123457
    deadline = time.time() + 80  # 80s safety margin
    
    # Compute largest prime factor for all numbers up to S
    P = [1] * (S + 1)
    for i in range(2, S + 1):
        if P[i] == 1:  # i is prime
            for j in range(i, S + 1, i):
                P[j] = i
    
    best_score = float('inf')
    best_a = 1
    best_b = S - 1
    
    # Linear search with deadline check
    for a in range(1, S // 2 + 1):
        if time.time() > deadline:
            break
        b = S - a
        pa = P[a]
        pb = P[b]
        score = pa if pa > pb else pb
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
            # If we found a perfect smooth pair, we can't do better
            if best_score == 1:
                break
    
    return [best_a, best_b]
