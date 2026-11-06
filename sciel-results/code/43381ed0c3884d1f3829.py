import time
import random

def _score_set(S):
    count = 0
    for d in range(1, 256):
        if S & (S >> d) & (S >> (2 * d)):
            count += 1
    return count

def construct():
    deadline = time.time() + 140.0
    
    # Initial set: 1..32
    S = 0
    for i in range(1, 33):
        S |= (1 << i)
    
    current_score = _score_set(S)
    best_S = S
    best_score = current_score
    
    # Helper to get set bits
    def get_bits(mask):
        bits = []
        m = mask
        while m:
            lsb = m & -m
            bits.append(lsb.bit_length() - 1)
            m ^= lsb
        return bits

    # Quick heuristic improvement
    for _ in range(100):
        bits = get_bits(S)
        x = random.choice(bits)
        y = random.randint(1, 512)
        if x == y:
            continue
        if (S >> y) & 1:
            continue
        new_S = S ^ (1 << x) ^ (1 << y)
        new_score = _score_set(new_S)
        if new_score > current_score:
            S = new_S
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_S = S

    T = 5.0
    min_T = 0.1
    cooling = 0.99
    
    while time.time() < deadline:
        bits = get_bits(S)
        x = random.choice(bits)
        y = random.randint(1, 512)
        if x == y:
            continue
        if (S >> y) & 1:
            continue
        
        new_S = S ^ (1 << x) ^ (1 << y)
        new_score = _score_set(new_S)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < pow(2.718281828, -delta / T):
            S = new_S
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_S = S
        
        T *= cooling
        if T < min_T:
            T = min_T

    result = get_bits(best_S)
    return result
