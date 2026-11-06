import time
import random

def construct():
    N = 17
    MIN_DIST = 6
    DEADLINE = 85.0  # Leave 5s margin
    
    start_time = time.time()
    
    # 1. Generate the 256-word linear code
    G = [
        0b11111111111111110,
        0b11110001101010010,
        0b11010100010101100,
        0b11001010101011000,
        0b10101100110100100,
        0b10100011011001010,
        0b10010111001110100,
        0b01001110101111001
    ]
    
    linear_code = []
    for mask in range(256):
        v = 0
        for i in range(8):
            if (mask >> i) & 1:
                v ^= G[i]
        linear_code.append(v)
        
    # Verify min distance
    min_d = 17
    for i in range(256):
        for j in range(i+1, 256):
            d = (linear_code[i] ^ linear_code[j]).bit_count()
            if d < min_d:
                min_d = d
    if min_d < 6:
        linear_code = [] # Fallback

    best_code = linear_code
    best_size = len(linear_code)
    
    # 2. Greedy Search
    for restart in range(50):
        if time.time() - start_time > DEADLINE:
            break
            
        random.seed(restart)
        
        candidates = list(range(2**17))
        random.shuffle(candidates)
        
        current_code = []
        
        for cand in candidates:
            if time.time() - start_time > DEADLINE:
                break
                
            compatible = True
            for w in current_code:
                if (cand ^ w).bit_count() < MIN_DIST:
                    compatible = False
                    break
            
            if compatible:
                current_code.append(cand)
                
        if len(current_code) > best_size:
            best_size = len(current_code)
            best_code = current_code
            
    return best_code
