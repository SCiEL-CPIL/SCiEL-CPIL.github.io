import time
import random

def score_set(S):
    # S is a set of integers 1..512
    # Convert to bitset
    # bit i corresponds to value i+1
    bs = 0
    for v in S:
        bs |= (1 << (v - 1))
    
    count = 0
    # Max difference d
    # Max value 512. Min value 1.
    # a, a+d, a+2d <= 512
    # 1 + 2d <= 512 => 2d <= 511 => d <= 255
    for d in range(1, 256):
        # Check if exists a such that a in S, a+d in S, a+2d in S
        # a in S  => bit (a-1) is 1 in bs
        # a+d in S => bit (a+d-1) is 1 in bs => bit (a-1) is 1 in (bs >> d)
        # a+2d in S => bit (a-1) is 1 in (bs >> 2d)
        
        # Intersection of bs, bs>>d, bs>>2d
        # If non-zero, there is at least one 'a'
        inter = bs & (bs >> d) & (bs >> (2*d))
        if inter != 0:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial set: random 32 distinct ints in 1..512
    S = set(random.sample(range(1, 513), 32))
    current_score = score_set(S)
    best_score = current_score
    best_S = set(S)
    
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    # Simulated Annealing
    while time.time() < deadline and T > min_T:
        # Generate neighbor
        # Pick x in S, y not in S
        x = random.choice(list(S))
        # Pick y from 1..512 not in S
        # To be fast, pick random y, if in S, retry? 
        # Or maintain a list of candidates? 
        # 512 is small. 
        y = random.randint(1, 512)
        if y in S:
            # Try a few times
            for _ in range(10):
                y = random.randint(1, 512)
                if y not in S:
                    break
            else:
                # If still in S, skip or handle
                continue
                
        S.remove(x)
        S.add(y)
        
        new_score = score_set(S)
        
        delta = new_score - current_score
        if delta > 0:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_S = set(S)
        else:
            # Accept with probability exp(-delta / T)
            # delta is negative
            p = random.random()
            if p < pow(2.718281828, delta / T):
                current_score = new_score
            else:
                # Reject: revert
                S.remove(y)
                S.add(x)
        
        T *= cooling
        
    # Final Polish: Hill Climbing
    # Just keep trying swaps that improve score
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in list(best_S):
            if time.time() > deadline:
                break
            for y in random.sample([v for v in range(1, 513) if v not in best_S], min(10, len([v for v in range(1, 513) if v not in best_S]))):
                new_S = set(best_S)
                new_S.remove(x)
                new_S.add(y)
                new_score = score_set(new_S)
                if new_score > best_score:
                    best_S = new_S
                    best_score = new_score
                    improved = True
                    break # Restart x loop or just update
            if improved:
                break # Restart x loop with new best_S
        
    return list(best_S)
