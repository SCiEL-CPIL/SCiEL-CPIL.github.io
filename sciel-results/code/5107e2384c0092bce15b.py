import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Initial set: all numbers 129 to 256
    # We verified that any pair in this range has lcm > 256
    S = list(range(129, 257))
    
    def get_score(S):
        return sum(1.0 / x for x in S)
    
    def is_compatible(x, S):
        for y in S:
            if x == y:
                continue
            if lcm(x, y) <= 256:
                return False
        return True
        
    best_S = S[:]
    best_score = get_score(S)
    
    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding elements
        for x in range(2, 129):
            if x not in S:
                if is_compatible(x, S):
                    new_S = S + [x]
                    new_score = get_score(new_S)
                    if new_score > best_score:
                        best_score = new_score
                        best_S = new_S
                        S = new_S
                        improved = True
        
        # Try swapping: remove y from S, add x not in S
        # To save time, only try swaps if we have time
        if time.time() < deadline - 10.0:
            for y in S:
                if time.time() > deadline:
                    break
                S_minus_y = [z for z in S if z != y]
                for x in range(2, 257):
                    if x in S:
                        continue
                    if is_compatible(x, S_minus_y):
                        new_S = S_minus_y + [x]
                        new_score = get_score(new_S)
                        if new_score > best_score:
                            best_score = new_score
                            best_S = new_S
                            S = new_S
                            improved = True
                            break # Restart loop with new S
                if improved:
                    break
    
    return best_S
