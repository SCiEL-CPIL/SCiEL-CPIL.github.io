import time
import random

N = 256

def construct():
    best_size = N + 1
    best_set = []
    start_time = time.time()
    deadline = start_time + 25.0  # Safety margin
    
    # Precompute all pairs? No, just check on the fly.
    
    while time.time() < deadline:
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        S = []
        sums = set()
        
        for x in candidates:
            # Check if x can be added
            valid = True
            # New sums: 2x, x+s for s in S
            if 2 * x in sums:
                valid = False
            else:
                for s in S:
                    if x + s in sums:
                        valid = False
                        break
            
            if valid:
                S.append(x)
                sums.add(2 * x)
                for s in S[:-1]:
                    sums.add(x + s)
                
                # Check if maximal
                # If any y not in S can be added, not maximal
                maximal = True
                for y in candidates:
                    if y in S:
                        continue
                    # Check if y can be added to current S
                    if 2 * y in sums:
                        continue
                    can_add = True
                    for s in S:
                        if y + s in sums:
                            can_add = False
                            break
                    if can_add:
                        maximal = False
                        break
                
                if maximal:
                    if len(S) < best_size:
                        best_size = len(S)
                        best_set = S[:]
                    break
        
    return best_set
