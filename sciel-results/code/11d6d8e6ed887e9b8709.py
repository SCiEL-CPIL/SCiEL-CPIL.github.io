import time
import random

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAXV = 1000000
N = len(DIVISORS)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    best_span = MAXV
    best_result = None
    
    def get_span(reps):
        return max(reps) - min(reps)
        
    def random_initial():
        # Pick a random center
        # The optimal solution is likely to have values within a small window.
        # Let's pick a center between 10 and 1000000
        center = random.randint(10, MAXV - 100)
        reps = []
        used = set()
        # Shuffle divisors to break symmetry in initialization
        indices = list(range(N))
        random.shuffle(indices)
        
        # Assign in random order
        for idx in indices:
            d = DIVISORS[idx]
            # Find a multiple of d near center
            k = center // d
            if k < 1:
                k = 1
            val = k * d
            
            found = False
            for offset in range(0, 100):
                v1 = val + offset * d
                if v1 <= MAXV and v1 not in used:
                    reps.append((idx, v1))
                    used.add(v1)
                    found = True
                    break
                v2 = val - offset * d
                if v2 >= 1 and v2 not in used:
                    reps.append((idx, v2))
                    used.add(v2)
                    found = True
                    break
            
            if not found:
                # Fallback
                for k in range(1, MAXV // d + 1):
                    v = k * d
                    if v not in used:
                        reps.append((idx, v))
                        used.add(v)
                        found = True
                        break
                
            if not found:
                return None
                
        res = [0] * N
        for idx, val in reps:
            res[idx] = val
        return res

    def hill_climb(reps):
        reps = reps[:]
        current_span = get_span(reps)
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            indices = list(range(N))
            random.shuffle(indices)
            
            for i in indices:
                d = DIVISORS[i]
                current_val = reps[i]
                
                # Try lowering
                # Only useful if current_val is not the minimum (or if lowering it doesn't increase span)
                # We just check if span decreases.
                new_val = current_val - d
                while new_val >= 1:
                    collision = False
                    for j in range(N):
                        if j != i and reps[j] == new_val:
                            collision = True
                            break
                            
                    if not collision:
                        new_reps = reps[:]
                        new_reps[i] = new_val
                        new_span = get_span(new_reps)
                        
                        if new_span < current_span:
                            reps = new_reps
                            current_span = new_span
                            improved = True
                            current_val = new_val
                            new_val -= d
                        else:
                            break 
                    new_val -= d
                    if time.time() > deadline:
                        return reps, current_span
                        
                # Try raising
                current_val = reps[i]
                new_val = current_val + d
                while new_val <= MAXV:
                    collision = False
                    for j in range(N):
                        if j != i and reps[j] == new_val:
                            collision = True
                            break
                            
                    if not collision:
                        new_reps = reps[:]
                        new_reps[i] = new_val
                        new_span = get_span(new_reps)
                        
                        if new_span < current_span:
                            reps = new_reps
                            current_span = new_span
                            improved = True
                            current_val = new_val
                            new_val += d
                        else:
                            break 
                    new_val += d
                    if time.time() > deadline:
                        return reps, current_span
                        
        return reps, current_span

    # Main Loop
    while time.time() < deadline:
        reps = random_initial()
        if reps is None:
            continue
            
        reps, span = hill_climb(reps)
        
        if span < best_span:
            best_span = span
            best_result = reps[:]
            
        if best_span <= 2:
             # Can't get better than 1 (distinct), so 2 is very good.
             # We can continue to see if we find 1, but 1 is impossible for distinct integers?
             # Yes, distinct integers means min < max, so span >= 1.
             # Span 1 is possible (e.g., 10, 11).
             # If we find 1, we can stop? Or keep looking for 0? 0 is impossible.
             # So if we find 1, it's the theoretical minimum.
             # But wait, are there constraints that prevent span 1?
             # We need 10 distinct multiples.
             # If span is 1, the numbers must be k, k+1.
             # Only 2 numbers. We need 10.
             # So span must be at least 9 (10 distinct integers in a window of length S requires S >= 9).
             # So min possible span is 9.
             if best_span <= 9:
                 # 9 is the theoretical minimum for 10 distinct integers.
                 # If we find 9, we can stop?
                 # It might be impossible due to divisibility constraints.
                 # But it's a strong lower bound.
                 # Let's keep searching a bit more to be sure, or break if we want to save time.
                 # Given the time limit, breaking on 9 is risky if 9 is impossible and 10 is possible.
                 # But if 9 is found, it's optimal.
                 # If 9 is not found, we might find 10.
                 # Let's not break immediately, but we can reduce search effort?
                 # No, just let it run.
                 pass

    if best_result is None:
        return DIVISORS[:]
        
    return [int(x) for x in best_result]
