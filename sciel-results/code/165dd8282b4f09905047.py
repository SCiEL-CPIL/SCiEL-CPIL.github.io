import time
import random

def _score(S):
    """
    Calculate the number of distinct d > 0 such that a, a+d, a+2d in S.
    Optimized using pair iteration.
    """
    S_set = set(S)
    valid_ds = set()
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        xi = S_list[i]
        for j in range(i + 1, n):
            xj = S_list[j]
            diff = xj - xi
            if diff % 2 == 0:
                d = diff // 2
                mid = xi + d
                if mid in S_set:
                    valid_ds.add(d)
    return len(valid_ds)

def _hill_climb(S, max_iters=1000, time_limit=None, start_time=0):
    """
    Perform hill climbing. Try all single-element moves.
    """
    current_score = _score(S)
    S = list(S)
    
    for _ in range(max_iters):
        if time_limit and (time.time() - start_time) > time_limit:
            break
            
        best_score = current_score
        best_move = None
        
        # Try all moves: remove x, add y
        for i in range(len(S)):
            x_remove = S[i]
            S_new = S[:i] + S[i+1:]
            # Try adding y from 1 to 512
            # To speed up, we can sample or iterate all. 512 is small.
            for y_add in range(1, 513):
                if y_add in S_new:
                    continue
                S_candidate = S_new + [y_add]
                # Optimization: Don't sort if not needed for score? 
                # _score doesn't require sorted input.
                sc = _score(S_candidate)
                if sc > best_score:
                    best_score = sc
                    best_move = (i, y_add)
        
        if best_move is None:
            # Local optimum reached. Try a random move to escape?
            # Or break.
            # Let's try a random perturbation
            if random.random() < 0.1:
                i = random.randint(0, len(S)-1)
                y_add = random.randint(1, 512)
                if y_add not in S:
                    S[i] = y_add
                    current_score = _score(S)
                else:
                    break
            else:
                break
        else:
            i, y_add = best_move
            S[i] = y_add
            current_score = best_score
            
    return S, current_score

def construct():
    start_time = time.time()
    time_limit = 100  # Seconds
    
    best_S = None
    best_score = 0
    
    # Initializations
    initializations = []
    
    # 1. Random sets
    for _ in range(5):
        initializations.append(random.sample(range(1, 513), 32))
        
    # 2. Dense block
    initializations.append(list(range(1, 33)))
    initializations.append(list(range(500, 532))[:32]) # Shifted
    
    # 3. Interleaved
    # 1, 33, 2, 34... ?
    base = list(range(1, 33))
    interleaved = []
    for i in range(16):
        interleaved.append(1 + i)
        interleaved.append(33 + i)
    initializations.append(interleaved)
    
    # 4. Structured spread
    # Powers of 2?
    p2 = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    while len(p2) < 32:
        p2.append(random.randint(1, 512))
    # Ensure distinct
    p2 = list(set(p2))
    while len(p2) < 32:
        p2.append(random.randint(1, 512))
    initializations.append(p2[:32])

    for init_S in initializations:
        if (time.time() - start_time) > time_limit:
            break
            
        # Hill climb
        S, sc = _hill_climb(init_S, max_iters=50, time_limit=time_limit - (time.time() - start_time), start_time=start_time)
        
        if sc > best_score:
            best_score = sc
            best_S = S
            
        # If we have time, do a bit of SA on the best found so far?
        # Or just keep hill climbing with random restarts.
        
    # Final polish: Run a few more hill climbs on the best solution with random perturbations
    if best_S and (time.time() - start_time) < time_limit:
        for _ in range(10):
            if (time.time() - start_time) > time_limit:
                break
            # Perturb best_S
            S_perturbed = list(best_S)
            for _ in range(3):
                idx = random.randint(0, 31)
                new_val = random.randint(1, 512)
                if new_val not in S_perturbed:
                    S_perturbed[idx] = new_val
            
            S_final, sc_final = _hill_climb(S_perturbed, max_iters=20, time_limit=time_limit - (time.time() - start_time), start_time=start_time)
            if sc_final > best_score:
                best_score = sc_final
                best_S = S_final
                
    if best_S is None:
        return list(range(1, 33))
        
    return best_S
