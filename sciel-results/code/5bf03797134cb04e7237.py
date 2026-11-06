import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_set = None
    best_score = 0
    
    def get_score(S):
        S_set = set(S)
        diffs = set()
        # Use list for iteration to avoid set iteration overhead? 
        # Actually set is fine for membership. List for iteration.
        S_list = list(S)
        for i in range(len(S_list)):
            u = S_list[i]
            for j in range(i + 1, len(S_list)):
                v = S_list[j]
                d = v - u
                # Check if u-d or v+d is in S
                if (u - d) in S_set or (v + d) in S_set:
                    diffs.add(d)
        return len(diffs)
    
    def random_set():
        return random.sample(range(1, 513), 32)
    
    # Run multiple independent searches or one long SA?
    # One long SA with restarts is usually good.
    
    for restart in range(100):  # Max restarts, but break on time
        if time.time() > deadline:
            break
            
        current_set = random_set()
        current_score = get_score(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = current_set.copy()
            
        temp = 50.0
        min_temp = 1.0
        cooling = 0.99
        
        # Limit steps per restart to avoid hanging on one bad restart
        max_steps = 5000
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            # Pick an element to move
            idx = random.randint(0, 31)
            old_val = current_set[idx]
            
            # Pick a new value
            new_val = random.randint(1, 512)
            if new_val in current_set:
                continue
                
            # Apply move
            current_set[idx] = new_val
            new_score = get_score(current_set)
            
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718, delta / temp):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set.copy()
            else:
                # Revert
                current_set[idx] = old_val
                
            temp *= cooling
            if temp < min_temp:
                break
                
    if best_set is None:
        return list(range(1, 33))
        
    return best_set
