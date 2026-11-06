import random
import math

def construct():
    n = 256
    # Precompute 3-APs and 4-APs
    # Represent each AP as a tuple of indices (0-based)
    ap3s = []
    for d in range(1, n):
        for a in range(0, n - 2 * d):
            # a, a+d, a+2d
            ap3s.append((a, a + d, a + 2 * d))
            
    ap4s = []
    for d in range(1, n):
        for a in range(0, n - 3 * d):
            # a, a+d, a+2d, a+3d
            ap4s.append((a, a + d, a + 2 * d, a + 3 * d))
            
    # For fast lookup of which APs an index is in
    index_to_ap3 = [[] for _ in range(n)]
    for i, ap in enumerate(ap3s):
        for idx in ap:
            index_to_ap3[idx].append(i)
            
    index_to_ap4 = [[] for _ in range(n)]
    for i, ap in enumerate(ap4s):
        for idx in ap:
            index_to_ap4[idx].append(i)

    # Initialize state
    # Start with a random set, then repair 4-APs
    state = [0] * n
    # Heuristic initialization: include numbers based on some pattern or random
    # Let's try a random initialization with p=0.5
    for i in range(n):
        if random.random() < 0.5:
            state[i] = 1
            
    # Repair function: remove elements to break all 4-APs
    def repair(state):
        changed = True
        while changed:
            changed = False
            for ap in ap4s:
                if all(state[i] == 1 for i in ap):
                    # Remove one random element from this AP
                    # Prefer removing one that is in few 3-APs? 
                    # For simplicity, remove the first one or random
                    victim = random.choice(ap)
                    state[victim] = 0
                    changed = True
        return state

    state = repair(state)
    
    # Calculate current score
    def count_3aps(state):
        count = 0
        for ap in ap3s:
            if all(state[i] == 1 for i in ap):
                count += 1
        return count
        
    def count_4aps(state):
        count = 0
        for ap in ap4s:
            if all(state[i] == 1 for i in ap):
                count += 1
        return count

    current_3ap_count = count_3aps(state)
    current_4ap_count = count_4aps(state) # Should be 0 after repair
    
    # Simulated Annealing
    T = 100.0
    T_min = 0.01
    alpha = 0.9995
    penalty = 10000
    best_state = state[:]
    best_score = current_3ap_count - penalty * current_4ap_count
    
    import time
    start_time = time.time()
    max_time = 50.0 # Leave some buffer
    
    while T > T_min:
        if time.time() - start_time > max_time:
            break
            
        # Pick a random index to flip
        idx = random.randint(0, n - 1)
        old_val = state[idx]
        new_val = 1 - old_val
        
        # Calculate delta in 3-APs
        delta_3 = 0
        for ap_idx in index_to_ap3[idx]:
            ap = ap3s[ap_idx]
            # Check if the other 2 are present
            other_present = 0
            for i in ap:
                if i != idx and state[i] == 1:
                    other_present += 1
            if new_val == 1:
                # We are adding idx. If other 2 are present, we form a new 3-AP
                if other_present == 2:
                    delta_3 += 1
            else:
                # We are removing idx. If other 2 are present, we break a 3-AP
                if other_present == 2:
                    delta_3 -= 1
                    
        # Calculate delta in 4-APs
        delta_4 = 0
        for ap_idx in index_to_ap4[idx]:
            ap = ap4s[ap_idx]
            other_present = 0
            for i in ap:
                if i != idx and state[i] == 1:
                    other_present += 1
            if new_val == 1:
                if other_present == 3:
                    delta_4 += 1
            else:
                if other_present == 3:
                    delta_4 -= 1
        
        # Calculate delta in score
        delta_score = delta_3 - penalty * delta_4
        
        # Acceptance probability
        if delta_score >= 0:
            accept = True
        else:
            accept = random.random() < math.exp(delta_score / T)
            
        if accept:
            state[idx] = new_val
            current_3ap_count += delta_3
            current_4ap_count += delta_4
            current_score = current_3ap_count - penalty * current_4ap_count
            
            if current_score > best_score:
                best_score = current_score
                best_state = state[:]
                
        T *= alpha
        
    # Ensure the best state is valid (no 4-APs)
    # If the best state found during SA had 4-APs (which shouldn't happen if penalty is high enough 
    # and we started valid, but delta_4 could become positive if we accept a move that creates a 4-AP 
    # because delta_3 was very large? No, penalty is 10000, max delta_3 is small. 
    # However, it's possible to drift into invalid states if T is high. 
    # So we must repair the best state if it's invalid.)
    
    if count_4aps(best_state) > 0:
        best_state = repair(best_state)
        # Re-calculate score for the repaired state
        final_3ap_count = count_3aps(best_state)
        # The score we return is just the 3-AP count, as the task asks to maximize 3-APs 
        # subject to no 4-APs.
        
    # Convert 0-based index list to 1-based sorted set
    result = [i + 1 for i in range(n) if best_state[i] == 1]
    
    return result
