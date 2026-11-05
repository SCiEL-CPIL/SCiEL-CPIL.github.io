import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    N = 32
    RANGE = 512
    
    def score_set(s):
        """Count distinct d > 0 such that a, a+d, a+2d all in s for some a."""
        s_set = set(s)
        distinct_d = set()
        # Optimization: iterate over pairs (a, b) with a < b
        # d = b - a
        # Check if a + 2d is in s_set
        # Note: a, a+d, a+2d implies a, b, c with b-a = c-b = d
        # So for every pair (a, c) in s, if (a+c)/2 is in s, then d = (c-a)/2 is a valid difference?
        # No, the definition is: a, a+d, a+2d.
        # Let x, y, z be in s. If y-x == z-y, then d = y-x is a valid difference.
        # So we just need to find all pairs (x, z) such that (x+z) is even and (x+z)/2 is in s.
        # Then d = (z-x)/2.
        
        # Alternative: Iterate all pairs (a, b) from s, a != b.
        # d = abs(b-a).
        # Check if a+2d or b+2d or a-2d etc. is in s?
        # The progression is a, a+d, a+2d.
        # So if we have x, y in s, and we guess d = y-x (assuming x < y), we check if x+2d is in s.
        # Or if we have x, z in s, and x < z, and (x+z) is even, let m = (x+z)/2.
        # If m is in s, then d = m-x = z-m is a valid difference.
        
        # The second method (checking midpoints) is O(N^2) and checks if the midpoint is in the set.
        # This is faster and cleaner.
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                x = s[i]
                z = s[j]
                if (x + z) % 2 == 0:
                    m = (x + z) // 2
                    if m in s_set:
                        d = m - x
                        if d > 0:
                            distinct_d.add(d)
        return len(distinct_d)

    def greedy_construct():
        """Construct a set by greedily adding elements that maximize distinct differences."""
        s = []
        s_set = set()
        current_d = set()
        
        # Pre-calculate potential candidates? No, just scan 1..512
        # To make it faster, we can sample candidates, but 512 is small.
        
        for _ in range(N):
            best_val = -1
            best_score = -1
            best_new_d = set()
            
            # Randomize the order of candidates to break ties and add diversity
            candidates = list(range(1, RANGE + 1))
            random.shuffle(candidates)
            
            for val in candidates:
                if val in s_set:
                    continue
                
                # Check score if we add val
                new_s = s + [val]
                new_s_set = s_set | {val}
                
                # We only need to check new progressions involving val
                # A progression (a, a+d, a+2d) is new if it involves val.
                # Cases:
                # 1. val is a: check if val+d, val+2d in new_s_set
                # 2. val is a+d: check if val-d, val+d in new_s_set
                # 3. val is a+2d: check if val-d, val-2d in new_s_set
                
                new_d = set(current_d) # Start with existing
                
                # Case 1: val is the start
                for b in new_s_set:
                    if b == val: continue
                    d = b - val
                    if d <= 0: continue
                    if val + 2 * d in new_s_set:
                        new_d.add(d)
                
                # Case 2: val is the middle
                for a in new_s_set:
                    if a == val: continue
                    d = val - a
                    if d <= 0: continue
                    if a + 2 * d in new_s_set:
                        new_d.add(d)
                
                # Case 3: val is the end
                for a in new_s_set:
                    if a == val: continue
                    d = val - a
                    if d <= 0: continue
                    if a - d in new_s_set: # a, a+d, a+2d=val => a+d = val-d
                        new_d.add(d)
                
                score = len(new_d)
                if score > best_score:
                    best_score = score
                    best_val = val
                    best_new_d = new_d
            
            if best_val != -1:
                s.append(best_val)
                s_set.add(best_val)
                current_d = best_new_d
            else:
                # Fallback if no improvement found (should not happen)
                s.append(random.randint(1, RANGE))
        
        return s

    def simulated_annealing_2swap(s_init, time_limit):
        """SA with 1-move and 2-swap moves."""
        s = list(s_init)
        current_score = score_set(s)
        best_s = list(s)
        best_score = current_score
        
        temp = 10.0
        cooling = 0.999
        min_temp = 0.1
        
        while time.time() < time_limit and temp > min_temp:
            move_type = random.random()
            
            if move_type < 0.7:
                # 2-Swap Move: Pick 2 indices, assign 2 new random values
                i, j = random.sample(range(N), 2)
                old_i, old_j = s[i], s[j]
                
                new_i = random.randint(1, RANGE)
                new_j = random.randint(1, RANGE)
                
                # Ensure distinctness
                if new_i == new_j:
                    continue
                if new_i == old_j or new_j == old_i:
                    # Swap the new values to avoid collision with the other old value?
                    # Or just retry
                    pass
                if new_i in s and new_i != old_i and new_i != old_j:
                    continue
                if new_j in s and new_j != old_i and new_j != old_j:
                    continue
                
                s[i] = new_i
                s[j] = new_j
                new_score = score_set(s)
                delta = new_score - current_score
                
                if delta > 0 or random.random() < pow(2.718281828, delta / temp):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_s = list(s)
                else:
                    s[i] = old_i
                    s[j] = old_j
            else:
                # 1-Move: Pick 1 index, assign 1 new random value
                i = random.randint(0, N - 1)
                old_val = s[i]
                new_val = random.randint(1, RANGE)
                
                if new_val == old_val:
                    continue
                if new_val in s:
                    continue
                
                s[i] = new_val
                new_score = score_set(s)
                delta = new_score - current_score
                
                if delta > 0 or random.random() < pow(2.718281828, delta / temp):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_s = list(s)
                else:
                    s[i] = old_val
            
            temp *= cooling
            
        return best_s, best_score

    # Phase 1: Greedy Construction
    # We do a few greedy runs with different random seeds/shuffles to get diverse starting points
    best_overall = None
    best_overall_score = 0
    
    # Run greedy a few times
    greedy_time_limit = start_time + 10
    inits = []
    for _ in range(3):
        if time.time() > greedy_time_limit:
            break
        g = greedy_construct()
        inits.append(g)
    
    # Also add a random init
    inits.append(random.sample(range(1, RANGE + 1), N))
    
    # Phase 2: SA on each init
    sa_time_per_run = 20
    
    for init in inits:
        if time.time() > deadline - 5:
            break
            
        sa_limit = min(time.time() + sa_time_per_run, deadline - 5)
        s_res, score_res = simulated_annealing_2swap(init, sa_limit)
        
        if score_res > best_overall_score:
            best_overall_score = score_res
            best_overall = s_res
            
        # If we hit a very high score, we can stop early? 
        # Max possible distinct d is 255 (since max d is (512-1)/2 = 255.5 -> 255).
        # Actually max d is 255. If we reach 200+, it's likely optimal or near.
        if best_overall_score >= 180:
            break

    # Final check
    if best_overall is None:
        best_overall = list(range(1, N + 1))
        
    result = sorted(best_overall)
    if len(set(result)) != N or not all(1 <= x <= RANGE for x in result):
        result = list(range(1, N + 1))
        
    return result
