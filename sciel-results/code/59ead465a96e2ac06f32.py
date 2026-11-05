import random
import time

def get_differences(S):
    S_set = set(S)
    diffs = set()
    # Optimization: only check d up to max(S) - min(S)
    max_diff = max(S) - min(S)
    for d in range(1, max_diff + 1):
        # Check if there exists a in S such that a+d in S and a+2d in S
        # a must be in S, a+d in S, a+2d in S
        # So a is in S, a+d is in S, a+2d is in S
        # a is in S, and (a+d) is in S, and (a+2d) is in S
        # We can iterate a in S:
        found = False
        for a in S:
            if (a + d) in S_set and (a + 2 * d) in S_set:
                found = True
                break
        if found:
            diffs.add(d)
    return diffs

def count_small_diffs(D, threshold=15):
    return len([d for d in D if d <= threshold])

def run_search_heuristic(initial_S, time_budget):
    start_time = time.time()
    S = initial_S.copy()
    current_D = get_differences(S)
    current_score = len(current_D)
    best_S = S.copy()
    best_score = current_score
    best_D = current_D.copy()
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.99
    
    while time.time() - start_time < time_budget:
        new_S = S.copy()
        idx = random.randint(0, 31)
        old_val = new_S[idx]
        
        # Try to move to a random position
        # Bias towards positions that might create large differences?
        # Or just random.
        while True:
            new_val = random.randint(1, 512)
            if new_val not in new_S:
                break
        new_S[idx] = new_val
        
        new_D = get_differences(new_S)
        new_score = len(new_D)
        
        # Heuristic: Prefer moves that don't lose small differences
        small_curr = count_small_diffs(current_D)
        small_new = count_small_diffs(new_D)
        
        accept = False
        if new_score > current_score:
            # Accept if we gain score
            # But be careful if we lose many small diffs?
            # The directive says "dense in small integers", so preserving them is key.
            # If we gain 1 large diff but lose 5 small diffs, is it good?
            # Total score is just count. So +1 -5 = -4. Bad.
            # So simple count comparison is sufficient if we maximize total count.
            # However, the directive implies a structural preference.
            # Let's stick to maximizing total count, but maybe penalize loss of small diffs in tie-breaking or acceptance probability.
            accept = True
        elif new_score == current_score:
            # If score is same, prefer the one with more small diffs
            if small_new > small_curr:
                accept = True
            elif small_new == small_curr:
                # Random walk
                accept = random.random() < 0.5
            else:
                # Lost small diffs, keep them
                accept = False
        else:
            # new_score < current_score
            # Only accept if we gain small diffs? Unlikely to be worth it.
            # Or use simulated annealing
            delta = current_score - new_score
            # Penalize loss of small diffs more
            small_delta = small_curr - small_new
            if small_delta > 0:
                # We lost small diffs
                # Effective delta is larger
                eff_delta = delta + 2 * small_delta 
            else:
                eff_delta = delta
            
            accept = random.random() < pow(2.71828, -eff_delta / T)
            
        if accept:
            S = new_S
            current_D = new_D
            current_score = new_score
            if current_score > best_score:
                best_S = S.copy()
                best_score = current_score
                best_D = current_D.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_S, best_score

def construct():
    start_time = time.time()
    total_time = 115
    
    # Start from a dense small set
    seed_S = list(range(1, 33))
    
    # Run multiple times with different seeds or randomizations
    best_S = seed_S
    best_score = len(get_differences(seed_S))
    
    # Run 1: Heuristic search from seed
    S1, score1 = run_search_heuristic(seed_S, total_time / 3)
    if score1 > best_score:
        best_S = S1
        best_score = score1
        
    # Run 2: Start from a different structure?
    # Maybe two blocks?
    seed_S2 = list(range(1, 17)) + list(range(100, 116))
    S2, score2 = run_search_heuristic(seed_S2, total_time / 3)
    if score2 > best_score:
        best_S = S2
        best_score = score2
        
    # Run 3: Random
    seed_S3 = random.sample(range(1, 513), 32)
    S3, score3 = run_search_heuristic(seed_S3, total_time / 3)
    if score3 > best_score:
        best_S = S3
        best_score = score3
        
    return best_S
