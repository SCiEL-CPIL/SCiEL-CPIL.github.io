import time

def construct():
    N = 128
    S = set()
    
    def is_valid_with(S, x):
        # Check if adding x to S keeps it valid
        # Only need to check triples involving x
        # Triples: {x, a, b} for a,b in S
        # Sums: x+a, x+b, a+b
        # Must not have all 3 in S U {x}
        # a+b is in S? 
        # x+a in S U {x}?
        # x+b in S U {x}?
        S_new = S | {x}
        S_set = S_new
        # Iterate pairs a,b in S
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                # Check if x+a, x+b, a+b are in S_new
                if (x + a) in S_set and (x + b) in S_set and (a + b) in S_set:
                    return False
        return True

    # Greedy from top
    for i in range(N, 0, -1):
        if is_valid_with(S, i):
            S.add(i)
            
    # Local Search to improve
    start_time = time.time()
    limit = 80.0
    best_S = S
    
    # Try to swap elements to increase size? 
    # Greedy is maximal, so we can't just add.
    # We can try to replace k elements with k+1 elements.
    # Or just random restarts of the greedy?
    
    # Let's try random restarts
    best_size = len(S)
    best_result = list(S)
    
    for _ in range(50):
        if time.time() - start_time > limit:
            break
        S_trial = set()
        # Random order? Or top-down with random skips?
        # Let's try top-down with random probability of skipping to break symmetry
        for i in range(N, 0, -1):
            if is_valid_with(S_trial, i):
                # Add with high probability, skip with low to explore
                if random.random() < 0.9:
                    S_trial.add(i)
        if len(S_trial) > best_size:
            best_size = len(S_trial)
            best_result = list(S_trial)
            
    return best_result
