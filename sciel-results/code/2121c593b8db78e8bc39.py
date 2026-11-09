import time
import random

def is_valid_addition(S, x):
    """Check if adding x to S keeps it 3-AP free."""
    S_list = list(S)
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(len(S_list)):
            if i == j: continue
            b = S_list[j]
            # Check if a, b, x form a 3-AP
            # x can be 3rd: 2b - a == x
            # x can be 2nd: 2x - a == b => 2x == a + b
            # x can be 1st: 2a - b == x => 2a == b + x
            if 2*b - a == x or 2*a - b == x or 2*x == a + b:
                return False
    return True

def greedy_construct(randomize=False, k=1):
    S = set()
    candidates = list(range(1, 257))
    while candidates:
        valid = []
        for x in candidates:
            if is_valid_addition(S, x):
                valid.append(x)
        if not valid:
            break
        
        if randomize and len(valid) > k:
            # Pick randomly from the smallest k valid candidates
            valid_sorted = sorted(valid)[:k]
            x = random.choice(valid_sorted)
        else:
            # Pick the smallest valid candidate
            x = min(valid)
            
        S.add(x)
        candidates.remove(x)
        
    return S

def local_search_improve(S):
    """Try to swap elements for smaller ones to increase reciprocal sum."""
    S_list = list(S)
    improved = True
    while improved:
        improved = False
        # Sort S to try removing larger ones first? 
        # We want to replace u with v < u.
        for u in sorted(S_list, reverse=True):
            # Try to find a valid v < u not in S
            # Candidates v are 1..u-1 not in S
            for v in range(1, u):
                if v in S:
                    continue
                # Check if S - {u} + {v} is valid
                S_new = S - {u}
                if is_valid_addition(S_new, v):
                    S = S_new
                    S.add(v)
                    S_list = list(S)
                    improved = True
                    break # Restart inner loop for this u? Or break both?
            if improved:
                break
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 10s buffer
    
    best_S = None
    best_score = 0.0
    
    # Run multiple randomized greedy + local search
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Randomized greedy
        # Use different k values or just randomize
        S = greedy_construct(randomize=True, k=3)
        
        # Local search
        S = local_search_improve(S)
        
        # Calculate score
        score = sum(1.0/x for x in S)
        
        if score > best_score:
            best_score = score
            best_S = S
            
    # If no time for restarts or just to be safe, ensure we have a valid set
    if best_S is None:
        best_S = greedy_construct(randomize=False)
        
    return list(best_S)
