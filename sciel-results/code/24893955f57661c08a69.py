import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    n = 256
    
    def is_valid_set(S):
        """Check if set S satisfies: for all a<b<=c in S, a does not divide b+c"""
        S_list = sorted(S)
        n_list = len(S_list)
        for i in range(n_list):
            a = S_list[i]
            # For each a, check all pairs (b,c) with b>c>a
            for j in range(i + 1, n_list):
                b = S_list[j]
                for k in range(j, n_list):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    # Strategy: The upper tail [L, 256] is a natural starting point.
    # Let's find the best L such that [L, 256] is valid.
    # Then try to add more elements from below L.
    
    # First, find the maximum valid upper tail
    best_L = 1
    for L in range(1, n + 1):
        if time.time() > deadline:
            break
        S = set(range(L, n + 1))
        if is_valid_set(S):
            best_L = L
            break
    
    # Start with the largest valid upper tail
    current_set = set(range(best_L, n + 1))
    
    # Try to add elements from 1 to best_L-1
    # For each candidate, check if adding it keeps the set valid
    candidates = list(range(1, best_L))
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline:
            break
        if x not in current_set:
            # Check if adding x is valid
            test_set = current_set | {x}
            if is_valid_set(test_set):
                current_set.add(x)
    
    # Now try some local search / hill climbing
    # Try to swap elements: remove one, add another
    # Or try to add more elements
    
    # Let's also try a different approach: start from a larger lower bound
    # and see if we can get a bigger set
    
    best_size = len(current_set)
    best_result = set(current_set)
    
    # Try multiple random starts with different lower bounds
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        # Random lower bound
        L = random.randint(1, n)
        S = set(range(L, n + 1))
        
        if not is_valid_set(S):
            continue
        
        # Try to add elements from below
        add_candidates = list(range(1, L))
        random.shuffle(add_candidates)
        
        for x in add_candidates:
            if time.time() > deadline:
                break
            if x not in S:
                test_set = S | {x}
                if is_valid_set(test_set):
                    S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_result = set(S)
    
    return list(best_result)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
