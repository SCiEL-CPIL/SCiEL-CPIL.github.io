import time
import random

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin for 90s limit
    
    N = 256
    nodes = list(range(1, N + 1))
    
    # Precompute neighbors
    neighbors = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                neighbors[i].append(j)
                neighbors[j].append(i)
                
    # Initial set: all odds
    S = set(range(1, N + 1, 2))
    best_set = set(S)
    best_size = len(S)
    
    def is_independent(candidate, current_set):
        for n in neighbors[candidate]:
            if n in current_set:
                return False
        return True
        
    # To speed up "find all valid candidates", we can maintain a list of candidates
    # but simple re-scanning is fast enough for N=256.
    
    while time.time() - start_time < timeout:
        # Greedy addition phase
        # Get non-members
        non_members = [i for i in nodes if i not in S]
        if not non_members:
            break # Maximal size reached (impossible unless N=256 and no conflicts)
            
        random.shuffle(non_members)
        
        # Try to add valid ones
        # We can break early if we add many? No, adding all possible makes it maximal.
        for c in non_members:
            if time.time() - start_time > timeout:
                break
            if is_independent(c, S):
                S.add(c)
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
        # Perturbation phase
        # If we are in a maximal independent set (which we are after the loop above, 
        # unless we broke due to timeout), we need to escape.
        # Even if we didn't add anything, we are in a maximal set.
        
        if len(S) > 0:
            # Randomly remove some elements to allow different combinations
            # The number of removals can be tuned. 1-5 seems reasonable.
            max_removals = min(5, len(S))
            num_removals = random.randint(1, max_removals)
            to_remove = random.sample(list(S), num_removals)
            for r in to_remove:
                S.discard(r)
                
    return list(best_set)
