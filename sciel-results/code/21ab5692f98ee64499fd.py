import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Precompute conflict matrix and conflict lists
    n = 256
    C = [[False] * (n + 1) for _ in range(n + 1)]
    conflicts = [set() for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if (i * j) % (i + j) == 0:
                C[i][j] = True
                C[j][i] = True
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # Helper: Check if x can be added to set S
    def can_add(x, S):
        # x can be added if no element in S conflicts with x
        # i.e., S ∩ conflicts[x] is empty
        return len(S & conflicts[x]) == 0
        
    # Greedy initialization: Start with all odd numbers, then try to add even numbers
    # that don't conflict with the current set
    S = set(range(1, n + 1, 2))
    
    # Try to add even numbers greedily
    for x in range(2, n + 1, 2):
        if can_add(x, S):
            S.add(x)
    
    best_S = set(S)
    best_size = len(best_S)
    
    # Simulated Annealing with improved move selection
    T = 1.0
    cooling = 0.999
    min_T = 0.001
    
    # Precompute list of numbers for random selection
    all_nums = list(range(1, n + 1))
    
    # Track stagnation for restarts
    stagnation_count = 0
    stagnation_threshold = 200
    restart_T = 0.3
    
    # Count how many times we've tried each type of move
    move_counts = {'add': 0, 'remove': 0, 'swap': 0}
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        # Generate a move with adaptive probabilities based on current set size
        r = random.random()
        
        # Try to Add (higher probability when set is small)
        add_prob = 0.6 if len(S) < 100 else 0.4
        if r < add_prob:
            # Prefer candidates that have few conflicts with S
            candidates = [x for x in all_nums if x not in S]
            if candidates:
                # Score candidates by how many conflicts they have with S
                scored = [(len(conflicts[x] & S), x) for x in candidates]
                scored.sort()
                # Pick from top 20% with some randomness
                top_k = max(1, len(scored) // 5)
                x = random.choice([item[1] for item in scored[:top_k]])
                if can_add(x, S):
                    S.add(x)
                    move_counts['add'] += 1
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
                else:
                    stagnation_count += 1
        # Try to Remove (lower probability)
        elif r < add_prob + 0.15:
            if S:
                u = random.choice(list(S))
                # Accept removal with probability exp(-1/T)
                if random.random() < math.exp(-1.0 / T):
                    S.remove(u)
                    move_counts['remove'] += 1
                    stagnation_count += 1
        # Try to Swap
        else:
            if S:
                u = random.choice(list(S))
                # Find candidates that can replace u
                candidates = [x for x in all_nums if x not in S]
                if candidates:
                    # Score candidates by how many conflicts they have with S\{u}
                    S_without_u = S - {u}
                    scored = [(len(conflicts[x] & S_without_u), x) for x in candidates]
                    scored.sort()
                    # Pick from top 30% with some randomness
                    top_k = max(1, len(scored) // 3)
                    x = random.choice([item[1] for item in scored[:top_k]])
                    # Check if x is compatible with S \ {u}
                    inter = S & conflicts[x]
                    if not inter or (inter == {u}):
                        # Size remains same. Accept to explore
                        if random.random() < 0.7:
                            S.remove(u)
                            S.add(x)
                            move_counts['swap'] += 1
                            stagnation_count += 1
        
        # Restart if stuck
        if stagnation_count > stagnation_threshold:
            T = restart_T
            stagnation_count = 0
            # Also try a different initialization strategy
            if random.random() < 0.3:
                # Try starting from a random valid subset
                S = set()
                # Randomly add elements that don't conflict
                for x in random.sample(all_nums, n):
                    if can_add(x, S):
                        S.add(x)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
        
        # Cool down
        T *= cooling
        
    return list(best_S)
