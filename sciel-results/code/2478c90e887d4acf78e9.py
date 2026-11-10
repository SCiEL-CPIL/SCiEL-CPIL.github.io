import random
import time

# Precompute conflict triples for each x in [1, 512]
# aps[x] is a list of tuples (a, b, c) such that {x, a, b, c} is a 4-AP
# We store the other 3 elements.
N = 512
aps = [[] for _ in range(N + 1)]

for x in range(1, N + 1):
    # Case 1: x is the 1st term: x, x+d, x+2d, x+3d
    # d >= 1, x + 3d <= N
    d = 1
    while x + 3 * d <= N:
        aps[x].append((x + d, x + 2 * d, x + 3 * d))
        d += 1
        
    # Case 2: x is the 4th term: x-3d, x-2d, x-d, x
    # d >= 1, x - 3d >= 1
    d = 1
    while x - 3 * d >= 1:
        aps[x].append((x - 3 * d, x - 2 * d, x - d))
        d += 1
        
    # Case 3: x is the 2nd term: x-d, x, x+d, x+2d
    # d >= 1, x - d >= 1, x + 2d <= N
    d = 1
    while x - d >= 1 and x + 2 * d <= N:
        aps[x].append((x - d, x + d, x + 2 * d))
        d += 1
        
    # Case 4: x is the 3rd term: x-2d, x-d, x, x+d
    # d >= 1, x - 2d >= 1, x + d <= N
    d = 1
    while x - 2 * d >= 1 and x + d <= N:
        aps[x].append((x - 2 * d, x - d, x + d))
        d += 1

def construct():
    best_size = 0
    best_set = set()
    
    # Determine number of runs based on time
    # Estimate time per run
    start_time = time.time()
    
    # Do a quick run to estimate
    order = list(range(1, N + 1))
    random.shuffle(order)
    
    S = set()
    for x in order:
        can_add = True
        for triple in aps[x]:
            if triple[0] in S and triple[1] in S and triple[2] in S:
                can_add = False
                break
        if can_add:
            S.add(x)
            
    end_time = time.time()
    time_per_run = end_time - start_time
    
    # We have 60s limit. Leave some buffer.
    # If time_per_run is very small, we can do more runs.
    # Let's target 50 seconds total.
    if time_per_run < 0.001:
        num_runs = 5000
    elif time_per_run < 0.01:
        num_runs = 1000
    else:
        num_runs = int(50 / time_per_run)
        
    # Ensure at least 1 run
    num_runs = max(1, num_runs)
    
    # The first run is already done
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
        
    for _ in range(num_runs - 1):
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S = set()
        for x in order:
            can_add = True
            for triple in aps[x]:
                if triple[0] in S and triple[1] in S and triple[2] in S:
                    can_add = False
                    break
            if can_add:
                S.add(x)
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    return sorted(best_set)
