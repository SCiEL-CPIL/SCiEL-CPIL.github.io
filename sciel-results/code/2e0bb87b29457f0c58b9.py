import time
import random

N = 512

# Precompute all 4-APs in [1, 512]
all_4aps = []
for d in range(1, N // 3 + 1):
    for a in range(1, N - 3 * d + 1):
        all_4aps.append((a, a + d, a + 2 * d, a + 3 * d))

# Map each element to the 4-APs it belongs to
element_to_aps = {i: [] for i in range(1, N + 1)}
for idx, ap in enumerate(all_4aps):
    for x in ap:
        element_to_aps[x].append(idx)

def has_4ap_with(S, x):
    """Check if adding x to S creates a 4-AP. S is a set."""
    for idx in element_to_aps[x]:
        ap = all_4aps[idx]
        count = 0
        for y in ap:
            if y == x:
                continue
            if y in S:
                count += 1
            if count == 3:
                return True
    return False

def greedy_construct():
    """Standard greedy: add if valid."""
    S = set()
    for x in range(1, N + 1):
        if not has_4ap_with(S, x):
            S.add(x)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin for safety
    
    # 1. Get a baseline
    best_set = greedy_construct()
    best_size = len(best_set)
    
    # 2. Local Search
    current_set = set(best_set)
    
    # Random seed for reproducibility in debugging, but random is fine for search
    random.seed(42) 
    
    while time.time() < deadline:
        # Try to add elements greedily/randomly
        # We'll do a batch of random additions
        for _ in range(20):
            if time.time() >= deadline:
                break
            y = random.randint(1, N)
            if y not in current_set:
                if not has_4ap_with(current_set, y):
                    current_set.add(y)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
        # If we are stuck (size didn't increase in last batch), do a swap
        # To detect "stuck", we can just always do a swap after a batch
        # or if size == best_size and no addition happened.
        # Simpler: Just do a swap every iteration.
        
        # Pick a random y not in current_set
        y = random.randint(1, N)
        if y in current_set:
            continue
            
        if has_4ap_with(current_set, y):
            # Find a conflicting 4-AP
            for idx in element_to_aps[y]:
                ap = all_4aps[idx]
                others = [z for z in ap if z != y]
                if all(z in current_set for z in others):
                    z_to_remove = random.choice(others)
                    current_set.remove(z_to_remove)
                    current_set.add(y)
                    break
        else:
            current_set.add(y)

    return sorted(best_set)
