import random
import time
import itertools

def is_strong_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_subset_size(S):
    n = len(S)
    # Check subsets in descending order of size
    for k in range(n, 0, -1):
        for combo in itertools.combinations(S, k):
            if is_strong_sidon(list(combo)):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    best_set = None
    best_score = 13
    
    # Heuristic: Use a smaller range to increase collision probability
    # Try different ranges
    ranges_to_try = [
        list(range(0, 20)),
        list(range(0, 30)),
        list(range(0, 50)),
        list(range(0, 129))
    ]
    
    # Initial guesses: structured sets
    # Example: 0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14
    structured = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
    if len(set(structured)) == 12 and all(0 <= x <= 128 for x in structured):
        s = max_sidon_subset_size(structured)
        if s < best_score:
            best_score = s
            best_set = structured[:]
            
    current_set = random.sample(range(0, 30), 12)
    current_score = max_sidon_subset_size(current_set)
    if current_score < best_score:
        best_score = current_score
        best_set = current_set[:]
        
    iterations = 0
    while time.time() < deadline:
        # Local search
        # Sometimes switch range
        if iterations % 100 == 0:
            # Try a different range for restart
            r = random.choice(ranges_to_try)
            current_set = random.sample(r, 12)
            current_score = max_sidon_subset_size(current_set)
            if current_score < best_score:
                best_score = current_score
                best_set = current_set[:]
                
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        # Bias towards smaller numbers?
        # Or just random in 0-40
        new_val = random.randint(0, 40)
        
        if new_val in current_set:
            continue
            
        current_set[idx] = new_val
        new_score = max_sidon_subset_size(current_set)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current_set[:]
        else:
            # Revert
            current_set[idx] = old_val
            
        iterations += 1
        
    return best_set
