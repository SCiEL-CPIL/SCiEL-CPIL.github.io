import random
import time
import itertools

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def count_sidon_4_subsets(s):
    count = 0
    for combo in itertools.combinations(s, 4):
        if is_sidon(combo):
            count += 1
    return count

def largest_sidon_subset_size(s):
    # Check size 6, 5, 4, 3
    # For n=12, max sidon is small.
    # We can check all subsets of size k downwards.
    # But it's expensive.
    # However, we only need to know if size 4 exists.
    # If size 4 exists, return 4 (or check 5? Unlikely for N=12 in [0,128] with dense sets).
    # Actually, if we find a set with no Sidon 4-subsets, the max size is 3.
    # So we just need to check if there are ANY Sidon 4-subsets.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Initial candidates
    # 1. AP
    cand_ap = list(range(12))
    # 2. Random
    random.seed(42)
    best_set = cand_ap
    best_score = 100 # Number of Sidon 4-subsets, lower is better
    
    # Quick check on AP
    ap_score = count_sidon_4_subsets(cand_ap)
    if ap_score < best_score:
        best_score = ap_score
    
    # Local Search
    current_set = set(random.sample(range(129), 12))
    current_score = count_sidon_4_subsets(list(current_set))
    
    if current_score < best_score:
        best_set = list(current_set)
        best_score = current_score
    
    while time.time() < deadline:
        if best_score == 0:
            break
            
        # Random restart or step
        # Try swapping one element
        new_set = list(current_set)
        idx = random.randint(0, 11)
        old_val = new_set[idx]
        
        # Pick a new value
        # Prefer values that create collisions?
        # Just random for now
        new_val = random.randint(0, 128)
        
        if new_val in new_set and new_val != old_val:
            continue
            
        new_set[idx] = new_val
        
        # Check if this swap improves the score (reduces Sidon 4-subsets)
        # We only need to check 4-subsets involving the changed element
        # Or just recount all? 12 choose 4 is 495. Very fast.
        new_score = count_sidon_4_subsets(new_set)
        
        if new_score < current_score:
            current_score = new_score
            current_set = set(new_set)
            if current_score < best_score:
                best_score = current_score
                best_set = list(current_set)
        else:
            # Revert? Or keep and random walk?
            # Simple hill climbing with random restarts is better.
            # Let's do random restarts if stuck
            pass
            
        # Occasionally restart
        if random.random() < 0.01:
            current_set = set(random.sample(range(129), 12))
            current_score = count_sidon_4_subsets(list(current_set))
            if current_score < best_score:
                best_score = current_score
                best_set = list(current_set)
                
    # Ensure we return a valid list of 12 distinct integers
    # If best_set is AP, return it.
    # If best_set is better, return it.
    return best_set
