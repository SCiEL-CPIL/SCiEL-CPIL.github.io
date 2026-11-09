import random
import time
import itertools

def is_sidon(subset):
    """
    Check if a subset is Sidon (all unordered pair sums including diagonals are unique).
    """
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def count_non_sidon_4_subsets(s):
    """
    Count the number of 4-element subsets of s that are NOT Sidon.
    We want to maximize this count. If it reaches C(12,4)=495, 
    then no 4-element subset is Sidon, so max Sidon size <= 3.
    """
    count = 0
    for combo in itertools.combinations(s, 4):
        if not is_sidon(combo):
            count += 1
    return count

def largest_sidon_subset_size(s):
    """
    Exhaustively compute the largest Sidon subset size.
    """
    n = len(s)
    # Check sizes from n down to 1
    # For n=12, checking 12 down to 4 is expensive if done naively, 
    # but we only need to do this for the final candidate or occasionally.
    # In the search loop, we use count_non_sidon_4_subsets as a proxy.
    # If count_non_sidon_4_subsets == 495, max size is <= 3.
    # We can then check if any 3-subset is Sidon.
    
    # Check size 4
    # If there is any Sidon 4-subset, return 4 (or higher, but unlikely >4 for dense sets)
    # Actually, we should check 5, 6 etc just in case, but for 12 elements in [0,128],
    # it's rare to have large Sidon subsets unless the set is sparse.
    # Our search aims for dense collision sets, so 4 is the upper bound we care about.
    
    # Optimization: If we know count_non_sidon_4 is 495, we know max <= 3.
    # Then check 3.
    
    # For the evaluator, we just need to return the set. 
    # This function is for internal verification if needed.
    for k in range(n, 3, -1):
        found = False
        for combo in itertools.combinations(s, k):
            if is_sidon(combo):
                return k
        if not found:
            # If no k-subset is Sidon, try k-1
            continue
    # Check size 3
    for combo in itertools.combinations(s, 3):
        if is_sidon(combo):
            return 3
    return 2

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Initial candidates
    # 1. AP: range(12)
    # 2. Random
    random.seed(42)
    
    # Start with AP
    best_set = list(range(12))
    best_score = count_non_sidon_4_subsets(best_set)
    
    # Try a few random starts to see if they are better
    for _ in range(10):
        rand_set = random.sample(range(129), 12)
        score = count_non_sidon_4_subsets(rand_set)
        if score > best_score:
            best_score = score
            best_set = rand_set

    current_set = list(best_set)
    current_score = best_score
    
    # Local Search
    while time.time() < deadline:
        if best_score == 495: # C(12,4)
            break
            
        # Generate a move: swap one element
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        # Choose a new value. 
        # Heuristic: Values that are close to existing ones create more collisions?
        # Or just random.
        new_val = random.randint(0, 128)
        
        if new_val in current_set and new_val != old_val:
            continue
            
        new_set = current_set.copy()
        new_set[idx] = new_val
        
        new_score = count_non_sidon_4_subsets(new_set)
        
        if new_score > current_score:
            current_score = new_score
            current_set = new_set
            if current_score > best_score:
                best_score = current_score
                best_set = current_set.copy()
        elif random.random() < 0.05:
            # Accept worse moves with low probability to escape local optima
            current_score = new_score
            current_set = new_set
            
        # Random restart if stuck for a while?
        # Simple restart logic: if no improvement in last X steps, restart
        # For now, just keep going.
        
    return best_set
