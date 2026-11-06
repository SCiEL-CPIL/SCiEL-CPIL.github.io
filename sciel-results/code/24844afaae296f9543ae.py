import time
import random
from itertools import combinations

def is_sidon_4(subset):
    """Check if a 4-element subset has all unordered pair sums unique (including diagonals)."""
    sums = set()
    for i in range(4):
        for j in range(i, 4):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def count_sidon_4(S):
    """Count the number of 4-element subsets of S that are Sidon."""
    count = 0
    for combo in combinations(S, 4):
        if is_sidon_4(combo):
            count += 1
    return count

def max_sidon_size(S):
    """Find the largest Sidon subset size of S."""
    S = sorted(S)
    n = len(S)
    # We know min is 3. Check for 4.
    if count_sidon_4(S) > 0:
        # Check for 5? 
        # If we are optimizing for 3, we just need to know if it's >3.
        # But for safety, let's implement a general check or assume 4 is the max we care about.
        # If count_sidon_4 is 0, max is 3.
        # If count_sidon_4 > 0, max is >= 4.
        # We can return 4 for simplicity if we only care about beating 4.
        # But let's be precise.
        for k in range(n, 3, -1):
            for combo in combinations(S, k):
                # Check Sidon for k
                sums = set()
                is_s = True
                for i in range(k):
                    for j in range(i, k):
                        s = combo[i] + combo[j]
                        if s in sums:
                            is_s = False
                            break
                        sums.add(s)
                    if not is_s:
                        break
                if is_s:
                    return k
        return 4
    return 3

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Initial candidate: AP [0, 1, ..., 11]
    best_set = list(range(12))
    best_count = count_sidon_4(best_set)
    
    # Local search to minimize count_sidon_4
    # If we reach 0, we are done.
    
    current_set = best_set[:]
    current_count = best_count
    
    for i in range(10000): # Max iterations
        if time.time() > deadline:
            break
        
        if current_count == 0:
            break
        
        # Find a Sidon 4-subset to break
        sidon_subset = None
        for combo in combinations(current_set, 4):
            if is_sidon_4(combo):
                sidon_subset = combo
                break
        
        if sidon_subset is None:
            current_count = 0
            break
            
        # Try to perturb one element of the subset
        improved = False
        for elem in sidon_subset:
            # Try moving elem to a random new value
            for _ in range(50):
                new_val = random.randint(0, 128)
                if new_val in current_set:
                    continue
                
                # Create new set
                new_set = [x for x in current_set if x != elem]
                new_set.append(new_val)
                
                new_count = count_sidon_4(new_set)
                
                if new_count < current_count:
                    current_set = new_set
                    current_count = new_count
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # If no improvement found, try a random restart or larger move
            # Randomly pick an element and change it
            idx = random.randint(0, 11)
            old_val = current_set[idx]
            new_val = random.randint(0, 128)
            if new_val not in current_set:
                current_set[idx] = new_val
                current_count = count_sidon_4(current_set)
        
        # Track best
        if current_count < best_count:
            best_count = current_count
            best_set = current_set[:]
            
    return best_set
