import random
import itertools
import time

def is_sidon(S):
    """Check if set S is a Sidon set (all unordered pair sums unique including diagonals)."""
    sums = set()
    for i, x in enumerate(S):
        for j, y in enumerate(S):
            if i <= j:
                s = x + y
                if s in sums:
                    return False
                sums.add(s)
    return True

def has_sidon_subset_of_size_k(A, k):
    """Check if A contains a Sidon subset of size k."""
    for combo in itertools.combinations(A, k):
        if is_sidon(combo):
            return True
    return False

def count_unique_sums(A):
    """Count the number of unique pair sums in A."""
    sums = set()
    for i, x in enumerate(A):
        for j, y in enumerate(A):
            if i <= j:
                sums.add(x + y)
    return len(sums)

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    best_set = None
    best_unique_sums = float('inf')
    best_sidon_size = float('inf')
    
    # We will run several local searches starting from different initial points
    # Initial points: Random, AP, etc.
    
    initial_sets = [
        list(range(12)),
        random.sample(range(129), 12),
        random.sample(range(129), 12),
        [0, 1, 2, 4, 8, 16, 32, 64, 96, 112, 120, 128] # Powers of 2 plus some
    ]
    
    for init_set in initial_sets:
        if time.time() > deadline - 10:
            break
            
        current_set = init_set[:]
        current_score = count_unique_sums(current_set)
        
        # Local search
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try swapping each element with a random other value
            for i in range(12):
                if time.time() > deadline - 10:
                    break
                current_val = current_set[i]
                # Try a few random replacements
                for _ in range(20):
                    new_val = random.randint(0, 128)
                    if new_val in current_set:
                        continue
                    new_set = current_set[:]
                    new_set[i] = new_val
                    new_score = count_unique_sums(new_set)
                    if new_score < current_score:
                        current_set = new_set
                        current_score = new_score
                        improved = True
                        break
        
        # Evaluate the best found so far
        if current_score < best_unique_sums:
            best_unique_sums = current_score
            best_set = current_set[:]
            
            # Check actual Sidon size
            # We only care if it's < 5
            # If it is 5, it's a tie with baseline.
            # If it is 4, it's an improvement.
            
            if not has_sidon_subset_of_size_k(best_set, 5):
                # Found a set with max Sidon size <= 4
                # This is an improvement over baseline 5
                return best_set
                
    # If we didn't find a set with max Sidon size < 5, return the best heuristic candidate
    # Or the baseline if we didn't find anything better
    if best_set is None:
        return list(range(12))
        
    return best_set
