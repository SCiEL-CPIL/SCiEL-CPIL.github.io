import random
import time
from itertools import combinations

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

def max_sidon_size(S):
    n = len(S)
    # Check from largest to smallest
    for k in range(n, 1, -1):
        for comb in combinations(S, k):
            if is_sidon(comb):
                return k
    return 2 # If no 3+ found, max is 2 (any 2 distinct are Sidon)
    # Note: if n<2, return n. But n is always 12.

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin
    
    best_set = None
    best_score = 13 # Max possible is 12, so 13 is safe init
    
    # Initial guess: Arithmetic Progression
    current_set = list(range(12))
    current_score = max_sidon_size(current_set)
    
    if current_score < best_score:
        best_score = current_score
        best_set = list(current_set)
        
    if best_score == 2:
        return best_set

    # Hill Climbing with Random Restarts
    # We can do multiple restarts if time permits
    
    while time.time() < deadline:
        # Random restart
        # Generate a random set of 12 distinct ints in [0, 128]
        current_set = random.sample(range(129), 12)
        current_score = max_sidon_size(current_set)
        
        if current_score < best_score:
            best_score = current_score
            best_set = list(current_set)
            if best_score == 2:
                return best_set
                
        # Local search from this start
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all single-element changes?
            # 12 elements * 118 options = 1416 neighbors.
            # Evaluate each neighbor.
            # 1416 * cost_eval.
            # cost_eval is ~1ms. 1416ms per iteration. Too slow for many iterations?
            # Let's optimize neighbor generation or use random neighbors.
            
            # Let's use random neighbors to speed up
            for _ in range(50): # Try 50 random moves
                idx = random.randint(0, 11)
                old_val = current_set[idx]
                # Pick a new value not in the set
                while True:
                    new_val = random.randint(0, 128)
                    if new_val not in current_set:
                        break
                
                # Apply move
                current_set[idx] = new_val
                new_score = max_sidon_size(current_set)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_set = list(current_set)
                        if best_score == 2:
                            return best_set
                else:
                    # Revert
                    current_set[idx] = old_val
                    
    return best_set
