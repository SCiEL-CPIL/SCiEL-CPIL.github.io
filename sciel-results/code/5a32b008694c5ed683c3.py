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

def count_sidon_subsets_of_size_k(S, k):
    count = 0
    for sub in combinations(S, k):
        if is_sidon(sub):
            count += 1
    return count

def max_sidon_size(S):
    # Check from largest possible down
    # Max size for 12 elements is 12, but practically <= 6 or 7
    # We care about minimizing, so we check 5 and 6 primarily.
    # If we find a set with 0 subsets of size 5, max is <= 4.
    
    # Check size 6
    if count_sidon_subsets_of_size_k(S, 6) > 0:
        return 6
    # Check size 5
    if count_sidon_subsets_of_size_k(S, 5) > 0:
        return 5
    # Check size 4
    if count_sidon_subsets_of_size_k(S, 4) > 0:
        return 4
    return 3

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # Initial candidate
    # range(12) has max sidon size 5
    current = list(range(12))
    current_score = max_sidon_size(current)
    current_count5 = count_sidon_subsets_of_size_k(current, 5)
    
    best = current
    best_score = current_score
    best_count5 = current_count5
    
    random.seed(42)
    
    # Local search
    # We want to minimize max_sidon_size, then minimize count of sidon subsets of that size
    # Priority: get to score 4.
    
    while time.time() < deadline:
        # Generate neighbor
        new_set = current[:]
        idx = random.randint(0, 11)
        old_val = new_set[idx]
        # Try a few new values
        found_better = False
        for _ in range(20):
            new_val = random.randint(0, 128)
            if new_val == old_val:
                continue
            if new_val in new_set:
                continue
            new_set[idx] = new_val
            
            # Quick check: is it valid? (distinct)
            # We ensured distinctness above
            
            # Evaluate
            # To speed up, we can just check if it has any Sidon subset of size 5 or 6
            # But we need the score.
            
            # Optimization: 
            # If current_score is 5, we hope to get 4.
            # If current_score is 4, we hope to get 3 (unlikely) or maintain 4 with fewer subsets.
            
            # Let's just compute the count of size 5 subsets.
            # If count5 == 0, score is 4 (assuming no size 6, which is also 0 if count5 is 0? No, size 6 implies size 5 subsets are not necessarily sidon? 
            # Wait. If a set has a Sidon subset of size 6, does it have a Sidon subset of size 5?
            # Yes, any subset of a Sidon set is Sidon.
            # So if count6 > 0, then count5 > 0.
            # So if count5 == 0, then count6 == 0.
            # So we only need to check count5.
            
            new_count5 = count_sidon_subsets_of_size_k(new_set, 5)
            
            # Determine new score
            if new_count5 == 0:
                new_score = 4
            else:
                # Check if size 6 exists? 
                # If count5 > 0, score is at least 5.
                # We need to distinguish 5 and 6.
                # But if we are searching for 4, we only care if count5 becomes 0.
                # If count5 > 0, we need to know if it's 5 or 6 to compare?
                # Minimizing score: 4 < 5 < 6.
                # So if new_score < current_score, accept.
                # If new_score == current_score, accept if new_count5 < current_count5 (for score 5) or similar.
                
                # To be precise:
                if count_sidon_subsets_of_size_k(new_set, 6) > 0:
                    new_score = 6
                else:
                    new_score = 5
            
            # Acceptance criteria
            if new_score < best_score:
                best = new_set[:]
                best_score = new_score
                best_count5 = new_count5
                current = new_set[:]
                current_score = new_score
                current_count5 = new_count5
                found_better = True
                break
            elif new_score == best_score:
                # Prefer fewer Sidon subsets of the max size?
                # If score is 5, fewer size-5 subsets is better (closer to 4).
                # If score is 4, fewer size-4 subsets?
                # Let's just prefer lower count5.
                if new_count5 < current_count5:
                    current = new_set[:]
                    current_score = new_score
                    current_count5 = new_count5
                    found_better = True
                    # Don't break, keep looking for better in this iteration?
                    # Or break to save time?
                    # Let's break to save time.
                    break
            
        if found_better:
            continue
            
        # If no improvement, maybe restart or perturb more strongly?
        # For now, just keep trying.
        
    return best

# EVOLVE-BLOCK-START
# The code above is the implementation.
# We need to wrap it in the construct function as required.
# The prompt asks to keep function names and signatures.
# The current program has `def construct():`.
# I will place the helper functions outside and `construct` inside the block?
# No, the EVOLVE-BLOCK is just the code that is evolved.
# The seed program has:
# def construct():
#     return list(range(12))
# I should replace the whole file or just the block?
# "Output the COMPLETE program in a single 