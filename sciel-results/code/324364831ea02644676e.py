import random
import itertools
import time

def _is_sidon(subset):
    """Check if a list of distinct ints is a Sidon set."""
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(A):
    """Find the maximum size of a Sidon subset of A."""
    # Since we suspect max is 3 or 4, we can check for 4 first.
    # If no 4, check for 3.
    # If no 3, check for 2.
    # For efficiency in the search, we just need to know if it's <= 3.
    pass

def _has_sidon_subset_of_size(A, k):
    """Check if A contains a Sidon subset of size k."""
    for subset in itertools.combinations(A, k):
        if _is_sidon(list(subset)):
            return True
    return False

def _find_sidon_4subset(A):
    """Find one Sidon subset of size 4 in A, or return None."""
    for subset in itertools.combinations(A, 4):
        if _is_sidon(list(subset)):
            return subset
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Seed set
    seed_set = list(range(12))
    best_set = seed_set
    best_sidon_4_count = float('inf')
    
    # We want to find a set with 0 Sidon 4-subsets.
    # If we can't, we want to minimize the count?
    # Actually, if count > 0, max size is at least 4.
    # The score is the max size.
    # If we find a set with 0 Sidon 4-subsets, max size is 3 (likely).
    # If we don't, we return the one with the fewest Sidon 4-subsets?
    # No, if it has ANY Sidon 4-subset, the max size is at least 4.
    # It could be 5, 6, etc.
    # But breaking 4s is the first step.
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Initialize random set
        A = random.sample(range(129), 12)
        
        # Local search
        for step in range(200):
            if time.time() > deadline:
                break
                
            subset = _find_sidon_4subset(A)
            
            if subset is None:
                # Found a set with no Sidon 4-subsets
                # Max Sidon size is at most 3
                # Verify it's not 2 (unlikely)
                # If it has a Sidon 3-subset, score is 3.
                # If not, score is 2.
                # We can just return it.
                return A
            
            # We found a Sidon 4-subset.
            # Pick one element from it to replace.
            # To increase diversity, pick randomly from the subset.
            elem_to_replace = random.choice(subset)
            
            # Find its index in A
            idx = A.index(elem_to_replace)
            
            # Generate a new random value not in A
            available = [x for x in range(129) if x not in A]
            if not available:
                break # Should not happen as 129 > 12
            new_val = random.choice(available)
            
            A[idx] = new_val
            # A is no longer sorted, but that's fine for combinations
            
    # If we didn't find a perfect set, return the seed or the best effort?
    # The seed has Sidon 4-subsets.
    # We should return the set with the fewest Sidon 4-subsets found?
    # Or just the seed?
    # Actually, the local search might have reduced the count.
    # Let's track the best set (lowest count of Sidon 4-subsets).
    # But counting is expensive.
    # However, if we just return `A` from the last restart, it might be better than seed.
    # Let's try to count Sidon 4-subsets for the last few candidates?
    
    # For simplicity and safety, if we fail to find 0, we return the seed.
    # But we can do better: return the `A` from the last local search run, 
    # as it has been "optimized" to break Sidon 4s, even if not all.
    # However, without knowing the count, it's a gamble.
    # Given the time constraint, finding 0 is the primary goal.
    # If not found, returning seed is safe (score 4).
    # Returning a random A might be worse (score 5+).
    # So returning seed is safer if we fail.
    
    return seed_set
