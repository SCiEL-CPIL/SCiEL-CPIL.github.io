"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

# EVOLVE-BLOCK-START
def _largest_sidon_subset_size(ambient):
    """
    Exhaustively compute the largest subset of 'ambient' such that all unordered
    pair sums (including diagonals, i.e., a_i + a_j for i <= j) are unique.
    """
    n = len(ambient)
    if n == 0:
        return 0
    
    # Precompute all pair sums for the full set? No, we need subset sums.
    # We will iterate by size k from n down to 1.
    # For each combination of size k, check if it is a Sidon set.
    
    # Optimization: Precompute all possible pair sums for the full ambient set
    # is not directly useful for subsets, but we can precompute indices.
    
    # A set S is Sidon if all sums a+b (a<=b) are distinct.
    # Number of sums is k*(k+1)/2.
    # Range of sums: min(ambient)*2 to max(ambient)*2.
    
    # To speed up, we can use a set to store seen sums.
    
    # We iterate k from n down to 1.
    # If we find a subset of size k, return k.
    
    # Note: The problem states "Evaluator exhaustively computes". 
    # With n=12, the worst case is checking C(12, 12) = 1, then C(12, 11) = 12, etc.
    # The most expensive check is for large k.
    # C(12, 6) = 924. C(12, 7) = 792.
    # Total combinations is 2^12 = 4096.
    # Checking each subset involves O(k^2) operations to compute sums and check uniqueness.
    # 4096 * 144 ~ 600k operations. This is very fast in Python (< 0.1s).
    
    # However, to be safe and fast, we can use a helper.
    
    def is_sidon(subset_indices):
        sums = set()
        for i in range(len(subset_indices)):
            for j in range(i, len(subset_indices)):
                s = ambient[subset_indices[i]] + ambient[subset_indices[j]]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Iterate by size descending
    for k in range(n, 0, -1):
        for combo in itertools.combinations(range(n), k):
            if is_sidon(combo):
                return k
    return 0

def _evaluate_candidate(ambient):
    """Returns the largest Sidon subset size for the given ambient set."""
    return _largest_sidon_subset_size(ambient)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # The goal is to MINIMIZE the largest Sidon subset size.
    # The trivial set [0..11] has size 12 (it is itself Sidon? 
    # 0+1=1, 0+2=2, 1+1=2 -> Collision! 0+2 == 1+1. So [0..11] is NOT Sidon.
    # Wait, let's check [0, 1, 2, 3].
    # Sums:
    # 0+0=0, 0+1=1, 0+2=2, 0+3=3
    # 1+1=2 (Collision with 0+2), 1+2=3 (Collision with 0+3), 1+3=4
    # 2+2=4 (Collision with 1+3), 2+3=5
    # 3+3=6
    # So [0,1,2,3] is not Sidon.
    # Largest Sidon subset of [0,1,2,3]?
    # Size 3: {0,1,3} -> sums: 0,1,3, 2,4, 6. Unique? 0+1=1, 1+1=2, 1+3=4, 0+3=3, 0+0=0, 3+3=6. All unique. Yes.
    # So max Sidon size for [0..3] is 3.
    
    # For [0..11], what is the max Sidon size?
    # It's likely around 5 or 6.
    
    # We want to find a set of 12 integers in [0, 128] that MINIMIZES this max size.
    # Ideally, we want a set where no subset of size 5 is Sidon, i.e., max size is 4.
    # Or even 3?
    # If max size is 3, it means every 4-element subset has a collision.
    
    # Strategy: Random search with local improvement.
    # 1. Generate random sets.
    # 2. Evaluate them.
    # 3. Keep the best one.
    # 4. Try to improve the best one by swapping elements.
    
    best_ambient = None
    best_score = 12  # Max possible is 12
    
    # Initial random guesses
    # We should bias towards "dense" sets? No, we want to FORCE collisions.
    # Collisions are more likely if numbers are close together?
    # Actually, Sidon sets are sparse. So a "dense" set of integers is likely to have small Sidon subsets.
    # So [0, 1, ..., 11] is a good starting point.
    
    # Let's evaluate [0..11] first
    candidate = list(range(12))
    score = _evaluate_candidate(candidate)
    if score < best_score:
        best_score = score
        best_ambient = candidate[:]
    
    # Also try some other structured sets
    # Arithmetic progressions?
    # If we take an AP, a, a+d, ..., a+11d.
    # Sums are 2a + (i+j)d.
    # Collision if i+j = k+l for distinct pairs.
    # This is the same as the [0..11] case shifted and scaled.
    # So APs behave like [0..11].
    
    # What if we cluster?
    # e.g., {0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23}
    # This might force more collisions within clusters?
    # Or maybe it prevents cross-cluster collisions?
    # Actually, we want to MINIMIZE the Sidon size.
    # A Sidon set cannot have two elements from the same "collision-prone" structure if that structure forces a collision in any subset.
    
    # Let's do random search.
    # Sample 12 distinct integers from [0, 128].
    
    random.seed(42)
    
    iterations = 0
    while time.time() < deadline:
        # Generate a random candidate
        # Bias towards low numbers?
        # Or uniform?
        # Let's try uniform first.
        candidate = random.sample(range(129), 12)
        candidate.sort()
        
        score = _evaluate_candidate(candidate)
        if score < best_score:
            best_score = score
            best_ambient = candidate[:]
            # If we find a very good score (e.g., 4), we can try to polish it.
            # But let's just keep searching for now.
        
        iterations += 1
        # Limit iterations to avoid infinite loop if time check is slow?
        # The time check is inside the loop.
        
        # Occasionally, try a local search around the best found
        if iterations % 100 == 0 and best_ambient is not None:
            # Local search: swap one element
            for i in range(12):
                for val in range(129):
                    if val in best_ambient:
                        continue
                    new_ambient = best_ambient[:]
                    new_ambient[i] = val
                    new_ambient.sort()
                    # Check validity (distinct, in range)
                    if len(set(new_ambient)) == 12:
                        new_score = _evaluate_candidate(new_ambient)
                        if new_score < best_score:
                            best_score = new_score
                            best_ambient = new_ambient[:]
                            break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break

    return best_ambient

# EVOLVE-BLOCK-END
