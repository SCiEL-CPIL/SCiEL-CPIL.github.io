import random
import time
from math import lcm
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # LCM of 1..22
    L = 232792560
    weights = [L // i for i in range(1, 23)] # indices 0..21 correspond to 1..22
    
    def is_dissociated(subset_indices):
        """
        Checks if the weights corresponding to subset_indices form a dissociated set.
        i.e., all subset sums are distinct.
        """
        k = len(subset_indices)
        if k == 0:
            return True
        w = [weights[i] for i in subset_indices]
        
        # Generate all subset sums
        # For k=15, 2^15 = 32768 sums.
        # We can use a set to store them.
        sums = set()
        # Iterative generation
        sums.add(0)
        for wi in w:
            new_sums = set()
            for s in sums:
                ns = s + wi
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
            if len(sums) != 2 ** (len(w) - len(w[:len(w)-1])): # This check is inefficient, just check at end or rely on early exit
                pass
        # If we reach here, all sums were distinct
        return len(sums) == 2 ** k

    # Strategy: Try to find a set of size 15, then 14.
    # We will use a local search / greedy extension.
    
    best_set = [0, 1, 2, 4, 6, 10, 12, 16, 18] # The seed set (indices for 1,2,3,5,7,11,13,17,19)
    
    # Try to extend the seed set or find a better random one.
    # Let's try random restarts with local improvement.
    
    def try_find_max_size(target_size):
        # Try to find a valid set of exactly target_size
        # We can start with a random subset of size target_size and check
        # Or build up from size 1.
        
        # Building up is safer.
        # Start with empty set, add random elements.
        # If collision, backtrack or swap.
        
        # Given time constraints, simple random sampling of large sets might be too slow if rare.
        # But dissociated sets are not that rare.
        
        # Let's try a greedy algorithm:
        # Start with empty set.
        # Iterate through elements in random order.
        # Try to add the element. If it creates a collision, skip it.
        # This might get stuck in a local minimum.
        
        # Better: Random Local Search on the set of size K.
        
        import random
        all_indices = list(range(22))
        
        for _ in range(1000): # max iterations for this target size
            if time.time() > deadline:
                return None
            
            # Start with a random subset of size target_size
            current = set(random.sample(all_indices, target_size))
            
            if is_dissociated(list(current)):
                return list(current)
                
            # If not, try to fix it by swapping elements
            # This is a heuristic.
            
        return None

    # Try sizes from 15 down to 1
    for size in [15, 14, 13, 12, 11, 10, 9]:
        if time.time() > deadline:
            break
        
        # Try random restarts
        found = None
        attempts = 0
        while time.time() < deadline - 5.0:
            attempts += 1
            if attempts > 5000:
                break
                
            # Random subset of this size
            candidate = random.sample(range(22), size)
            
            if is_dissociated(candidate):
                found = candidate
                break
                
        if found:
            # Check if this is better than current best
            if len(found) > len(best_set):
                best_set = found
            # If we found size 15, we can stop or try for 16?
            # The max possible is 22. But 15 is likely the max or close.
            # Let's try to extend if found size < 15?
            # Actually, if we found size 15, we should try to see if 16 is possible.
            # But the loop goes 15, 14... so if 15 is found, we return it immediately?
            # No, we want the MAXIMUM.
            # So if we find 15, we should try 16.
            
            # Let's restructure: Try to find max possible.
            # We'll just run a search for 16, 15, 14...
            
            # If we found a valid set of this size, we can return it if we are iterating downwards?
            # No, we want the largest.
            # So we should try 16 first, then 15, etc.
            
            # Let's restart the logic.
            pass

    # Restart logic properly
    best_found = None
    
    # Try sizes from 16 down to 9
    for size in range(16, 8, -1):
        if time.time() > deadline:
            break
            
        # Try to find a valid set of this size
        # Use random sampling
        # To improve chances, we can bias towards "sparse" sets?
        # Or just random.
        
        found = None
        max_attempts = 2000
        for _ in range(max_attempts):
            if time.time() > deadline - 2.0:
                break
            candidate = random.sample(range(22), size)
            if is_dissociated(candidate):
                found = candidate
                break
                
        if found:
            best_found = found
            # If we found size 16, we can't go higher easily, but we can try.
            # If we found 15, we should have tried 16 first.
            # Since we are iterating downwards, the first found is the largest.
            break
            
    if best_found:
        return [i + 1 for i in best_found] # Convert indices back to 1-based values
    else:
        # Fallback to seed
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]

# EVOLVE-BLOCK-END
