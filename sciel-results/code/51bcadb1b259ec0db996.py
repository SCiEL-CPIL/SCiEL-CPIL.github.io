import time
import random

def construct():
    """
    Find the largest set of distinct integers in 1..256 such that
    subset sums of different cardinalities are disjoint.
    """
    start = time.time()
    DEADLINE = 85.0
    
    # Known valid solution of size 31
    best = list(range(226, 257))
    best_size = len(best)
    
    def count_collisions(arr):
        """
        Count the number of pairs (i, j) with i != j such that
        there exists a sum s achievable by a subset of size i and
        a subset of size j.
        Returns the total count of such collisions (sum over all i<j of |R_i intersect R_j|).
        """
        n = len(arr)
        max_sum = sum(arr)
        # DP[j] is a bitset (integer) where bit s is 1 if sum s is achievable with exactly j elements
        DP = [0] * (n + 1)
        DP[0] = 1  # sum 0 with 0 elements
        
        for x in arr:
            # Update in reverse to avoid using x multiple times
            for j in range(n, 0, -1):
                DP[j] |= (DP[j - 1] << x)
                
        collisions = 0
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                collisions += (DP[i] & DP[j]).bit_count()
                
        return collisions

    def is_valid(arr):
        return count_collisions(arr) == 0

    # Try to find a valid set of size 32 using local search
    # Start with a random set of 32 elements
    # Or start with the best known and try to swap to allow adding a 32nd?
    # No, just search for a set of size 32 directly.
    
    # Random restarts
    for restart in range(10000):
        if time.time() > DEADLINE:
            break
            
        # Generate a random set of 32 distinct integers from 1..256
        # To bias towards larger numbers (which are more likely to work due to separation),
        # we can sample from a distribution or just uniform.
        # Uniform is fine.
        candidate = random.sample(range(1, 257), 32)
        
        current_cost = count_collisions(candidate)
        
        if current_cost == 0:
            best = candidate
            best_size = 32
            break
            
        # Local search
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            # Try all possible swaps
            # Pick two indices i, j
            # Swap candidate[i] and candidate[j]
            # If cost decreases, accept
            # To speed up, we can try random swaps instead of all pairs
            # Number of pairs is 32*31/2 = 496.
            # Counting collisions takes O(n^2 * bitops).
            # Bitops on 8192-bit ints are fast.
            # Let's try a few random swaps per iteration.
            
            for _ in range(100):
                if time.time() > DEADLINE:
                    break
                i, j = random.sample(range(32), 2)
                if candidate[i] == candidate[j]:
                    continue
                candidate[i], candidate[j] = candidate[j], candidate[i]
                new_cost = count_collisions(candidate)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    if current_cost == 0:
                        break
                else:
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    
            if current_cost == 0:
                best = candidate
                best_size = 32
                break
                
            # Also try replacing one element with a new one
            # Pick index i, pick new value v from 1..256 not in candidate
            # If cost decreases, accept
            for _ in range(50):
                if time.time() > DEADLINE:
                    break
                i = random.randint(0, 31)
                old_val = candidate[i]
                # Pick a random new value
                new_val = random.randint(1, 256)
                if new_val == old_val or new_val in candidate:
                    continue
                candidate[i] = new_val
                new_cost = count_collisions(candidate)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    if current_cost == 0:
                        break
                else:
                    candidate[i] = old_val

    # If we found size 32, we are done.
    # If not, return best (size 31).
    
    # Ensure types are correct
    return [int(x) for x in best]
