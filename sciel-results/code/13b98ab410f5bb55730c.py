import random
import time
from itertools import combinations

def is_sidon(s):
    """
    Check if a list of integers s is a Sidon set.
    A Sidon set requires all pairwise sums s[i] + s[j] for i <= j to be distinct.
    """
    sums = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            val = s[i] + s[j]
            if val in sums:
                return False
            sums.add(val)
    return True

def count_sidon_subsets(s, k):
    """
    Count the number of subsets of size k in s that are Sidon sets.
    """
    count = 0
    for subset in combinations(s, k):
        if is_sidon(list(subset)):
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # We want to minimize the maximum Sidon subset size.
    # Let's try to find a set with max Sidon size <= 3 (i.e., no Sidon-4).
    # If that's too hard, try <= 4 (no Sidon-5), etc.
    
    # Target sizes to avoid: 4, 5, 6
    # If we find a set with no Sidon-4, max size is at most 3.
    # If we find a set with no Sidon-5, max size is at most 4.
    
    best_set = None
    best_max_size = 12 # Worst case
    
    # We will iterate through target max sizes
    # If we can achieve max size 3, that's ideal.
    # Let's try to find sets that violate Sidon-4, Sidon-5, etc.
    
    # Strategy:
    # 1. Try to find a set with 0 Sidon-4 subsets.
    # 2. If not found quickly, try to find a set with 0 Sidon-5 subsets (allowing Sidon-4s).
    #    But wait, if we allow Sidon-4s, the max size is at least 4.
    #    So we prioritize finding a set with no Sidon-4.
    
    # Let's run a local search to minimize the count of Sidon-k subsets.
    # We'll try k=4 first.
    
    def local_search(target_k, time_limit):
        """
        Attempt to find a set of 12 integers in [0, 128] with 0 Sidon subsets of size target_k.
        Returns the set if found, else None.
        """
        end_time = time.time() + time_limit
        best_count = float('inf')
        best_s = None
        
        # Use a smaller range to encourage collisions
        # Range [0, 40] is a good start
        R = 40 
        
        # Initial random set
        s = random.sample(range(R + 1), 12)
        s.sort()
        
        c = count_sidon_subsets(s, target_k)
        if c == 0:
            return s
        
        best_count = c
        best_s = s[:]
        
        while time.time() < end_time:
            # Local search step
            # Pick a random index to move
            idx = random.randint(0, 11)
            old_val = s[idx]
            
            # Try a few random moves
            improved = False
            for _ in range(5):
                new_val = random.randint(0, R)
                if new_val == old_val or new_val in s:
                    continue
                
                s[idx] = new_val
                s.sort()
                
                new_c = count_sidon_subsets(s, target_k)
                
                if new_c < c:
                    c = new_c
                    improved = True
                    if c == 0:
                        return s
                    if c < best_count:
                        best_count = c
                        best_s = s[:]
                    break
                else:
                    # Revert
                    s[idx] = old_val
                    s.sort()
            
            if not improved:
                # Revert if no improvement found in this batch
                # Note: if we broke out of the inner loop with improvement, 
                # s is already updated. If not, s is reverted in each failed try.
                # But if we never entered the loop (all new_val invalid), s is unchanged.
                # Let's ensure s is consistent.
                # Actually, the logic above reverts immediately if new_c >= c.
                # So if no improvement, s is back to old state.
                
                # Occasional random restart or larger jump
                if random.random() < 0.05:
                    # Jump to a new random point in the neighborhood
                    for _ in range(3):
                        idx2 = random.randint(0, 11)
                        s[idx2] = random.randint(0, R)
                    s = list(set(s))
                    # If we lost elements due to collision, fill them
                    while len(s) < 12:
                        new_v = random.randint(0, R)
                        if new_v not in s:
                            s.append(new_v)
                    s.sort()
                    c = count_sidon_subsets(s, target_k)
                    if c < best_count:
                        best_count = c
                        best_s = s[:]
                    if c == 0:
                        return s
                        
        return best_s if best_count == 0 else None

    # Try to find a set with no Sidon-4 (max size <= 3)
    # Give it 15 seconds
    result = local_search(4, 15.0)
    if result:
        # Verify max size is indeed <= 3
        # Check for Sidon-4 again to be sure
        if count_sidon_subsets(result, 4) == 0:
            return result
        # If it failed verification (shouldn't happen), fall through
    
    # Try to find a set with no Sidon-5 (max size <= 4)
    # Give it 15 seconds
    result = local_search(5, 15.0)
    if result:
        if count_sidon_subsets(result, 5) == 0:
            # It might have Sidon-4s, but max is 4.
            # This is better than default 1.0 if default has max 5 or more.
            # We return this.
            return result

    # Try to find a set with no Sidon-6 (max size <= 5)
    # Give it remaining time
    remaining = deadline - time.time()
    if remaining > 2:
        result = local_search(6, remaining - 2)
        if result:
            if count_sidon_subsets(result, 6) == 0:
                return result

    # Fallback: Return a set with high collision density
    # Just use the best found in any of the above, or a random one
    # The local_search returns best_s even if count > 0 if we modify it to return best.
    # But my local_search returns None if best_count != 0.
    # Let's modify local_search to always return the best set found.
    
    # Re-running with a modified local_search that returns best set
    def local_search_best(target_k, time_limit):
        end_time = time.time() + time_limit
        best_count = float('inf')
        best_s = None
        R = 40
        
        s = random.sample(range(R + 1), 12)
        s.sort()
        c = count_sidon_subsets(s, target_k)
        best_count = c
        best_s = s[:]
        
        while time.time() < end_time:
            idx = random.randint(0, 11)
            old_val = s[idx]
            
            improved = False
            for _ in range(5):
                new_val = random.randint(0, R)
                if new_val == old_val or new_val in s:
                    continue
                
                s[idx] = new_val
                s.sort()
                
                new_c = count_sidon_subsets(s, target_k)
                
                if new_c < c:
                    c = new_c
                    improved = True
                    if c == 0:
                        return s
                    if c < best_count:
                        best_count = c
                        best_s = s[:]
                    break
                else:
                    s[idx] = old_val
                    s.sort()
            
            if not improved:
                if random.random() < 0.05:
                    for _ in range(3):
                        idx2 = random.randint(0, 11)
                        s[idx2] = random.randint(0, R)
                    s = list(set(s))
                    while len(s) < 12:
                        new_v = random.randint(0, R)
                        if new_v not in s:
                            s.append(new_v)
                    s.sort()
                    c = count_sidon_subsets(s, target_k)
                    if c < best_count:
                        best_count = c
                        best_s = s[:]
                    if c == 0:
                        return s
                        
        return best_s

    # If previous attempts didn't return a 0-count set, get the best partial result
    # We prioritize the one with the smallest target_k that had the lowest count?
    # Actually, a set with 1 Sidon-4 is worse than a set with 10 Sidon-4s?
    # No, we want to minimize MAX size.
    # A set with 1 Sidon-4 has max size >= 4.
    # A set with 0 Sidon-4 has max size <= 3.
    # So finding 0 is the goal.
    
    # If we couldn't find 0, we should return the set that minimizes the actual max Sidon size.
    # We can estimate this by checking for Sidon-4, then Sidon-5, etc.
    
    # Let's just return the best set found in the last search, or a random one if time is up.
    # The previous `local_search` calls returned None if not 0.
    # Let's do one final best-effort search for min Sidon-4 count, and if not 0, return it.
    # It's better to have a set with few Sidon-4s than a random one.
    
    remaining = deadline - time.time()
    if remaining > 1:
        best_s = local_search_best(4, remaining)
        if best_s:
            return best_s
            
    return list(range(12))
