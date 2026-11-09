import random
import time

N = 256

def construct():
    # We will try to find a small maximal Sidon set using randomized DFS.
    # We iterate on the target size k, starting from a lower bound.
    # For each k, we attempt to find a Sidon set of size k that is maximal.
    
    start_time = time.time()
    time_limit = 25.0  # Leave 5s buffer for safety
    
    # Lower bound for maximal Sidon set size in [1, 256]
    # It's known that max Sidon size is ~sqrt(N).
    # Min maximal size is smaller. Let's start from 10.
    # If we can't find one, we go up.
    
    for k in range(10, 17):
        # Try to find a maximal Sidon set of size k
        # We use randomized DFS to explore the space of Sidon sets of size k
        
        attempts = 0
        max_attempts = 5000  # Limit attempts per k to avoid getting stuck
        
        while attempts < max_attempts:
            if time.time() - start_time > time_limit:
                break
            
            # Randomly shuffle the candidate list to get different DFS paths
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            
            found_set = dfs_find_sidon(candidates, k, start_time, time_limit)
            
            if found_set is not None:
                # Check if it is maximal
                if is_maximal(found_set):
                    return found_set
            attempts += 1
            
        # If we finished attempts for this k without success, try larger k
        # Or if time is up, break
        if time.time() - start_time > time_limit:
            break

    # Fallback: Use the original greedy method if DFS fails or times out
    # This ensures we always return a valid result
    return greedy_fallback()

def dfs_find_sidon(candidates, target_size, start_time, time_limit):
    """
    Randomized DFS to find a Sidon set of size target_size.
    Returns the set if found, None otherwise.
    """
    selected = []
    sums = set()
    
    def dfs(idx):
        if time.time() - start_time > time_limit:
            return False
        if len(selected) == target_size:
            return True
        
        # Pruning: not enough candidates left
        if idx + (target_size - len(selected)) > len(candidates):
            return False
            
        for i in range(idx, len(candidates)):
            x = candidates[i]
            
            # Check Sidon property for x
            valid = True
            new_sums = []
            
            # Check against existing sums
            for s in selected:
                val = x + s
                if val in sums or val in new_sums:
                    valid = False
                    break
                new_sums.append(val)
                
            if valid:
                val = 2 * x
                if val in sums or val in new_sums:
                    valid = False
                else:
                    new_sums.append(val)
            
            if valid:
                # Add x
                selected.append(x)
                sums.update(new_sums)
                
                if dfs(i + 1):
                    return True
                
                # Backtrack
                selected.pop()
                sums.difference_update(new_sums)
                
        return False
        
    return selected if dfs(0) else None

def is_maximal(sidon_set):
    """
    Check if the Sidon set is inclusion-maximal.
    """
    sums = set()
    for a in sidon_set:
        for b in sidon_set:
            if a <= b:
                sums.add(a + b)
    
    for x in range(1, N + 1):
        if x in sidon_set:
            continue
        # Check if x can be added
        valid = True
        new_sums = []
        for s in sidon_set:
            if s <= x:
                val = x + s
                if val in sums or val in new_sums:
                    valid = False
                    break
                new_sums.append(val)
            else:
                # If s > x, the pair (x, s) has sum x+s.
                # We need to check if x+s is already in sums.
                # Note: sums contains a+b for a<=b.
                # If s > x, then x+s is a new sum to check against existing sums.
                val = x + s
                if val in sums:
                    valid = False
                    break
                # We don't need to check new_sums for s > x because 
                # new_sums only contains sums from pairs (x, s') where s' <= x.
                # Wait, the check "val in new_sums" is for internal collisions among new sums.
                # If s > x, x+s is distinct from x+s' (s'<=x) because s != s'.
                # So we just check against old sums.
                pass
        
        if valid:
            val = 2 * x
            if val in sums or val in new_sums:
                valid = False
        
        if valid:
            return False
            
    return True

def greedy_fallback():
    """
    Original greedy method as a fallback.
    """
    best_set = None
    best_size = N + 1
    
    for _ in range(200):
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        selected = []
        sums = set()
        
        for x in candidates:
            new_sums = []
            valid = True
            for s in selected:
                val = x + s
                if val in sums or val in new_sums:
                    valid = False
                    break
                new_sums.append(val)
            
            if valid:
                val = 2 * x
                if val in sums or val in new_sums:
                    valid = False
                else:
                    new_sums.append(val)
            
            if valid:
                selected.append(x)
                sums.update(new_sums)
                
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            if best_size <= 10:
                break

    return best_set
