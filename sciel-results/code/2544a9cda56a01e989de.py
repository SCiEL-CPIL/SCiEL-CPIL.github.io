import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin from 90s limit
    
    MAX_SUM = 48 * 512  # 24576
    # Mask to limit bitset size. We only care about sums up to MAX_SUM.
    # Using a mask ensures bitsets don't grow indefinitely, which speeds up operations.
    MASK = (1 << (MAX_SUM + 1)) - 1
    
    def is_valid_set(S_list):
        """
        Check if S_list is a valid set.
        Returns True if no element is the mean of any subset of at least 2 others.
        """
        n = len(S_list)
        if n == 0:
            return True
        
        # For each element a, we need to check if a is the mean of a subset T of S\{a}
        # Mean condition: sum(T) / len(T) == a  => sum(T) == a * len(T)
        # Let k = len(T). We need to check if k*a is a subset sum of S\{a} of size k.
        # k ranges from 2 to n-1.
        
        for i, a in enumerate(S_list):
            # Build DP for subset sums of S\{a}
            # dp[k] is a bitset where bit s is set if sum s is achievable with k elements
            dp = [0] * n
            dp[0] = 1  # 0 elements, sum 0
            
            # Iterate through all elements except a
            for j, x in enumerate(S_list):
                if j == i:
                    continue
                # Update dp in reverse order to avoid using the same element multiple times
                # dp[k] |= (dp[k-1] << x)
                for k in range(n - 1, 0, -1):
                    if dp[k-1]:
                        # Apply mask to keep integers small
                        dp[k] |= (dp[k-1] << x) & MASK
            
            # Check for violations
            for k in range(2, n):
                target = k * a
                if target > MAX_SUM:
                    break
                if (dp[k] >> target) & 1:
                    return False
        
        return True

    def try_add(S_list, x):
        """Check if x can be added to S_list to form a valid set."""
        if x in S_list:
            return False
        new_S = S_list + [x]
        return is_valid_set(new_S)

    def local_search(S, deadline):
        """
        Improve set S using local search.
        S is a set of integers.
        """
        S = set(S)
        S_list = list(S)
        all_vals = list(range(1, 513))
        
        stuck_count = 0
        while time.time() < deadline:
            added = False
            
            # Try to add a random element
            candidates = [x for x in all_vals if x not in S]
            if not candidates:
                break
                
            random.shuffle(candidates)
            # Try a few candidates
            for x in candidates[:20]:
                if try_add(S_list, x):
                    S.add(x)
                    S_list.append(x)
                    added = True
                    stuck_count = 0
                    break
            
            if added:
                continue
            
            # If no addition, try swaps
            stuck_count += 1
            if stuck_count > 5:
                # Try a more aggressive swap strategy
                # Remove a random element, try to add a random one
                if S_list:
                    idx = random.randint(0, len(S_list) - 1)
                    remove_x = S_list[idx]
                    S.remove(remove_x)
                    S_list.pop(idx)
                    
                    # Now try to add something
                    candidates = [x for x in all_vals if x not in S]
                    random.shuffle(candidates)
                    swap_success = False
                    for x in candidates[:30]:
                        if try_add(S_list, x):
                            S.add(x)
                            S_list.append(x)
                            swap_success = True
                            break
                            
                    if not swap_success:
                        # Revert
                        S.add(remove_x)
                        S_list.append(remove_x)
                        
            # Small chance to do a "kick" - remove 2, add 2?
            # This is expensive, so we do it rarely.
            if random.random() < 0.1 and len(S_list) > 2:
                # Pick two random elements to remove
                idx1, idx2 = random.sample(range(len(S_list)), 2)
                rem1, rem2 = S_list[idx1], S_list[idx2]
                S.remove(rem1)
                S.remove(rem2)
                S_list.remove(rem1)
                S_list.remove(rem2)
                
                candidates = [x for x in all_vals if x not in S]
                random.shuffle(candidates)
                added_count = 0
                for x in candidates[:50]:
                    if try_add(S_list, x):
                        S.add(x)
                        S_list.append(x)
                        added_count += 1
                        if added_count == 2:
                            break
                            
                if added_count < 2:
                    # Revert
                    S.add(rem1)
                    S.add(rem2)
                    S_list.append(rem1)
                    S_list.append(rem2)

        return S

    best_size = 0
    best_set = []
    
    # Multiple restarts
    while time.time() < deadline:
        # Start with a small random valid set
        S = set()
        S_list = []
        
        # Greedy initial construction
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if try_add(S_list, x):
                S.add(x)
                S_list.append(x)
                if len(S) == 48:
                    break
        
        # Local search to improve
        S = local_search(S, deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            if best_size == 48:
                break

    return best_set

# EVOLVE-BLOCK-END
