import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    # LCM of 1..22
    # 2^4 * 3^2 * 5 * 7 * 11 * 13 * 17 * 19
    lcm_val = 232792560
    
    # Precompute weights
    weights = {i: lcm_val // i for i in range(1, 23)}
    
    # Sort indices by weight descending (i.e., index ascending)
    # Larger weights are more likely to be independent
    indices_by_weight = sorted(range(1, 23), key=lambda x: -weights[x])
    
    def get_subset_sums(S):
        """Returns a set of all subset sums of weights for S."""
        if not S:
            return {0}
        sums = {0}
        for x in S:
            w = weights[x]
            # Add w to all existing sums
            new_sums = set()
            for s in sums:
                new_sums.add(s + w)
            sums.update(new_sums)
        return sums
        
    def can_add(S, sums, x):
        """Checks if x can be added to S without collision."""
        w = weights[x]
        # New sums will be {s + w for s in sums}
        # Collision if any (s + w) is in sums
        # Or if any (s1 + w) == (s2 + w) which implies s1 == s2, so no internal collision
        # We just need to check intersection of (sums + w) and sums
        # Optimization: iterate and check
        for s in sums:
            if (s + w) in sums:
                return False, None
        # If no collision, return new sums
        new_sums = set(sums)
        for s in sums:
            new_sums.add(s + w)
        return True, new_sums

    def is_dissociated(S):
        """Checks if all subset sums are distinct."""
        return len(get_subset_sums(S)) == (1 << len(S))
        
    def greedy_construct():
        S = []
        sums = {0}
        remaining = set(range(1, 23))
        
        # Try to add elements in order of decreasing weight (increasing index value is wrong, decreasing index is correct for larger weight)
        # Wait, weight w_i = L/i. So w_1 > w_2 > ...
        # So we iterate i from 1 to 22.
        
        # We can also try a more flexible greedy: at each step, pick the best candidate
        # But simple greedy by weight is a good baseline.
        
        for x in range(1, 23):
            if x not in remaining:
                continue
            ok, new_sums = can_add(S, sums, x)
            if ok:
                S.append(x)
                sums = new_sums
                remaining.remove(x)
                
        return S, sums

    # Initial greedy construction
    best_set, best_sums = greedy_construct()
    best_size = len(best_set)
    
    # Local search to try to improve
    # Strategy: If we can't add more, try swapping an element in S with one in remaining
    # to see if it allows adding another element or just to find a better configuration
    
    def local_search(S, sums, time_limit):
        S = list(S)
        sums = set(sums)
        remaining = set(range(1, 23)) - set(S)
        size = len(S)
        
        # Try to add elements
        # We already did greedy, so this might not add anything immediately
        # But after swaps, it might.
        
        while time.time() < time_limit:
            # 1. Try to add any remaining element
            added = False
            for x in list(remaining):
                ok, new_sums = can_add(S, sums, x)
                if ok:
                    S.append(x)
                    sums = new_sums
                    remaining.remove(x)
                    size += 1
                    added = True
                    break # Restart loop to check for more additions
            
            if added:
                continue
                
            # 2. If no addition possible, try a swap
            # Pick a random element in S and a random element in remaining
            # Check if the swap maintains dissociation
            # Then try to add again
            
            if not S or not remaining:
                break
                
            i = random.randint(0, len(S)-1)
            j = random.choice(list(remaining))
            
            old_x = S[i]
            new_x = j
            
            # Remove old_x from S and sums
            # This is expensive to do incrementally. 
            # For simplicity, rebuild sums for the candidate set.
            # Candidate set: S with S[i] replaced by new_x
            candidate_S = list(S)
            candidate_S[i] = new_x
            
            # Check if candidate_S is dissociated
            # We can do this by computing sums
            # To speed up, we can check if the swap resolves a conflict that prevents addition?
            # No, we just check if candidate_S is valid.
            
            # Optimization: 
            # If we swap, we lose the sums structure.
            # Given size <= 17, 2^17 = 131072. Computing sums is fast.
            
            candidate_sums = get_subset_sums(candidate_S)
            if len(candidate_sums) == (1 << len(candidate_S)):
                # Valid swap
                S = candidate_S
                sums = candidate_sums
                remaining = set(range(1, 23)) - set(S)
                # Loop back to try adding
                
        return S, sums

    # Run local search for a few seconds
    # We have ~50s. Greedy is fast.
    # Let's run local search until deadline - 5s
    search_deadline = start_time + 50
    
    current_set, current_sums = local_search(best_set, best_sums, search_deadline)
    
    if len(current_set) > len(best_set):
        best_set = current_set
        best_sums = current_sums
        
    # Final check and return
    # Ensure we return a valid set
    # If best_set is not valid (shouldn't happen), trim it
    while best_set and not is_dissociated(best_set):
        # Remove last element
        best_set.pop()
        
    return best_set

# To ensure we don't miss a better solution, we can run multiple random restarts of the greedy phase
# But the greedy by weight is deterministic. 
# We can randomize the order of "remaining" in the greedy step if we want.
# However, the local search should handle the randomness.

# Let's refine the local search to be more aggressive with swaps if stuck.
# Also, we should consider that the maximum size might be 17 or 18.
# The current score 1.7778 is 16. 
# If we find 17, score is 17/9 = 1.888.
# If we find 18, score is 18/9 = 2.0.

# The program above looks solid. I will wrap it in the required function.

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    lcm_val = 232792560
    weights = {i: lcm_val // i for i in range(1, 23)}
    
    def get_subset_sums(S):
        if not S:
            return {0}
        sums = {0}
        for x in S:
            w = weights[x]
            new_sums = set()
            for s in sums:
                new_sums.add(s + w)
            sums.update(new_sums)
        return sums
        
    def can_add(S, sums, x):
        w = weights[x]
        for s in sums:
            if (s + w) in sums:
                return False, None
        new_sums = set(sums)
        for s in sums:
            new_sums.add(s + w)
        return True, new_sums

    def is_dissociated(S):
        return len(get_subset_sums(S)) == (1 << len(S))
        
    def greedy_construct():
        S = []
        sums = {0}
        remaining = set(range(1, 23))
        
        for x in range(1, 23):
            if x not in remaining:
                continue
            ok, new_sums = can_add(S, sums, x)
            if ok:
                S.append(x)
                sums = new_sums
                remaining.remove(x)
                
        return S, sums

    best_set, best_sums = greedy_construct()
    
    def local_search(S, sums, time_limit):
        S = list(S)
        sums = set(sums)
        remaining = set(range(1, 23)) - set(S)
        
        while time.time() < time_limit:
            added = False
            # Try to add any remaining element
            # Shuffle remaining to avoid bias
            rem_list = list(remaining)
            random.shuffle(rem_list)
            for x in rem_list:
                ok, new_sums = can_add(S, sums, x)
                if ok:
                    S.append(x)
                    sums = new_sums
                    remaining.remove(x)
                    added = True
                    break
            
            if added:
                continue
                
            # Try a swap
            if not S or not remaining:
                break
                
            i = random.randint(0, len(S)-1)
            j = random.choice(list(remaining))
            
            old_x = S[i]
            new_x = j
            
            candidate_S = list(S)
            candidate_S[i] = new_x
            
            candidate_sums = get_subset_sums(candidate_S)
            if len(candidate_sums) == (1 << len(candidate_S)):
                S = candidate_S
                sums = candidate_sums
                remaining = set(range(1, 23)) - set(S)
                
        return S, sums

    search_deadline = start_time + 50
    current_set, current_sums = local_search(best_set, best_sums, search_deadline)
    
    if len(current_set) > len(best_set):
        best_set = current_set
        best_sums = current_sums
        
    # Final validation
    while best_set and not is_dissociated(best_set):
        best_set.pop()
        
    return best_set
