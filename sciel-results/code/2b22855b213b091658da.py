import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 4096
    max_sum = 3 * n
    # bytearray for O(1) lookup of used sums. Index is the sum.
    # We need to store 0 or 1.
    
    best_set = []
    best_len = 0
    
    # Initial guess: Power of 3s is a valid B_3[1] set
    # 1, 3, 9, 27, 81, 243, 729, 2187
    # Next is 6561 which is > 4096.
    # So size 8 is a safe fallback.
    power3_set = [3**i for i in range(8) if 3**i <= n]
    best_set = list(power3_set)
    best_len = len(power3_set)
    
    def is_valid_set(S):
        """Check if set S has all unique unordered triple sums."""
        if not S:
            return True
        sums_used = bytearray(max_sum + 1)
        S = list(S)
        k = len(S)
        for i in range(k):
            si = S[i]
            for j in range(i, k):
                sj = S[j]
                for l in range(j, k):
                    sl = S[l]
                    s = si + sj + sl
                    if sums_used[s]:
                        return False
                    sums_used[s] = 1
        return True

    def get_new_sums(x, S):
        """
        Generate all new sums involving x when adding to S.
        Returns a list of sums.
        """
        new_sums = []
        # Triple (x, x, x)
        new_sums.append(3 * x)
        # Triples (x, x, y) where y in S
        for y in S:
            new_sums.append(2 * x + y)
        # Triples (x, y, z) where y, z in S, y <= z
        for i in range(len(S)):
            yi = S[i]
            for j in range(i, len(S)):
                zj = S[j]
                new_sums.append(x + yi + zj)
        return new_sums

    def can_add(x, S, sums_used):
        """
        Check if x can be added to S given the current sums_used bytearray.
        Returns True if valid, and the list of new sums if valid.
        """
        new_sums = get_new_sums(x, S)
        
        # Check for collisions with existing sums
        for s in new_sums:
            if sums_used[s]:
                return False, None
        
        # Check for internal collisions within new_sums
        # We need to ensure all elements in new_sums are distinct.
        # Since we are adding x to a valid set S, collisions can only happen
        # if two different triples involving x sum to the same value.
        # E.g., 2x + y1 == 2x + y2 implies y1==y2 (impossible as S is a set).
        # E.g., 2x + y1 == x + y2 + y3 implies x + y1 == y2 + y3.
        # This is a collision between a "type 2" sum and a "type 3" sum.
        # We must check if any sum in new_sums appears more than once.
        
        # Optimization: Instead of a full set check, we can just check 
        # if len(set(new_sums)) == len(new_sums).
        # But creating a set is slow. 
        # However, the number of new sums is small: 1 + |S| + |S|(|S|+1)/2.
        # For |S|=15, this is 1 + 15 + 120 = 136. A set check is fine.
        
        if len(set(new_sums)) != len(new_sums):
            return False, None
            
        return True, new_sums

    def greedy_build(ordering, sums_used):
        """
        Build a set using the given ordering of candidates.
        """
        S = []
        for x in ordering:
            valid, new_sums = can_add(x, S, sums_used)
            if valid:
                S.append(x)
                for s in new_sums:
                    sums_used[s] = 1
        return S

    # Strategy 1: Randomized Greedy with multiple seeds
    # We want to explore different parts of the search space.
    
    # Pre-compute a "smallest first" ordering for baseline
    smallest_order = list(range(1, n + 1))
    
    # Run a few randomized restarts
    # Time budget per restart
    num_restarts = 50
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for seed in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        
        # Create a randomized ordering.
        # Pure random can be bad because it might pick large numbers early,
        # wasting the low sum space.
        # A better strategy: "Biased Random" or "Shuffled with Constraints".
        # Let's try shuffling the range, but maybe bias towards smaller numbers
        # initially? Or just standard shuffle.
        
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        sums_used = bytearray(max_sum + 1)
        S = greedy_build(candidates, sums_used)
        
        if len(S) > best_len:
            best_set = S
            best_len = len(S)
            
            # If we found a large set, we might want to try to improve it further
            # via local search, but let's keep the loop simple for now.
            # If we find 15+, that's a good jump.
            
            # Early termination if we hit a known high bound?
            # Theoretical max for B_3[1] in [1, N] is ~ c * N^(1/3).
            # For N=4096, N^(1/3)=16.
            # If we find 16, we are likely near optimal.
            if best_len >= 16:
                # We can try to push for 17, but it might be impossible.
                # Let's continue searching a bit more to be sure.
                pass

    # Strategy 2: Local Search / Hill Climbing on the best set found
    # Try to swap elements to see if we can add another one.
    
    if best_len < 16 and time.time() < deadline - 5:
        # Work on a copy of the best set
        current_S = list(best_set)
        
        # Reconstruct sums_used for current_S
        sums_used = bytearray(max_sum + 1)
        for i in range(len(current_S)):
            si = current_S[i]
            for j in range(i, len(current_S)):
                sj = current_S[j]
                for k in range(j, len(current_S)):
                    sk = current_S[k]
                    sums_used[si + sj + sk] = 1
        
        # Attempt swaps
        # For each element in S, try replacing it with a non-member
        # to see if it allows adding a new element.
        
        # This is expensive. Let's do a simpler "add if possible" scan first.
        # Check if we can add any element to the current best set.
        for x in range(1, n + 1):
            if x in current_S:
                continue
            valid, new_sums = can_add(x, current_S, sums_used)
            if valid:
                current_S.append(x)
                for s in new_sums:
                    sums_used[s] = 1
                if len(current_S) > best_len:
                    best_set = current_S
                    best_len = len(current_S)
        
        # If we didn't improve, try 1-swap moves.
        # Remove one element, see if we can add two? Or just add one?
        # If we remove one, we free up sums. Then we try to add one.
        # If the new set is larger, we keep it.
        
        # Let's try to find a set of size best_len + 1 by searching for swaps.
        # This is a heuristic.
        
        # A more effective local search:
        # Try to find a set of size best_len + 1 by randomizing the last few steps.
        # Or, simply run more randomized greedy restarts with a bias towards
        # numbers that "fit well" with the current best set.
        
        # Let's just run more restarts with different random seeds if time permits.
        pass

    # Final validation
    if best_set and not is_valid_set(best_set):
        # Fallback to power of 3
        best_set = [3**i for i in range(8) if 3**i <= n]

    return best_set
