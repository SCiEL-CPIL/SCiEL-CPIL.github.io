import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # We want distinct integers in 1..256.
    # No two distinct subset sums may divide one another.
    # Equal sums from different subsets are invalid.
    # Maximize set size.
    
    # Strategy:
    # 1. Build a conflict graph where nodes are candidate integers.
    #    Two integers conflict if adding one to a set containing the other
    #    would create a divisibility conflict or duplicate sum.
    #    But conflicts are context-dependent, so we use a different approach.
    
    # Better strategy: Use a greedy approach with backtracking.
    # Start with a good seed set, then try to add more elements.
    # Use the fact that larger numbers are less likely to divide each other.
    
    # Key insight: If all elements are > 128, then any subset sum of 2+ elements
    # is > 256, so we only need to check single-element sums and the total sum.
    # Actually, subset sums can be large. Let's think more carefully.
    
    # For a set S, the subset sums are all sums of non-empty subsets.
    # If all elements are > n, then any sum of k elements is > k*n.
    # For divisibility: if a divides b, then b >= a.
    # If all elements are in (m, 2m], then:
    #   - Single elements: no two divide each other if we pick carefully
    #   - Sums of 2+ elements: > 2m
    #   - A single element x in (m, 2m] cannot divide a sum of 2+ elements if that sum < 2x
    #   - Actually, x can divide 2x if 2x is a subset sum.
    
    # Let's try a different approach: use numbers that are pairwise "independent"
    # in the sense that no subset sum divides another.
    
    # Greedy with validation:
    def is_valid_set(S):
        """Check if set S satisfies all conditions."""
        if len(S) == 0:
            return True
        if len(S) != len(set(S)):
            return False
        if any(x < 1 or x > 256 for x in S):
            return False
        
        # Compute all non-empty subset sums
        elements = list(S)
        n = len(elements)
        sums_set = set()
        
        # Use bitmask enumeration
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            if s in sums_set:
                return False  # Duplicate sum
            sums_set.add(s)
        
        # Check divisibility
        sums_list = sorted(sums_set)
        n_sums = len(sums_list)
        for i in range(n_sums):
            for j in range(i + 1, n_sums):
                if sums_list[j] % sums_list[i] == 0:
                    return False
        return True
    
    # Fast validity check for adding one element
    def can_add(S, candidate):
        """Check if candidate can be added to S without violating conditions."""
        if candidate in S:
            return False
        if candidate < 1 or candidate > 256:
            return False
        
        # Compute current subset sums
        elements = list(S)
        n = len(elements)
        current_sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            current_sums.add(s)
        
        # New sums: candidate itself, and candidate + each current sum
        new_sums = set()
        new_sums.add(candidate)
        for s in current_sums:
            new_sums.add(s + candidate)
        
        # Check for duplicates
        all_sums = current_sums | new_sums
        if len(all_sums) != len(current_sums) + len(new_sums):
            return False
        
        # Check divisibility: only need to check pairs involving new sums
        # against all sums (since current_sums was already valid)
        for a in new_sums:
            for b in all_sums:
                if a == b:
                    continue
                if a % b == 0 or b % a == 0:
                    return False
        return True
    
    # Greedy construction with multiple strategies
    def greedy_construct(order, deadline):
        S = set()
        for c in order:
            if time.time() > deadline:
                break
            if can_add(S, c):
                S.add(c)
        return S
    
    # Local search: try to improve by swapping elements
    def local_search(S, candidates, deadline):
        best_S = set(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing one element and adding another
            for remove_c in list(best_S):
                if time.time() > deadline:
                    break
                temp_S = best_S - {remove_c}
                # Try adding candidates not in temp_S
                for add_c in candidates:
                    if add_c in temp_S:
                        continue
                    if can_add(temp_S, add_c):
                        new_S = temp_S | {add_c}
                        if len(new_S) > len(best_S):
                            best_S = new_S
                            improved = True
                            break
            # Also try just adding (without removing)
            if not improved:
                for add_c in candidates:
                    if add_c in best_S:
                        continue
                    if can_add(best_S, add_c):
                        best_S.add(add_c)
                        improved = True
        return best_S
    
    # Candidate pool: focus on larger numbers
    # Numbers in 128..256 are promising
    # Also try some mid-range numbers
    
    # Strategy 1: Greedy from large to small
    candidates_large = list(range(256, 127, -1))
    S1 = greedy_construct(candidates_large, deadline)
    
    # Strategy 2: Greedy from small to large (might find smaller valid sets)
    candidates_small = list(range(3, 257))
    S2 = greedy_construct(candidates_small, deadline)
    
    # Strategy 3: Random shuffles
    best_S = S1 if len(S1) >= len(S2) else S2
    best_size = len(best_S)
    
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        shuffled = list(range(3, 257))
        random.shuffle(shuffled)
        S_trial = greedy_construct(shuffled, deadline)
        if len(S_trial) > best_size:
            best_S = S_trial
            best_size = len(S_trial)
    
    # Local search to improve
    if time.time() < deadline - 5:
        candidates_all = list(range(3, 257))
        improved_S = local_search(best_S, candidates_all, deadline - 2)
        if len(improved_S) > best_size:
            best_S = improved_S
            best_size = len(best_S)
    
    # Final verification
    if is_valid_set(best_S):
        return [int(x) for x in sorted(best_S)]
    
    # Fallback: return a small valid set
    # {3} is always valid
    return [3]
