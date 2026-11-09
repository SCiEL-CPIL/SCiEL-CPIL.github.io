import time
import random

def construct():
    start_time = time.time()
    # Evaluator kills at 90s. Keep safety margin.
    deadline = start_time + 85.0
    
    target = 200
    max_val = 100
    all_elements = list(range(1, max_val + 1))
    
    best_set = []
    best_size = 0
    
    def compute_reachable_mask(elements):
        """
        Computes the bitset of reachable sums for a set of elements.
        Returns an integer where bit i is 1 if sum i is reachable.
        """
        mask = 1  # bit 0 is set
        for x in elements:
            if x > target:
                continue
            # Shift and OR, but we only care about bits up to target
            # (mask << x) | mask
            # We need to mask off bits > target to keep the integer small and correct for "no sum > target" logic?
            # Actually, for the check "does sum 200 exist", we just need to know if bit 200 is set.
            # Sums > 200 don't help form 200. So we can truncate.
            new_mask = mask | (mask << x)
            # Truncate to target + 1 bits
            new_mask &= (1 << (target + 1)) - 1
            mask = new_mask
        return mask

    def has_conflict(elements):
        """
        Returns True if any subset of elements sums to target.
        """
        mask = compute_reachable_mask(elements)
        return (mask >> target) & 1

    def find_conflict_subset(elements):
        """
        Finds a specific subset of elements that sums to target.
        Returns list of elements or None.
        Uses DP with backtracking.
        """
        # reachable[s] = True if sum s is reachable
        # parent[s] = (prev_s, x) to backtrack
        reachable = [False] * (target + 1)
        parent = [None] * (target + 1)
        reachable[0] = True
        
        # We process elements in the given order.
        # The order affects which subset is found.
        for x in elements:
            if x > target:
                continue
            # Iterate backwards to ensure each element used at most once
            for s in range(target, x - 1, -1):
                if not reachable[s] and reachable[s - x]:
                    reachable[s] = True
                    parent[s] = (s - x, x)
        
        if not reachable[target]:
            return None
        
        # Backtrack
        subset = []
        s = target
        while s > 0:
            prev_s, x = parent[s]
            subset.append(x)
            s = prev_s
        return subset

    def try_update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]

    def strategy_addition_greedy(seed):
        """
        Start with empty set, add elements if valid.
        Try multiple random orderings.
        """
        local_best = []
        local_best_size = 0
        
        # Try a few random orderings
        for _ in range(5):
            random.seed(seed + _)
            candidates = all_elements[:]
            random.shuffle(candidates)
            
            current_set = []
            current_mask = 1
            
            for x in candidates:
                # Check if adding x creates a conflict
                # New mask would be current_mask | (current_mask << x)
                # We check if bit 'target' is set in the new mask
                new_mask = current_mask | (current_mask << x)
                new_mask &= (1 << (target + 1)) - 1
                if (new_mask >> target) & 1:
                    # Conflict, skip
                    continue
                else:
                    # Safe to add
                    current_set.append(x)
                    current_mask = new_mask
            
            if len(current_set) > local_best_size:
                local_best_size = len(current_set)
                local_best = current_set[:]
                
        return local_best

    def strategy_removal_advanced(seed):
        """
        Start with all elements, remove elements to resolve conflicts.
        Uses frequency heuristic for removal choice.
        """
        random.seed(seed)
        S = all_elements[:]
        
        # Phase 1: Removal
        # We want to remove elements that are in many conflicts.
        # We approximate this by sampling conflicts.
        
        while time.time() < deadline - 2.0:
            # Find one conflict to check if we are done
            # Use the current order of S for this check
            C = find_conflict_subset(S)
            if C is None:
                break
            
            # If we have a conflict, we need to remove an element from C.
            # To choose the best element to remove, sample more conflicts.
            # This helps identify "central" elements in the conflict structure.
            
            # Sample 10 conflicts by shuffling the input order for the DP
            # Note: The set S is fixed, but the DP path depends on order.
            conflicts_found = []
            for _ in range(10):
                shuffled_S = S[:]
                random.shuffle(shuffled_S)
                C_sample = find_conflict_subset(shuffled_S)
                if C_sample:
                    conflicts_found.append(C_sample)
            
            if not conflicts_found:
                # Should not happen if C was found, but safety
                break
                
            # Count frequency of each element in the sampled conflicts
            freq = {}
            for c in conflicts_found:
                for x in c:
                    freq[x] = freq.get(x, 0) + 1
            
            # Pick element from the primary conflict C with highest frequency
            # If tie, pick the largest element (heuristic: large elements are "scarce" in sum formation, 
            # so removing them might break more potential sums? Or is it the other way?
            # Actually, removing the LARGEST element in a conflict is often effective 
            # because it reduces the maximum possible sum contributions significantly.
            # Let's stick with: Highest Freq, then Largest Value.
            
            max_freq = max(freq.get(x, 0) for x in C)
            candidates = [x for x in C if freq.get(x, 0) == max_freq]
            x_to_remove = max(candidates)
            
            S.remove(x_to_remove)
        
        # Phase 2: Polish (Add back)
        # Compute mask of current valid set S
        mask = compute_reachable_mask(S)
        
        # Get removed elements
        removed = [x for x in all_elements if x not in S]
        
        # Try to add back removed elements
        # Randomize the order of adding back to explore different local optima
        random.shuffle(removed)
        
        for x in removed:
            if time.time() > deadline - 1.0:
                break
            # Check if adding x is valid
            new_mask = mask | (mask << x)
            new_mask &= (1 << (target + 1)) - 1
            if not ((new_mask >> target) & 1):
                S.append(x)
                mask = new_mask
        
        return S

    # Run strategies
    
    # 1. Greedy Addition (Fast, good baseline)
    for seed in range(20):
        if time.time() > deadline - 10.0:
            break
        res = strategy_addition_greedy(seed)
        try_update_best(res)
        
    # 2. Advanced Removal (More thorough)
    # Run with different seeds to explore different removal paths
    removal_seeds = list(range(50))
    random.shuffle(removal_seeds)
    
    for seed in removal_seeds:
        if time.time() > deadline - 2.0:
            break
        res = strategy_removal_advanced(seed)
        try_update_best(res)
        
    # Fallback if something went wrong (should not happen)
    if not best_set:
        best_set = [100]
        
    return best_set

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Best size: {len(res)}")
    print(f"Set: {sorted(res)}")
    # Verify
    def check(S):
        target = 200
        mask = 1
        for x in S:
            mask = mask | (mask << x)
            mask &= (1 << (target + 1)) - 1
        return (mask >> target) & 1
    print(f"Valid (no sum 200): {not check(res)}")
