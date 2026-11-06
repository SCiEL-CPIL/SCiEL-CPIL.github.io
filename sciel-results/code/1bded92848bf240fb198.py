import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    required = [1, 2, 5, 11]
    max_val = 284
    required_set = set(required)
    
    # Helper to get initial sums for a set
    def get_sums(seq):
        sums = [False] * (2 * max_val + 2)
        for i in range(len(seq)):
            for j in range(i, len(seq)):
                s = seq[i] + seq[j]
                sums[s] = True
        return sums

    # Helper to check if adding x to seq is valid
    def can_add(seq, x, sums_set):
        # sums_set is a list of booleans for sums
        # Check x + y for all y in seq
        for y in seq:
            if sums_set[x + y]:
                return False
        # Check 2*x
        if sums_set[2 * x]:
            return False
        return True

    def update_sums(seq, new_elem, sums_set):
        for y in seq:
            sums_set[new_elem + y] = True
        sums_set[2 * new_elem] = True

    # Known optimal 20-mark Golomb ruler (length 283)
    # Marks: 0, 1, 4, 10, 18, 23, 29, 42, 47, 58, 72, 85, 91, 101, 107, 132, 137, 162, 170, 195
    # We need to shift this to [1, 284]. 
    # The ruler marks are relative distances. 
    # Let's use a known good Sidon set in [1, 284] if possible, or construct one.
    # Actually, the problem states the optimal 20-mark ruler has length 283.
    # Marks: 0, 1, 4, 10, 18, 23, 29, 42, 47, 58, 72, 85, 91, 101, 107, 132, 137, 162, 170, 195
    # This set sums to max 195+195=390? No, Sidon condition is on sums a+b.
    # Wait, the standard Golomb ruler ensures distinct *differences*. 
    # Sidon sets require distinct *sums* a+b (a<=b).
    # A set with distinct differences is NOT necessarily a Sidon set for sums.
    # Example: {1, 4, 10}. Diffs: 3, 6, 9. Sums: 2, 5, 11, 8, 14, 20. Distinct.
    # However, the relationship is: if differences are distinct, sums might not be.
    # But there is a strong correlation. 
    # Let's rely on the randomized search to find a valid set, but use a good heuristic.
    
    # Initial best: The mandatory set
    best_set = required[:]
    best_size = len(best_set)
    best_sums = get_sums(best_set)
    
    # Candidates pool
    candidates = [x for x in range(1, max_val + 1) if x not in required_set]

    # Strategy 1: Randomized Greedy with Conflict Heuristic
    # We prioritize candidates that "fit" well. 
    # A proxy for "fit" is how many other candidates they conflict with.
    # We want to pick candidates that conflict with *few* other available candidates?
    # Or simply: pick randomly, but if stuck, use backtracking.
    
    def run_randomized_greedy():
        curr = required[:]
        curr_sums = get_sums(curr)
        
        # Create a pool
        pool = [x for x in candidates]
        random.shuffle(pool)
        
        # To make it "conflict-aware", we can sort the pool dynamically?
        # That's expensive. Instead, we just do standard randomized greedy.
        # The "structural change" is in the local search phase.
        
        for x in pool:
            if can_add(curr, x, curr_sums):
                curr.append(x)
                update_sums(curr, x, curr_sums)
        
        return curr

    # Strategy 2: Conflict-Driven Local Search (Min-Cut Approximation)
    # Idea: If we have a set S, we look for a candidate x not in S that we *want* to add.
    # If x conflicts with S, it means there exist a,b in S such that a+b = x+c for some c?
    # No, conflict means x+y is already a sum in S.
    # To add x, we must remove elements from S that cause the conflict.
    # This is a "Set Cover" or "Hitting Set" problem: which elements in S are responsible for the sums that block x?
    
    def find_conflicting_elements(curr, x, curr_sums):
        # Returns list of elements in curr that cause x to be invalid
        # x is invalid if (x+y) in curr_sums for some y in curr, or 2x in curr_sums.
        # If (x+y) in curr_sums, it means there exist a,b in curr such that a+b = x+y.
        # The "culprits" are a and b.
        # We want to remove the "most critical" culprit.
        
        # This is hard to track exactly without storing the pairs.
        # For now, we use a simpler heuristic:
        # If we can't add x, we pick a random element from curr (non-required) to remove,
        # but we bias towards elements that are "large" or "central"?
        # Let's stick to the inspiration's "kick-out" but refine it.
        
        pass

    # Improved Local Search:
    # 1. Take best_set.
    # 2. Identify "bottleneck" elements: elements that block many other candidates.
    # 3. Remove one bottleneck.
    # 4. Re-fill greedily.
    
    def calculate_bottleneck_score(curr, curr_sums):
        # For each element e in curr, how many candidates c (not in curr) are blocked by e?
        # A candidate c is blocked by e if (c+e) in curr_sums OR (c+e) == 2*e? No.
        # c is blocked if for some y in curr, c+y is a sum in curr_sums.
        # This is complex. 
        # Simpler heuristic: Remove the element that has the largest value? 
        # Or the element that is part of the most "tight" sums?
        
        # Let's use a simpler "random kick" but with a time check.
        pass

    # Let's implement the main loop
    while time.time() < deadline:
        # 1. Try to extend best_set
        # If best_set is not maximal, try to add more.
        # We do this by running a greedy pass on the remaining candidates.
        
        current = best_set[:]
        current_sums = get_sums(current)
        
        # Find valid additions
        valid_adds = []
        for x in candidates:
            if x not in set(current):
                if can_add(current, x, current_sums):
                    valid_adds.append(x)
        
        if valid_adds:
            # Add the one that seems best? Or just add all possible greedily?
            # Greedy addition: pick one, update, repeat.
            # We'll just pick a random valid add to explore different branches
            # But usually, adding *any* valid one is good.
            # To maximize size, we should add as many as possible.
            # So we loop.
            changed = True
            while changed:
                changed = False
                valid_adds = []
                for x in candidates:
                    if x not in set(current):
                        if can_add(current, x, current_sums):
                            valid_adds.append(x)
                if valid_adds:
                    # Pick the smallest? Largest? Random?
                    # Smallest might pack better?
                    # Let's pick random to avoid bias.
                    x = random.choice(valid_adds)
                    current.append(x)
                    update_sums(current, x, current_sums)
                    changed = True
            
            if len(current) > best_size:
                best_size = len(current)
                best_set = current[:]
                best_sums = current_sums[:]
            continue
        
        # 2. If no valid adds, we are stuck. Perform Local Search.
        # "Kick-out" a non-required element and try to refill.
        if len(best_set) <= len(required):
            break
            
        removable = [x for x in best_set if x not in required_set]
        if not removable:
            break
            
        # Pick a random element to remove
        # To make it "structural", let's pick the element that is "most isolated" or "least useful"?
        # Actually, let's just pick random, but try a few times.
        
        # To improve efficiency, we can try removing the element that allows the most new additions?
        # That requires lookahead.
        
        # Let's try a specific heuristic: Remove the largest element?
        # Often, large elements waste space in the sum range [1, 284].
        # If we remove a large element, we might be able to add two smaller ones?
        
        rem_val = max(removable) # Try removing the largest first
        
        temp_set = [x for x in best_set if x != rem_val]
        temp_sums = get_sums(temp_set)
        
        # Refill
        pool = [x for x in candidates if x not in set(temp_set)]
        random.shuffle(pool)
        
        for x in pool:
            if can_add(temp_set, x, temp_sums):
                temp_set.append(x)
                update_sums(temp_set, x, temp_sums)
        
        if len(temp_set) > best_size:
            best_size = len(temp_set)
            best_set = temp_set[:]
            best_sums = temp_sums[:]
        else:
            # If removing the largest didn't help, try a random one
            rem_val = random.choice(removable)
            temp_set = [x for x in best_set if x != rem_val]
            temp_sums = get_sums(temp_set)
            
            pool = [x for x in candidates if x not in set(temp_set)]
            random.shuffle(pool)
            
            for x in pool:
                if can_add(temp_set, x, temp_sums):
                    temp_set.append(x)
                    update_sums(temp_set, x, temp_sums)
            
            if len(temp_set) > best_size:
                best_size = len(temp_set)
                best_set = temp_set[:]
                best_sums = temp_sums[:]
                
        # Occasionally, do a full randomized restart to escape local optima
        if random.random() < 0.1:
            new_set = run_randomized_greedy()
            if len(new_set) > best_size:
                best_size = len(new_set)
                best_set = new_set[:]
                best_sums = get_sums(best_set)

    return sorted(best_set)
