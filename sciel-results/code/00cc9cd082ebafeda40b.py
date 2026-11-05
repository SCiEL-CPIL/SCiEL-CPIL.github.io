import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 4096
    max_sum = 3 * n
    # Use bytearray for fast membership and update
    used_sums = bytearray(max_sum + 1)
    
    S = []
    best_set = []
    best_size = 0
    
    def can_add(c):
        # Check 3c
        if used_sums[3 * c]:
            return False
        
        # Check 2c + x
        for x in S:
            if used_sums[2 * c + x]:
                return False
            
        # Check c + x + y
        len_S = len(S)
        for i in range(len_S):
            x = S[i]
            for j in range(i, len_S):
                y = S[j]
                if used_sums[c + x + y]:
                    return False
        return True

    def add(c):
        used_sums[3 * c] = 1
        for x in S:
            used_sums[2 * c + x] = 1
        len_S = len(S)
        for i in range(len_S):
            x = S[i]
            for j in range(i, len_S):
                y = S[j]
                used_sums[c + x + y] = 1
        S.append(c)

    def remove_last():
        c = S.pop()
        used_sums[3 * c] = 0
        for x in S:
            used_sums[2 * c + x] = 0
        len_S = len(S)
        for i in range(len_S):
            x = S[i]
            for j in range(i, len_S):
                y = S[j]
                used_sums[c + x + y] = 0

    def reset_used():
        for i in range(max_sum + 1):
            used_sums[i] = 0

    # 1. Initial Greedy
    for c in range(1, n + 1):
        if time.time() > deadline:
            break
        if can_add(c):
            add(c)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
    
    if len(S) == 0:
        return best_set

    # 2. Local Search / Backtracking to improve
    # Strategy: Try to replace the last d elements with d+1 elements.
    
    # We need a helper to check if a candidate can be added to a specific state
    # To speed up backtracking, we pass the current S and used_sums.
    
    def can_add_to_state(state_S, c, state_used):
        if state_used[3 * c]:
            return False
        for x in state_S:
            if state_used[2 * c + x]:
                return False
        len_S = len(state_S)
        for i in range(len_S):
            x = state_S[i]
            for j in range(i, len_S):
                y = state_S[j]
                if state_used[c + x + y]:
                    return False
        return True

    def add_to_state(state_S, c, state_used):
        state_used[3 * c] = 1
        for x in state_S:
            state_used[2 * c + x] = 1
        len_S = len(state_S)
        for i in range(len_S):
            x = state_S[i]
            for j in range(i, len_S):
                y = state_S[j]
                state_used[c + x + y] = 1
        state_S.append(c)

    def remove_last_from_state(state_S, state_used):
        c = state_S.pop()
        state_used[3 * c] = 0
        for x in state_S:
            state_used[2 * c + x] = 0
        len_S = len(state_S)
        for i in range(len_S):
            x = state_S[i]
            for j in range(i, len_S):
                y = state_S[j]
                state_used[c + x + y] = 0

    # Backtracking search for k elements from candidates
    def search_k_elements(base_S, base_used, candidates, k, limit_time):
        # base_S, base_used are the fixed prefix
        # We want to find k elements from candidates to append
        # Returns list of elements or None
        if k == 0:
            return []
        
        # Pruning: not enough candidates left?
        if len(candidates) < k:
            return None
            
        # Heuristic: try smaller candidates first? Or random?
        # Smaller is usually better for packing.
        
        for i, c in enumerate(candidates):
            if time.time() > limit_time:
                return None
            
            if can_add_to_state(base_S, c, base_used):
                # Tentatively add
                new_S = base_S + [c]
                # Copy used? This is expensive.
                # Better: modify and revert.
                # But we are in a recursive function.
                # We'll modify base_used and base_S in place?
                # No, base_S is a list. We can append/pop.
                # But base_used is a bytearray. We need to revert.
                
                # To avoid deep copying, we'll do this:
                # 1. Record the sums we are about to set.
                # 2. Set them.
                # 3. Recurse.
                # 4. Unset them.
                
                sums_to_set = [3 * c]
                for x in base_S:
                    sums_to_set.append(2 * c + x)
                len_S = len(base_S)
                for i2 in range(len_S):
                    x = base_S[i2]
                    for j2 in range(i2, len_S):
                        y = base_S[j2]
                        sums_to_set.append(c + x + y)
                
                # Check if any sum is already 1? 
                # can_add_to_state already checked.
                # So all are 0.
                
                for s in sums_to_set:
                    base_used[s] = 1
                base_S.append(c)
                
                # Recurse
                # We need to pass the remaining candidates
                # We can pass candidates[i+1:] but that creates a new list.
                # For small k, it's fine.
                res = search_k_elements(base_S, base_used, candidates[i+1:], k-1, limit_time)
                
                if res is not None:
                    return [c] + res
                
                # Revert
                base_S.pop()
                for s in sums_to_set:
                    base_used[s] = 0
                    
        return None

    # Main improvement loop
    # We try to increase size by 1 by swapping the last d elements.
    
    while time.time() < deadline:
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_set = S[:]
            
        # Try different window sizes d
        for d in range(1, 5):
            if len(S) < d:
                continue
                
            # Save state
            saved_S = S[:]
            saved_used = bytearray(used_sums)
            
            # Remove last d elements
            for _ in range(d):
                remove_last()
                
            # Candidates: all numbers not in S
            # We want to pick d+1 numbers.
            # To speed up, we can restrict candidates to a range.
            # But for now, use all.
            # Sorting candidates helps?
            # We want to find d+1 numbers.
            
            # Generate candidates
            # This is O(n).
            # We can maintain a set of used numbers?
            # For now, list comprehension.
            S_set = set(S)
            candidates = [c for c in range(1, n + 1) if c not in S_set]
            
            # Sort candidates?
            # Smaller numbers are generally better.
            candidates.sort()
            
            # Try to find d+1 elements
            # Limit time for this search
            search_deadline = time.time() + 1.0 # 1 second per attempt
            
            new_elements = search_k_elements(S, used_sums, candidates, d + 1, search_deadline)
            
            if new_elements is not None:
                # Success!
                # The state S and used_sums have been modified by the successful path?
                # No, search_k_elements reverts on failure.
                # On success, it returns, but does it leave the state modified?
                # Let's check search_k_elements logic.
                # It appends and sets used.
                # If res is not None, it returns [c] + res.
                # The state S and used_sums are currently in the state where 'c' was added 
                # and the rest of the recursion succeeded.
                # Wait, the recursion reverts on failure, but on success it returns up the stack.
                # The modifications from the successful branch are STILL IN PLACE.
                # So S and used_sums now contain the new elements.
                
                # We need to ensure we didn't mess up the "saved" state logic.
                # We removed d elements, then added d+1.
                # So size increased by 1.
                
                # Update best
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]
                    
                # Continue loop to try to improve further
                break # Break d loop, restart while loop with new S
                
            else:
                # Restore state
                S = saved_S
                used_sums = saved_used
                # Note: S is a list, so S = saved_S rebinds it.
                # But remove_last modified S in place.
                # So we must restore S content.
                # saved_S was a copy.
                # We need to clear S and extend?
                # Actually, in the loop, we did `remove_last()` which popped from S.
                # So S is now shorter.
                # We need to put the removed elements back.
                # It's safer to just reload S from saved_S.
                # But `S` is a local variable in construct().
                # `remove_last` modifies the list object.
                # `saved_S` is a new list object.
                # So `S = saved_S` makes S point to the old full list.
                # `used_sums = saved_used` makes used_sums point to the old bytearray.
                
                # However, `search_k_elements` modifies `base_S` and `base_used` which are `S` and `used_sums`.
                # Since we restored them, we are good.
                
    return best_set
