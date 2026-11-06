import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s to be safe
    
    required = [1, 2, 5, 11]
    best_set = set(required)
    best_size = len(best_set)
    
    # Max sum is 284 + 284 = 568. We need bits 0..568.
    # Python integers handle arbitrary size, so we can use a bitmask.
    
    def get_sum_mask(S):
        mask = 0
        S_list = sorted(S)
        for i in range(len(S_list)):
            for j in range(i, len(S_list)):
                mask |= (1 << (S_list[i] + S_list[j]))
        return mask

    def can_add(x, S, mask):
        # Check if x can be added to S
        # New sums: x + y for y in S, and 2*x
        for y in S:
            s = x + y
            if (mask >> s) & 1:
                return False
        s = 2 * x
        if (mask >> s) & 1:
            return False
        return True

    def add_to_mask(x, S, mask):
        new_mask = mask
        for y in S:
            new_mask |= (1 << (x + y))
        new_mask |= (1 << (2 * x))
        return new_mask

    def greedy_construct(S, mask, target_size):
        """Greedily add elements to reach target_size or stuck."""
        S = set(S)
        while len(S) < target_size:
            candidates = []
            for x in range(1, 285):
                if x not in S and can_add(x, S, mask):
                    candidates.append(x)
            if not candidates:
                break
            # Pick a random candidate to introduce diversity
            x = random.choice(candidates)
            S.add(x)
            mask = add_to_mask(x, S, mask)
        return S, mask

    def backtrack_search(S, mask, target_size, depth=0):
        """
        Randomized backtracking to reach target_size.
        Returns (success, S, mask)
        """
        if len(S) == target_size:
            return True, S, mask
        
        # Generate valid candidates
        candidates = []
        for x in range(1, 285):
            if x not in S and can_add(x, S, mask):
                candidates.append(x)
                
        if not candidates:
            return False, S, mask
            
        # Randomize order to explore different branches
        random.shuffle(candidates)
        
        # To avoid infinite loops or too deep searches, we can limit candidates tried per level
        # But since we have time limit, we just try them.
        # Optimization: If we are close to target, try all. If far, maybe sample?
        # Let's just iterate.
        
        for x in candidates:
            new_S = S | {x}
            new_mask = add_to_mask(x, S, mask)
            success, res_S, res_mask = backtrack_search(new_S, new_mask, target_size, depth + 1)
            if success:
                return True, res_S, res_mask
                
        return False, S, mask

    # Main loop
    while time.time() < deadline:
        # 1. Greedy warm-up
        S, mask = greedy_construct(set(required), get_sum_mask(required), 18)
        
        # 2. Backtracking to reach 20
        # If greedy got 20, we are done
        if len(S) == 20:
            if len(S) > best_size:
                best_size = 20
                best_set = S
            break
            
        # Try to extend from greedy result to 20
        success, final_S, final_mask = backtrack_search(S, mask, 20)
        
        if success and len(final_S) > best_size:
            best_size = len(final_S)
            best_set = final_S
            if best_size == 20:
                break
                
        # Also try a pure random restart from scratch with different greedy bias?
        # The current loop restarts the greedy every time.
        
    return list(best_set)
