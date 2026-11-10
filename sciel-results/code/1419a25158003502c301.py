import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Precompute reciprocals
    recip = {i: Fraction(1, i) for i in range(1, 25)}
    all_elements = list(range(1, 25))
    
    def find_violation(S):
        """
        Returns (x, subset) if 1/x == sum(1/y for y in subset) for some x in S and subset of S\{x}.
        Returns None if valid.
        """
        S_list = list(S)
        n = len(S_list)
        if n < 2:
            return None
            
        for i, x in enumerate(S_list):
            target = recip[x]
            others = S_list[:i] + S_list[i+1:]
            if not others:
                continue
                
            m = len(others)
            half = m // 2
            left = others[:half]
            right = others[half:]
            
            # MITM
            left_sums = [Fraction(0)]
            for y in left:
                ry = recip[y]
                new_sums = [s + ry for s in left_sums]
                left_sums.extend(new_sums)
                
            right_sums = [Fraction(0)]
            for y in right:
                ry = recip[y]
                new_sums = [s + ry for s in right_sums]
                right_sums.extend(new_sums)
                
            right_set = set(right_sums)
            
            # Check if target is in left_sums + right_set
            # We need to find l in left_sums such that target - l in right_set
            # Also need to reconstruct subset if found
            # For efficiency in repair, we just need to know IF it exists.
            # But to remove specific elements, it's better to know which ones.
            # However, any element from the violating set can be removed.
            # So we can just return x and the list of 'others' if a violation is found?
            # No, we should return the specific subset to be precise, or just the group 'others' if we remove randomly from it.
            # Actually, if we remove ANY element from 'others' or 'x', the specific equation breaks.
            # So we can just return (x, others) and let the caller pick an element to remove from {x} U others.
            
            found = False
            for l in left_sums:
                if (target - l) in right_set:
                    found = True
                    break
                    
            if found:
                return (x, others)
                
        return None

    def is_valid(S):
        return find_violation(S) is None

    best_set = set()
    best_size = 0
    
    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
    # Strategy: Iterative Repair and Grow
    # We will run multiple independent search trails.
    
    for attempt in range(200): # Max attempts, limited by time
        if time.time() > deadline:
            break
            
        # Initialize with full set or random large set
        if attempt == 0:
            S = set(all_elements)
        else:
            # Random subset of size 15-20
            size = random.randint(15, 20)
            S = set(random.sample(all_elements, size))
            
        # Phase 1: Repair
        # If S is not valid, remove elements until it is.
        # To avoid getting stuck in a small set, we remove "bad" elements.
        # We don't have a perfect notion of "bad", so we remove randomly from the violating group.
        
        max_repair_steps = 50
        for _ in range(max_repair_steps):
            if time.time() > deadline:
                break
            viol = find_violation(S)
            if viol is None:
                break
            
            x, others = viol
            # Candidate removals: x or any element in others
            candidates = [x] + list(others)
            # Randomly pick one to remove
            # Maybe bias towards removing smaller numbers? They are more likely to be in many sums.
            # Let's try random first.
            rem = random.choice(candidates)
            S.remove(rem)
            
        if not S:
            continue
            
        update_best(S)
        
        # Phase 2: Grow
        # Try to add elements from the complement in random order.
        complement = list(set(all_elements) - S)
        random.shuffle(complement)
        
        for c in complement:
            if time.time() > deadline:
                break
            new_S = S | {c}
            if is_valid(new_S):
                S = new_S
                update_best(S)
                
        # Phase 3: Perturb (Escape Local Optima)
        # If we can't add any more, remove a few random elements and try to grow again.
        # This is optional if we restart with new random sets, but helps within a trail.
        # Given the time limit, restarting with new random sets (attempt loop) is probably sufficient.
        # But let's add a simple kick: remove 1-2 random elements and grow.
        
        if len(S) > 5: # Don't kick if set is too small
            kick_size = random.randint(1, 2)
            to_remove = random.sample(list(S), kick_size)
            for r in to_remove:
                S.remove(r)
                
            # Regrow
            complement = list(set(all_elements) - S)
            random.shuffle(complement)
            for c in complement:
                if time.time() > deadline:
                    break
                new_S = S | {c}
                if is_valid(new_S):
                    S = new_S
                    update_best(S)

    # Fallback
    if not best_set:
        best_set = {24}
        
    return list(best_set)
