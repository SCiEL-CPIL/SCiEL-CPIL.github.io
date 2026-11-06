import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    def get_initial_set():
        """Generate a valid set using greedy approach."""
        S = []
        counts = {}
        
        # Helper to check and add
        def try_add(x):
            # Check sums
            for y in S:
                s = x + y
                if counts.get(s, 0) + 1 > 2:
                    return False
            if counts.get(2 * x, 0) + 1 > 2:
                return False
            
            # Update
            for y in S:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            counts[2 * x] = counts.get(2 * x, 0) + 1
            S.append(x)
            return True

        # Greedy from 1 to 1000
        for x in range(1, 1001):
            try_add(x)
            
        return S, counts

    def local_search(S, counts):
        """Improve the set using add/swap local search."""
        S = sorted(S)
        
        # Precompute candidates not in S
        # We will recompute this or maintain it? 
        # Given small size, recomputing is fine.
        
        while time.time() < deadline:
            # 1. Try to add an element
            added = False
            # Create list of candidates
            S_set = set(S)
            candidates = [x for x in range(1, 1001) if x not in S_set]
            
            # Shuffle to avoid bias
            random.shuffle(candidates)
            
            for x in candidates:
                # Check if x can be added
                valid = True
                for y in S:
                    s = x + y
                    if counts.get(s, 0) + 1 > 2:
                        valid = False
                        break
                if valid and counts.get(2 * x, 0) + 1 > 2:
                    valid = False
                
                if valid:
                    # Add x
                    for y in S:
                        s = x + y
                        counts[s] = counts.get(s, 0) + 1
                    counts[2 * x] = counts.get(2 * x, 0) + 1
                    S.append(x)
                    S.sort()
                    added = True
                    break
            
            if added:
                continue
            
            # 2. If no add possible, try a swap
            if not S:
                break
                
            # Pick random y to remove
            y = random.choice(S)
            
            # Try to find an x to replace y
            # We try a few random candidates
            # Or we can iterate through a subset of candidates
            candidates_swap = [x for x in range(1, 1001) if x not in S_set]
            random.shuffle(candidates_swap)
            
            # Try up to 100 random swaps to avoid spending too much time on one failed swap cycle
            swap_success = False
            for x in candidates_swap[:100]:
                # Check if S - {y} + {x} is valid
                # We need to simulate the counts
                # Decrement y's contributions
                # Increment x's contributions
                # Check max count <= 2
                
                # To speed up, we can just check the affected sums
                # Sums involving y: y+z, 2y
                # Sums involving x: x+z, 2x
                
                # Let's do a full check on the modified counts for affected sums only
                # But we need to be careful not to modify the original counts dict if it fails.
                
                # Create a temporary view or just check values
                # We'll check if any sum would exceed 2.
                
                # Sums to check:
                # 1. Sums that lose a representation (involving y). These will decrease, so no risk of >2.
                # 2. Sums that gain a representation (involving x). These might exceed 2.
                # 3. Sums that are both (if x+y == y+z? No, x!=y. If x+z == y+z? No. 
                #    Overlap: x+z1 == y+z2. 
                #    If we remove y, count of s=y+z2 decreases.
                #    If we add x, count of s=x+z1 increases.
                #    If s is the same, net change is 0? No, we remove one rep and add one rep.
                #    So we just need to check if any sum involving x has count > 2 in the *new* state.
                #    New count for sum s = old_count(s) - delta_remove(s) + delta_add(s).
                #    delta_remove(s) is 1 if s involves y, else 0.
                #    delta_add(s) is 1 if s involves x, else 0.
                
                valid_swap = True
                # Check all sums involving x
                # Note: Sums involving x are x+z for z in S (z!=y) and 2x.
                # Wait, z in S includes y? No, we are removing y. So z in S\{y}.
                
                for z in S:
                    if z == y:
                        continue
                    s = x + z
                    # Current count in original counts
                    c = counts.get(s, 0)
                    # If s involved y, it will decrease by 1
                    # Does s involve y? s = y + w for some w in S.
                    # If x + z = y + w, then c will decrease.
                    # It's complex to check "does s involve y" quickly.
                    # Simpler: Just calculate new count for all sums involving x.
                    # But we also need to ensure we don't miss sums that *only* involved y and are now freed?
                    # No, we only care about max count <= 2.
                    # Sums that decrease are safe.
                    # Sums that increase are x+z and 2x.
                    # So we just need to check if (new_count(x+z) <= 2) and (new_count(2x) <= 2).
                    
                    # new_count(s) = counts.get(s, 0) - (1 if s involves y else 0) + (1 if s involves x else 0)
                    # For s = x+z:
                    # It involves x, so +1.
                    # Does it involve y? 
                    # If x+z = y+w for some w in S, then it involves y.
                    # We can check if (x+z) - y is in S.
                    
                    s = x + z
                    base_c = counts.get(s, 0)
                    # Check if s involves y
                    # s - y = x + z - y. If this is in S, then s was formed by y + (s-y).
                    # Note: s-y could be y itself (if s=2y).
                    # If s-y in S, then base_c includes a rep from y.
                    # So new_c = base_c - 1 + 1 = base_c.
                    # If s-y not in S, then new_c = base_c + 1.
                    
                    # Wait, what if s involves y in multiple ways?
                    # s = y + w1 = y + w2? No, w1=w2.
                    # So s involves y at most once (as y+w).
                    # Exception: s=2y. Involves y as y+y.
                    
                    # So:
                    involves_y = False
                    diff = s - y
                    if diff in S: # And diff != y? No, if diff=y, s=2y. y is in S. So yes.
                        # We must ensure we are not counting the pair (y, y) if y is being removed?
                        # If s=2y, and we remove y, the rep (y,y) is gone.
                        # So yes, involves_y is True.
                        involves_y = True
                    
                    new_c = base_c
                    if involves_y:
                        new_c -= 1
                    new_c += 1 # from adding x
                    
                    if new_c > 2:
                        valid_swap = False
                        break
                
                if valid_swap:
                    # Check 2x
                    s = 2 * x
                    base_c = counts.get(s, 0)
                    involves_y = (s - y) in S
                    new_c = base_c
                    if involves_y:
                        new_c -= 1
                    new_c += 1
                    if new_c > 2:
                        valid_swap = False
                
                if valid_swap:
                    # Perform swap
                    # Update counts
                    # Remove y
                    for z in S:
                        if z == y:
                            continue
                        s = y + z
                        counts[s] -= 1
                        if counts[s] == 0:
                            del counts[s]
                    s = 2 * y
                    counts[s] -= 1
                    if counts[s] == 0:
                        del counts[s]
                    
                    # Add x
                    for z in S:
                        if z == y:
                            continue
                        s = x + z
                        counts[s] = counts.get(s, 0) + 1
                    s = 2 * x
                    counts[s] = counts.get(s, 0) + 1
                    
                    # Update S
                    S.remove(y)
                    S.append(x)
                    S.sort()
                    swap_success = True
                    break
            
            if not swap_success:
                # If no swap worked, we might be stuck.
                # To avoid infinite loop, we can do nothing and let the loop continue (which will try adds again and fail).
                # But we should probably change the random seed or restart if stuck for too long.
                # For now, just continue.
                pass

        return S

    best_S = []
    
    # Run multiple attempts with different random seeds/initializations
    # Since we have time, we can try a few different greedy starts or random restarts.
    
    # Attempt 1: Standard Greedy
    S, counts = get_initial_set()
    S = local_search(S, counts)
    if len(S) > len(best_S):
        best_S = S
    
    # Attempt 2: Randomized Greedy (skip some numbers initially)
    # We can try to start with a random subset and then complete greedily
    for _ in range(5): # 5 more attempts
        if time.time() > deadline - 20: break
        
        random.seed()
        S_init = set()
        # Pick random 20 elements
        init_elements = random.sample(range(1, 1001), 20)
        
        # Greedy complete
        S = []
        counts = {}
        def try_add(x):
            for y in S:
                s = x + y
                if counts.get(s, 0) + 1 > 2:
                    return False
            if counts.get(2 * x, 0) + 1 > 2:
                return False
            for y in S:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            counts[2 * x] = counts.get(2 * x, 0) + 1
            S.append(x)
            return True
        
        # Add init elements if valid
        for x in init_elements:
            try_add(x)
            
        # Complete greedily
        for x in range(1, 1001):
            if x not in S:
                try_add(x)
                
        S = local_search(S, counts)
        if len(S) > len(best_S):
            best_S = S

    return best_S
