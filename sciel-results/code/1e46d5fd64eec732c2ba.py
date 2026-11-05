"""Optimized construction: structured base + local search for max 3-APs with no 4-AP."""
import time
import random

def construct():
    start = time.time()
    deadline = start + 80  # 80s safety margin within 90s
    
    N = 256
    
    # Precompute: for each element x, list of (y, z) such that x, y, z, w form a 4-AP
    # i.e., y = x + d, z = x + 2d, w = x + 3d for some d > 0
    # When adding x, we need: no existing pair (y,z) where y-z = x-y = z-x... 
    # Actually: x,y,z,w with d>0. If x is new, conflicts:
    #   Case 1: x is first: need y=x+d, z=x+2d, w=x+3d all in set
    #   Case 2: x is second: need x-d, x+d, x+2d in set
    #   Case 3: x is third: need x-2d, x-d, x+d in set
    #   Case 4: x is fourth: need x-3d, x-2d, x-d in set
    
    # For fast checking, precompute for each x the list of "forbidden pairs"
    # If x is in set, then no pair (a,b) in set can complete a 4-AP with x.
    
    # Better approach: precompute for each x, the set of pairs that would conflict
    # When x is added, check if any 4-AP containing x has all other 3 elements present.
    
    # Let's precompute all 4-APs in [1,256]
    # A 4-AP: (a, a+d, a+2d, a+3d) where a >= 1, a+3d <= 256
    all_4aps = []
    for d in range(1, 86):  # 3d <= 255, so d <= 85
        for a in range(1, 257 - 3*d):
            all_4aps.append((a, a+d, a+2*d, a+3*d))
    
    # For each element x, which 4-APs contain it?
    # And for quick checking: when adding x to set S, is there a 4-AP containing x
    # where the other 3 elements are all in S?
    
    # Precompute: for each x, list of (other1, other2, other3) for each 4-AP containing x
    x_to_4ap_others = {x: [] for x in range(1, 257)}
    for (a, b, c, d_) in all_4aps:
        for x in (a, b, c, d_):
            others = tuple(v for v in (a, b, c, d_) if v != x)
            x_to_4ap_others[x].append(others)
    
    def has_4ap_conflict(x, S):
        """Check if adding x to S creates a 4-AP."""
        for (o1, o2, o3) in x_to_4ap_others[x]:
            if o1 in S and o2 in S and o3 in S:
                return True
        return False
    
    def count_3aps(S):
        """Count distinct 3-term APs (positive difference) in S."""
        S_set = S if isinstance(S, set) else set(S)
        count = 0
        for a in S_set:
            for d in range(1, 257 - a):
                if a + d in S_set and a + 2*d in S_set:
                    count += 1
        return count
    
    def count_3aps_fast(S_set):
        """Faster count using sorted list."""
        count = 0
        for a in S_set:
            # For each possible difference, check if a+d and a+2d are in set
            max_d = (256 - a) // 2
            for d in range(1, max_d + 1):
                if (a + d) in S_set and (a + 2*d) in S_set:
                    count += 1
        return count
    
    # Strategy: Start with a structured set that's known to have many 3-APs and no 4-APs
    # Then do local search.
    
    # Approach: Use a "greedy with lookahead" to build initial set, then optimize.
    
    # First, try a structured approach: take numbers based on base-3 representation
    # Numbers with no digit > 1 in base 3 avoid 3-APs, but we WANT 3-APs.
    # Instead, let's use a dense set and remove elements to break 4-APs.
    
    # Better: Start with a random-ish dense set, then iteratively:
    # 1. Remove elements that are in 4-APs (choosing removals that minimize 3-AP loss)
    # 2. Add elements that increase 3-APs without creating 4-APs
    # 3. Swap elements
    
    # Let's try multiple strategies and keep the best.
    
    best_set = set()
    best_score = -1
    
    def evaluate_set(S_set):
        """Return (num_3aps, is_valid) for a set."""
        # Check 4-AP validity
        for (a, b, c, d_) in all_4aps:
            if a in S_set and b in S_set and c in S_set and d_ in S_set:
                return 0, False
        # Count 3-APs
        cnt = 0
        for a in S_set:
            max_d = (256 - a) // 2
            for d in range(1, max_d + 1):
                if (a + d) in S_set and (a + 2*d) in S_set:
                    cnt += 1
        return cnt, True
    
    # Strategy 1: Greedy construction (like seed but with scoring)
    # For each element in random order, add it if it doesn't create a 4-AP
    # and maximizes marginal 3-AP gain
    
    def marginal_3ap_gain(x, S_set):
        """How many new 3-APs does adding x create?"""
        gain = 0
        # x as first term: x, x+d, x+2d
        for d in range(1, (256 - x) // 2 + 1):
            if (x + d) in S_set and (x + 2*d) in S_set:
                gain += 1
        # x as middle: x-d, x, x+d
        for d in range(1, x):
            if (x - d) in S_set and (x + d) in S_set:
                gain += 1
        # x as last: x-2d, x-d, x
        for d in range(1, (x - 1) // 2 + 1):
            if (x - 2*d) in S_set and (x - d) in S_set:
                gain += 1
        return gain
    
    # Run multiple random greedy constructions
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        rng = random.Random(trial * 12345 + 67890)
        S_set = set()
        order = list(range(1, 257))
        rng.shuffle(order)
        
        for x in order:
            if has_4ap_conflict(x, S_set):
                continue
            # Check if adding x is beneficial (it always adds some 3-APs potentially)
            S_set.add(x)
        
        score, valid = evaluate_set(S_set)
        if valid and score > best_score:
            best_score = score
            best_set = S_set.copy()
    
    # Strategy 2: Start dense, remove to break 4-APs
    for trial in range(10):
        if time.time() > deadline - 25:
            break
        rng = random.Random(trial * 9999 + 4321)
        # Start with all elements
        S_set = set(range(1, 257))
        
        # Iteratively remove elements that are in 4-APs
        # Choose removal that minimizes 3-AP loss
        for iteration in range(200):
            if time.time() > deadline - 25:
                break
            # Find all 4-APs fully in S_set
            found_4ap = None
            for (a, b, c, d_) in all_4aps:
                if a in S_set and b in S_set and c in S_set and d_ in S_set:
                    found_4ap = (a, b, c, d_)
                    break
            
            if found_4ap is None:
                break
            
            # Choose which element to remove from this 4-AP
            # Try each and pick the one with least 3-AP loss
            best_remove = None
            best_loss = float('inf')
            for elem in found_4ap:
                # Count 3-APs involving elem
                loss = 0
                # elem as first
                for d in range(1, (256 - elem) // 2 + 1):
                    if (elem + d) in S_set and (elem + 2*d) in S_set:
                        loss += 1
                # elem as middle
                for d in range(1, elem):
                    if (elem - d) in S_set and (elem + d) in S_set:
                        loss += 1
                # elem as last
                for d in range(1, (elem - 1) // 2 + 1):
                    if (elem - 2*d) in S_set and (elem - d) in S_set:
                        loss += 1
                if loss < best_loss:
                    best_loss = loss
                    best_remove = elem
            
            if best_remove is not None:
                S_set.discard(best_remove)
        
        score, valid = evaluate_set(S_set)
        if valid and score > best_score:
            best_score = score
            best_set = S_set.copy()
    
    # Strategy 3: Local search on best_set
    # Try adding, removing, and swapping elements
    if time.time() < deadline - 15:
        current_set = best_set.copy()
        current_score = best_score
        
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            
            # Try adding each element not in set
            for x in range(1, 257):
                if x in current_set:
                    continue
                if has_4ap_conflict(x, current_set):
                    continue
                # Calculate marginal gain
                gain = marginal_3ap_gain(x, current_set)
                if gain > 0:
                    current_set.add(x)
                    current_score += gain
                    improved = True
                    break
            
            # Try removing elements that are in 4-APs (shouldn't happen if valid)
            # Try swapping: remove one, add another
            if not improved:
                for x_remove in list(current_set):
                    if time.time() > deadline - 15:
                        break
                    temp_set = current_set - {x_remove}
                    # Count 3-APs lost
                    loss = 0
                    for d in range(1, (256 - x_remove) // 2 + 1):
                        if (x_remove + d) in temp_set and (x_remove + 2*d) in temp_set:
                            loss += 1
                    for d in range(1, x_remove):
                        if (x_remove - d) in temp_set and (x_remove + d) in temp_set:
                            loss += 1
                    for d in range(1, (x_remove - 1) // 2 + 1):
                        if (x_remove - 2*d) in temp_set and (x_remove - d) in temp_set:
                            loss += 1
                    
                    # Try adding each element not in temp_set
                    for x_add in range(1, 257):
                        if x_add in temp_set:
                            continue
                        if has_4ap_conflict(x_add, temp_set):
                            continue
                        gain = marginal_3ap_gain(x_add, temp_set)
                        if gain > loss:
                            current_set = temp_set | {x_add}
                            current_score = current_score - loss + gain
                            improved = True
                            break
                    if improved:
                        break
        
        if current_score > best_score:
            best_score = current_score
            best_set = current_set.copy()
    
    # Final validation
    final_score, valid = evaluate_set(best_set)
    if not valid or final_score == 0:
        # Fallback: use seed-like greedy
        S_set = set()
        for x in range(1, 257):
            if not has_4ap_conflict(x, S_set):
                S_set.add(x)
        return sorted(S_set)
    
    return sorted(best_set)
