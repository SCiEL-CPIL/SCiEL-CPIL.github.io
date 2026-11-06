import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    prescribed = [1, 2, 5, 11]
    prescribed_set = frozenset(prescribed)
    
    def is_valid_sidon(s):
        """Check if s is a valid Sidon set."""
        sums = set()
        sl = sorted(s)
        for i in range(len(sl)):
            for j in range(i, len(sl)):
                sm = sl[i] + sl[j]
                if sm in sums:
                    return False
                sums.add(sm)
        return True
    
    def get_sums(s):
        """Get all pairwise sums for a sorted list."""
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        return sums
    
    def can_add(x, s, sums):
        """Check if x can be added to Sidon set s with given sums."""
        for y in s:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True
    
    def get_new_sums(x, s):
        return {x + y for y in s} | {2 * x}
    
    def exhaustive_add(s, sums, candidates):
        """Try to add all possible candidates in order. Returns (new_s, new_sums, added_count)."""
        added = 0
        for x in candidates:
            if x in s:
                continue
            if can_add(x, s, sums):
                s.append(x)
                sums = sums | get_new_sums(x, s)
                added += 1
        return s, sums, added
    
    # Strategy 1: Deterministic greedy with multiple orderings
    best_set = list(prescribed)
    best_size = len(best_set)
    
    def try_greedy(order, time_limit=2.0):
        nonlocal best_set, best_size
        if time.time() > deadline - 5:
            return
        s = list(prescribed)
        sums = get_sums(s)
        start = time.time()
        
        for x in order:
            if time.time() - start > time_limit:
                break
            if x in s:
                continue
            if can_add(x, s, sums):
                s.append(x)
                sums = sums | get_new_sums(x, s)
        
        if len(s) > best_size:
            best_set = list(s)
            best_size = len(s)
    
    # Try greedy with various orderings
    remaining = list(range(12, 285))
    rng = random.Random(12345)
    
    # Ascending greedy
    try_greedy(remaining, time_limit=1.0)
    # Descending greedy  
    try_greedy(list(reversed(remaining)), time_limit=1.0)
    
    # Several random orderings
    for i in range(50):
        if time.time() > deadline - 10:
            break
        order = remaining[:]
        rng.shuffle(order)
        try_greedy(order, time_limit=0.3)
    
    # Strategy 2: Systematic local search - try all candidates, then swap
    def systematic_local_search(start_s, time_limit=60.0):
        nonlocal best_set, best_size
        if time.time() > deadline - 5:
            return
        s = sorted(list(start_s))
        sums = get_sums(s)
        start = time.time()
        
        all_vals = list(range(1, 285))
        s_set = set(s)
        
        # Phase 1: Exhaustively try to add any candidate
        improved = True
        while improved and time.time() - start < time_limit and time.time() < deadline - 5:
            improved = False
            for x in all_vals:
                if time.time() > deadline - 5:
                    return
                if x in s_set:
                    continue
                if can_add(x, s, sums):
                    s.append(x)
                    s_set.add(x)
                    sums = sums | get_new_sums(x, s)
                    if len(s) > best_size:
                        best_set = list(s)
                        best_size = len(s)
                    improved = True
                    break
        
        # Phase 2: Try single swaps (remove one non-prescribed, add one new)
        if time.time() - start < time_limit and time.time() < deadline - 5:
            for _ in range(200):  # Limit swap attempts
                if time.time() > deadline - 5:
                    return
                if time.time() - start > time_limit:
                    break
                
                removable = [x for x in s if x not in prescribed_set]
                if not removable:
                    break
                
                # Try each removable element
                found_swap = False
                for x_remove in removable:
                    if time.time() > deadline - 5:
                        return
                    if time.time() - start > time_limit:
                        break
                    
                    # Temporarily remove
                    s.remove(x_remove)
                    s_set.discard(x_remove)
                    sums = get_sums(s)
                    
                    # Try to add a new element
                    for x_add in all_vals:
                        if time.time() > deadline - 5:
                            # Restore and return
                            s.append(x_remove)
                            s_set.add(x_remove)
                            return
                        if x_add in s_set:
                            continue
                        if can_add(x_add, s, sums):
                            s.append(x_add)
                            s_set.add(x_add)
                            sums = sums | get_new_sums(x_add, s)
                            if len(s) > best_size:
                                best_set = list(s)
                                best_size = len(s)
                            found_swap = True
                            break
                    
                    # If no swap found, restore
                    if not found_swap:
                        s.append(x_remove)
                        s_set.add(x_remove)
                        sums = get_sums(s)
                    
                    if found_swap:
                        break
        
        # Phase 3: Final exhaustive add pass
        if time.time() - start < time_limit and time.time() < deadline - 5:
            for x in all_vals:
                if time.time() > deadline - 5:
                    return
                if x in s_set:
                    continue
                if can_add(x, s, sums):
                    s.append(x)
                    s_set.add(x)
                    sums = sums | get_new_sums(x, s)
                    if len(s) > best_size:
                        best_set = list(s)
                        best_size = len(s)
    
    # Run systematic local search from best solution
    if time.time() < deadline - 40:
        systematic_local_search(best_set, time_limit=35.0)
    
    # Strategy 3: More aggressive random restarts with greedy
    for trial in range(200):
        if time.time() > deadline - 15:
            break
        s = list(prescribed)
        sums = get_sums(s)
        # Randomly add elements
        candidates = remaining[:]
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 15:
                break
            if can_add(x, s, sums):
                s.append(x)
                sums = sums | get_new_sums(x, s)
        if len(s) > best_size:
            best_set = list(s)
            best_size = len(s)
    
    # Final systematic local search
    if time.time() < deadline - 10 and len(best_set) > 4:
        systematic_local_search(best_set, time_limit=10.0)
    
    # Verify and return
    result = sorted(best_set)
    if not is_valid_sidon(result):
        # Fall back to prescribed
        return list(prescribed)
    
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Result: {result}")
    sums = set()
    valid = True
    for i, x in enumerate(result):
        for j in range(i, len(result)):
            s = x + result[j]
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            break
    print(f"Valid: {valid}")
