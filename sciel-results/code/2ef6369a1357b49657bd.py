"""Sidon set construction with prescribed elements {1, 2, 5, 11} in [1, 284]."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # Prescribed elements
    required = [1, 2, 5, 11]
    
    def get_sums(s):
        """Compute all pairwise sums a+b where a<=b for sorted list s."""
        sums = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                sums.add(s[i] + s[j])
        return sums
    
    def is_valid_addition(s, sums, x):
        """Check if x can be added to s without creating duplicate sums."""
        # New sums would be: x + s[0], x + s[1], ..., x + s[-1], and 2*x
        # Check none of these are in existing sums, and all new sums are distinct
        new_sums = set()
        for y in s:
            ns = x + y
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        ns = 2 * x
        if ns in sums or ns in new_sums:
            return False
        new_sums.add(ns)
        return True
    
    def greedy_fill(s, sums, candidates, rng):
        """Try to add elements greedily from candidates."""
        for x in candidates:
            if x in s:
                continue
            if is_valid_addition(s, sums, x):
                s.append(x)
                s.sort()
                for y in s:
                    if y == x:
                        sums.add(2 * x)
                    else:
                        sums.add(x + y)
                if len(s) >= 20:
                    break
        return s, sums
    
    # Strategy: Multiple randomized restarts with greedy + local search
    best_size = 0
    best_set = None
    best_sums = None
    
    all_candidates = list(range(12, 285))  # Candidates beyond prescribed
    
    # Try multiple random orderings
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Randomize candidate order
        rng = random.Random(restart * 7919 + 42)
        candidates = all_candidates[:]
        rng.shuffle(candidates)
        
        # Start with required elements
        s = required[:]
        s.sort()
        sums = get_sums(s)
        
        # Greedy fill
        s, sums = greedy_fill(s, sums, candidates, rng)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
            best_sums = sums.copy()
            if best_size >= 20:
                break
        
        # Local search: try to improve by swapping
        if len(s) < 20 and time.time() < deadline - 5:
            improved = True
            while improved and len(s) < 20 and time.time() < deadline - 5:
                improved = False
                rng_local = random.Random(int(time.time() * 1000000) + restart)
                
                # Try removing one element and adding two
                for idx in range(len(s)):
                    if s[idx] in required:
                        continue
                    if time.time() > deadline - 5:
                        break
                    
                    x_to_remove = s[idx]
                    # Remove x_to_remove
                    s.remove(x_to_remove)
                    sums = get_sums(s)
                    
                    # Try to add two elements
                    added = 0
                    for y in candidates:
                        if y == x_to_remove or y in s:
                            continue
                        if is_valid_addition(s, sums, y):
                            s.append(y)
                            s.sort()
                            for z in s:
                                if z == y:
                                    sums.add(2 * y)
                                else:
                                    sums.add(y + z)
                            added += 1
                            if added == 2:
                                break
                    
                    if added == 2:
                        improved = True
                        if len(s) > best_size:
                            best_size = len(s)
                            best_set = s[:]
                            best_sums = sums.copy()
                            if best_size >= 20:
                                break
                    else:
                        # Restore
                        s.append(x_to_remove)
                        s.sort()
                        sums = get_sums(s)
                
                if best_size >= 20:
                    break
        
        # Try single swaps
        if len(s) < 20 and time.time() < deadline - 3:
            s_curr = best_set[:] if best_set else required[:]
            s_curr.sort()
            sums_curr = get_sums(s_curr)
            
            for idx in range(len(s_curr)):
                if s_curr[idx] in required:
                    continue
                if time.time() > deadline - 3:
                    break
                
                x_to_remove = s_curr[idx]
                s_curr.remove(x_to_remove)
                sums_curr = get_sums(s_curr)
                
                for y in all_candidates:
                    if y == x_to_remove or y in s_curr:
                        continue
                    if is_valid_addition(s_curr, sums_curr, y):
                        s_curr.append(y)
                        s_curr.sort()
                        sums_curr = get_sums(s_curr)
                        if len(s_curr) > best_size:
                            best_size = len(s_curr)
                            best_set = s_curr[:]
                            best_sums = sums_curr.copy()
                        break
                # Restore
                s_curr.append(x_to_remove)
                s_curr.sort()
                sums_curr = get_sums(s_curr)
    
    if best_set is None:
        return required
    
    return best_set
