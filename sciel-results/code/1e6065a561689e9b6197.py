import sys
import time

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s for exit overhead
    
    # Target: cover diffs 1..30
    # Bit i-1 corresponds to diff i
    target_mask = (1 << 30) - 1
    
    best_max = 10**6
    best_set = None
    
    def update_best(current, max_val):
        nonlocal best_max, best_set
        if max_val < best_max:
            best_max = max_val
            best_set = list(current)
            
    def popcount(x):
        return bin(x).count('1')

    def dfs(current, used_diffs, covered):
        nonlocal best_max, best_set, deadline
        
        if time.time() > deadline:
            return
        
        if covered == target_mask:
            update_best(current, current[-1])
            return
            
        if len(current) >= 32:
            return
            
        # Pruning: Can we cover the rest?
        remaining_needed = 30 - popcount(covered)
        remaining_slots = 32 - len(current)
        
        # Max diffs we can add with remaining_slots elements
        max_possible_new = 0
        for i in range(remaining_slots):
            max_possible_new += len(current) + i
            
        if remaining_needed > max_possible_new:
            return
            
        current_max = current[-1]
        
        # Generate candidates
        candidates = set()
        
        missing_diffs = []
        for d in range(1, 31):
            if not (covered & (1 << (d - 1))):
                missing_diffs.append(d)
                
        # If no missing diffs, we are done (handled above)
        
        for y in current:
            for d in missing_diffs:
                x = y + d
                if x > current_max and x < best_max:
                    candidates.add(x)
                    
        if not candidates:
            return
            
        # Sort candidates to try smaller ones first
        for x in sorted(candidates):
            if x >= best_max:
                continue
                
            # Check Sidon property
            new_used = set(used_diffs)
            valid = True
            for y in current:
                diff = x - y
                if diff in new_used:
                    valid = False
                    break
                new_used.add(diff)
                
            if not valid:
                continue
                
            # Update covered mask
            new_covered = covered
            for y in current:
                diff = x - y
                if 1 <= diff <= 30:
                    new_covered |= (1 << (diff - 1))
                    
            # Recurse
            dfs(current + (x,), new_used, new_covered)
            
    # Start DFS
    # We can seed with a greedy solution to get a good initial best_max
    # Greedy:
    a = [0]
    used = set()
    cov = 0
    for d in range(1, 31):
        if cov & (1 << (d-1)):
            continue
        found = False
        # Try to realize d by adding x
        # x - y = d => x = y + d
        for y in a:
            x = y + d
            if x in a: 
                continue
            # Check conflicts
            ok = True
            new_used = set(used)
            for z in a:
                diff = abs(x-z)
                if diff in new_used:
                    ok = False
                    break
                new_used.add(diff)
            if ok:
                a.append(x)
                used = new_used
                for z in a:
                    diff = abs(x-z)
                    if 1 <= diff <= 30:
                        cov |= (1 << (diff-1))
                a.sort()
                found = True
                break
        if not found:
            # Fallback: add large number
            x = 2*max(a) + 3*d + 1
            while True:
                new_used = set(used)
                ok = True
                for z in a:
                    diff = abs(x-z)
                    if diff in new_used:
                        ok = False
                        break
                    new_used.add(diff)
                if ok:
                    a.append(x)
                    used = new_used
                    for z in a:
                        diff = abs(x-z)
                        if 1 <= diff <= 30:
                            cov |= (1 << (diff-1))
                    a.sort()
                    break
                x += 1
    update_best(tuple(a), a[-1])
    
    # Now run DFS to improve
    dfs((0,), set(), 0)
    
    if best_set is None:
        return [0]
    return best_set
