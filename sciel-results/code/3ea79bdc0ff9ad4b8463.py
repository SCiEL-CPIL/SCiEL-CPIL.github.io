import time
import random

def construct():
    deadline = time.time() + 85  # 85s safety margin
    
    N = 256
    
    def has_4ap(s, x):
        """Check if adding x to set s creates a 4-AP."""
        # A 4-AP containing x: x can be at position 0,1,2,3
        # x = a, a+d, a+2d, a+3d
        # Case 1: x is first term: x, x+d, x+2d, x+3d all in s∪{x}
        #   => need x+d, x+2d, x+3d in s
        # Case 2: x is second term: x-d, x, x+d, x+2d
        #   => need x-d, x+d, x+2d in s
        # Case 3: x is third term: x-2d, x-d, x, x+d
        #   => need x-2d, x-d, x+d in s
        # Case 4: x is fourth term: x-3d, x-2d, x-d, x
        #   => need x-3d, x-2d, x-d in s
        for d in range(1, N+1):
            # Case 1: x, x+d, x+2d, x+3d
            if x + 3*d <= N:
                if (x+d) in s and (x+2*d) in s and (x+3*d) in s:
                    return True
            # Case 2: x-d, x, x+d, x+2d
            if x - d >= 1 and x + 2*d <= N:
                if (x-d) in s and (x+d) in s and (x+2*d) in s:
                    return True
            # Case 3: x-2d, x-d, x, x+d
            if x - 2*d >= 1 and x + d <= N:
                if (x-2*d) in s and (x-d) in s and (x+d) in s:
                    return True
            # Case 4: x-3d, x-2d, x-d, x
            if x - 3*d >= 1:
                if (x-3*d) in s and (x-2*d) in s and (x-d) in s:
                    return True
        return False
    
    def count_3aps(s):
        """Count distinct 3-APs in set s."""
        count = 0
        for a in s:
            for d in range(1, N+1):
                if a + 2*d <= N and (a+d) in s and (a+2*d) in s:
                    count += 1
        return count
    
    def delta_3aps_on_add(s, x):
        """Count how many new 3-APs are created by adding x to s."""
        count = 0
        # x can be first, second, or third term of a 3-AP
        # x, x+d, x+2d
        for d in range(1, N+1):
            if x + 2*d <= N:
                if (x+d) in s and (x+2*d) in s:
                    count += 1
        # x-d, x, x+d
        for d in range(1, N+1):
            if x - d >= 1 and x + d <= N:
                if (x-d) in s and (x+d) in s:
                    count += 1
        # x-2d, x-d, x
        for d in range(1, N+1):
            if x - 2*d >= 1:
                if (x-2*d) in s and (x-d) in s:
                    count += 1
        return count
    
    def delta_3aps_on_remove(s, x):
        """Count how many 3-APs are lost by removing x from s."""
        count = 0
        # x, x+d, x+2d
        for d in range(1, N+1):
            if x + 2*d <= N:
                if (x+d) in s and (x+2*d) in s:
                    count += 1
        # x-d, x, x+d
        for d in range(1, N+1):
            if x - d >= 1 and x + d <= N:
                if (x-d) in s and (x+d) in s:
                    count += 1
        # x-2d, x-d, x
        for d in range(1, N+1):
            if x - 2*d >= 1:
                if (x-2*d) in s and (x-d) in s:
                    count += 1
        return count
    
    # Greedy initial construction
    s = set()
    for x in range(1, N+1):
        if not has_4ap(s, x):
            s.add(x)
    
    best_s = set(s)
    best_count = count_3aps(best_s)
    
    # Local search
    random.seed(42)
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        improved = False
        
        # Try adding elements
        candidates = [x for x in range(1, N+1) if x not in s]
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline:
                break
            if not has_4ap(s, x):
                new_count = best_count + delta_3aps_on_add(s, x)
                s.add(x)
                best_count = new_count
                best_s = set(s)
                improved = True
                no_improve = 0
                break
        
        if improved:
            continue
        
        # Try swaps
        in_set = list(s)
        not_in_set = [x for x in range(1, N+1) if x not in s]
        random.shuffle(in_set)
        random.shuffle(not_in_set)
        
        for x in in_set:
            if time.time() > deadline:
                break
            for y in not_in_set:
                if time.time() > deadline:
                    break
                # Try swapping x for y
                s.remove(x)
                if not has_4ap(s, y):
                    new_count = best_count - delta_3aps_on_remove(s, x) + delta_3aps_on_add(s, y)
                    if new_count > best_count:
                        s.add(y)
                        best_count = new_count
                        best_s = set(s)
                        improved = True
                        no_improve = 0
                        break
                s.add(x)
            if improved:
                break
        
        if improved:
            continue
        
        no_improve += 1
        if no_improve >= max_no_improve:
            break
    
    return sorted(best_s)
