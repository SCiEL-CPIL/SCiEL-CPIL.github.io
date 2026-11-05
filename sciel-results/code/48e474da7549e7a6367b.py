import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 256
    
    def is_valid_set(S):
        """Check if S is valid: for all a<b<=c in S, a does not divide b+c."""
        S_sorted = sorted(S)
        m = len(S_sorted)
        for i in range(m):
            a = S_sorted[i]
            for j in range(i+1, m):
                b = S_sorted[j]
                for k in range(j, m):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def is_valid_addition(S, x):
        """Check if adding x to S keeps it valid."""
        S_sorted = sorted(S)
        m = len(S_sorted)
        # x must not already be in S
        # Check all triples involving x
        # Case 1: x is the smallest (a=x, b,c in S with b<=c, b>x)
        # Case 2: x is the middle (b=x, a in S with a<x, c in S with c>=x)
        # Case 3: x is the largest (c=x, a<b in S with a<b<=x)
        
        # Build lists
        less = [v for v in S_sorted if v < x]
        greater = [v for v in S_sorted if v > x]
        
        # Case 1: a=x, b,c from greater, b<=c
        for i in range(len(greater)):
            b = greater[i]
            for j in range(i, len(greater)):
                c = greater[j]
                if (b + c) % x == 0:
                    return False
        
        # Case 2: b=x, a from less, c from greater or c=x (but c>=b=x, so c=x means b=c=x, but a<b so a<x, valid)
        # Actually b=x, c>=b=x. c can be x itself (b=c=x) or c > x.
        # a < b = x, so a from less.
        for a in less:
            # c = x (b=c=x): check a | (x+x) = 2x
            if (2 * x) % a == 0:
                return False
            for c in greater:
                if (x + c) % a == 0:
                    return False
        
        # Case 3: c=x, a<b from less, a<b<=x. Since c=x and b<=c, b<=x.
        # a<b, both in less (since they're < x). 
        for i in range(len(less)):
            a = less[i]
            for j in range(i+1, len(less)):
                b = less[j]
                if (b + x) % a == 0:
                    return False
        
        return True
    
    def is_valid_removal(S, x):
        """Removing an element always keeps validity (removing can't create new conflicts)."""
        return True
    
    # Greedy construction: try to build a large set
    # Start with upper tail
    best_set = list(range(129, 257))
    best_size = len(best_set)
    
    # Check if upper tail is valid
    if not is_valid_set(best_set):
        # Remove elements to make it valid
        # Greedy removal: for each element, if it causes conflict, remove it
        S = set(range(129, 257))
        changed = True
        while changed:
            changed = False
            for x in sorted(S, reverse=True):
                # Check if x causes any conflict
                S_without_x = S - {x}
                # Check if x is the cause of any violation
                # For each pair (b,c) in S_without_x with b<=c, check if x | (b+c) and x < b
                # Or for each (a,c) with a<c, check if a | (b+c) where b=x
                # Or for each (a,b) with a<b, check if a | (b+c) where c=x
                conflict = False
                S_list = sorted(S_without_x)
                for i in range(len(S_list)):
                    a = S_list[i]
                    for j in range(i+1, len(S_list)):
                        b = S_list[j]
                        for k in range(j, len(S_list)):
                            c = S_list[k]
                            # Check if x participates
                            # This is getting complex. Let me just check full validity.
                            pass
                
                # Simpler: just check if removing x fixes things by checking if x is in a conflict
                # Actually, let me just do: try removing each element and see if the set becomes valid
                # But that's expensive. Let me use a different approach.
                
                # Check if x causes a conflict:
                # 1. x as 'a': exists b,c in S\{x} with x<b<=c and x|(b+c)
                # 2. x as 'b': exists a,c in S\{x} with a<x<=c and a|(x+c) [c>=x]
                # 3. x as 'c': exists a,b in S\{x} with a<b<=x and a|(b+x)
                
                # Case 1: x as a
                found = False
                for b in S_list:
                    if b <= x:
                        continue
                    for c in S_list:
                        if c < b:
                            continue
                        if (b + c) % x == 0:
                            found = True
                            break
                    if found:
                        break
                if found:
                    S.remove(x)
                    changed = True
                    continue
                
                # Case 2: x as b (a < x <= c)
                found = False
                for a in S_list:
                    if a >= x:
                        continue
                    # c >= x, c in S_list
                    for c in S_list:
                        if c < x:
                            continue
                        if (x + c) % a == 0:
                            found = True
                            break
                    if found:
                        break
                if found:
                    S.remove(x)
                    changed = True
                    continue
                
                # Case 3: x as c (a < b <= x)
                found = False
                for a in S_list:
                    for b in S_list:
                        if b <= a:
                            continue
                        if b > x:
                            continue
                        if (b + x) % a == 0:
                            found = True
                            break
                    if found:
                        break
                if found:
                    S.remove(x)
                    changed = True
                    continue
        
        best_set = list(S)
        best_size = len(best_set)
    
    # Now try to add elements from [1, 128]
    for x in range(1, 129):
        if time.time() > deadline - 5:
            break
        if is_valid_addition(best_set, x):
            best_set.append(x)
            best_size += 1
    
    # Local search: try to improve by swapping or adding
    # Try random restarts with greedy
    random.seed(42)
    
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        
        # Try different starting points
        start = random.randint(100, 160)
        S = list(range(start, 257))
        
        # Remove conflicts
        changed = True
        while changed:
            changed = False
            for x in sorted(S, reverse=True):
                S_set = set(S) - {x}
                S_list = sorted(S_set)
                found = False
                # Case 1: x as a
                for b in S_list:
                    if b <= x:
                        continue
                    for c in S_list:
                        if c < b:
                            continue
                        if (b + c) % x == 0:
                            found = True
                            break
                    if found:
                        break
                if found:
                    S.remove(x)
                    changed = True
                    continue
                # Case 2: x as b
                found = False
                for a in S_list:
                    if a >= x:
                        continue
                    for c in S_list:
                        if c < x:
                            continue
                        if (x + c) % a == 0:
                            found = True
                            break
                    if found:
                        break
                if found:
                    S.remove(x)
                    changed = True
                    continue
                # Case 3: x as c
                found = False
                for a in S_list:
                    for b in S_list:
                        if b <= a or b > x:
                            continue
                        if (b + x) % a == 0:
                            found = True
                            break
                    if found:
                        break
                if found:
                    S.remove(x)
                    changed = True
                    continue
        
        # Try to add smaller elements
        for x in range(1, start):
            if time.time() > deadline - 5:
                break
            if x not in S and is_valid_addition(S, x):
                S.append(x)
        
        if len(S) > best_size:
            best_set = S[:]
            best_size = len(S)
    
    # Final local search on best_set
    if time.time() < deadline - 5:
        S = best_set[:]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding each missing element
            for x in range(1, 257):
                if x not in S and is_valid_addition(S, x):
                    S.append(x)
                    improved = True
                    break
    
    return sorted(best_set)
