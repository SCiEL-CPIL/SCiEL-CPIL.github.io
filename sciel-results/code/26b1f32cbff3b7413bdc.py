# EVOLVE-BLOCK-START
import time

def construct():
    n = 256
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Base set: [171, 256] is valid because for any a < b <= c in [171, 256],
    # b+c >= 2*172 = 344 and a <= 255.
    # If a | (b+c), then b+c >= 2a (since b>a, c>=b>a).
    # Max b+c = 256+255 = 511.
    # Min a = 171. 2*171 = 342.
    # So b+c could be 2a.
    # We verified [171, 256] is valid.
    
    current_set = set(range(171, 257))
    candidates = list(range(1, 171))
    
    def is_valid_insert(x, s):
        """
        Check if adding x to set s maintains the property.
        Only checks triples involving x.
        """
        # x as a: x < b <= c, x | (b+c)
        for b in s:
            if b <= x:
                continue
            for c in s:
                if c >= b and (b + c) % x == 0:
                    return False
        
        # x as b: a < x <= c, a | (x+c)
        for a in s:
            if a >= x:
                continue
            for c in s:
                if c >= x and (x + c) % a == 0:
                    return False
        
        # x as c: a < b <= x, a | (b+x)
        for a in s:
            if a >= x:
                continue
            for b in s:
                if b <= x and a < b and (b + x) % a == 0:
                    return False
        
        # x as b and c (b=c=x): a < x, a | (2x)
        for a in s:
            if a < x and (2 * x) % a == 0:
                return False
        
        return True

    # Greedy addition
    # We want to add elements that cause the fewest conflicts.
    # To speed up sorting, we can just try adding all candidates in a specific order 
    # and re-evaluate. Or we can do a few passes of "add if valid".
    
    # Order candidates: 
    # Smaller x are more likely to divide b+c (as a). 
    # Larger x (close to 171) are less likely to divide b+c.
    # However, if x is small, it's harder for a|x to hold? No, a<x.
    # Let's try adding from 170 down to 1 first, as these are "closest" to the valid region 
    # and might have fewer conflicts with the large elements.
    
    # Pass 1: Try adding from 170 down to 1
    for x in range(170, 0, -1):
        if time.time() > deadline:
            break
        if x in current_set:
            continue
        if is_valid_insert(x, current_set):
            current_set.add(x)
            candidates.remove(x)
            
    # Pass 2: Try adding from 1 up to 170 (in case some small ones work now)
    # This might be slow if many candidates remain, but is_valid_insert is O(N^2) ~ 256^2 = 65k ops.
    # 170 candidates * 65k = 11M ops. Should be fast enough in Python? 
    # Python is slow. 11M simple operations might take ~1-2 seconds.
    # Let's optimize by skipping candidates that are definitely invalid?
    # Or just limit the number of checks.
    
    for x in range(1, 171):
        if time.time() > deadline:
            break
        if x in current_set:
            continue
        if is_valid_insert(x, current_set):
            current_set.add(x)
            candidates.remove(x)
            
    # Pass 3: Try random order for remaining candidates
    import random
    random.seed(42)
    remaining = [x for x in range(1, 171) if x not in current_set]
    random.shuffle(remaining)
    for x in remaining:
        if time.time() > deadline:
            break
        if is_valid_insert(x, current_set):
            current_set.add(x)
            # Remove from remaining? Not strictly necessary if we check membership
            
    # Final check
    # Verify the whole set to be safe (O(N^3) but N is at most 256, 256^3/6 ~ 2.8M triples. 
    # 2.8M checks in Python might take ~10-20s. 
    # We have ~85s budget.
    # If N=256, N^3/6 is 2.8M. 
    # If N=100, N^3/6 is 1.6M.
    # Let's do a faster verification or skip if time is tight.
    
    def is_valid_set_full(s):
        elems = sorted(s)
        L = len(elems)
        for i in range(L):
            a = elems[i]
            for j in range(i + 1, L):
                b = elems[j]
                for k in range(j, L):
                    c = elems[k]
                    if (b + c) % a == 0:
                        return False
        return True

    if time.time() < deadline - 10:
        if not is_valid_set_full(current_set):
            # Fallback to base set
            return list(range(171, 257))
            
    return sorted(current_set)
# EVOLVE-BLOCK-END
