import itertools

def _precompute_solutions():
    # Generate all 5-smooth numbers up to 128
    smooths = []
    for i in range(1, 129):
        x = i
        while x % 2 == 0:
            x //= 2
        while x % 3 == 0:
            x //= 3
        while x % 5 == 0:
            x //= 5
        if x == 1:
            smooths.append(i)
    
    smooths_set = set(smooths)
    solutions = []
    
    for n in range(1, 129):
        found = None
        
        # Try k=1
        if n in smooths_set:
            found = [n]
        else:
            # Try k=2
            # We need a+b=n, a,b in smooths, a < b, a not divide b
            # Iterate a from smallest to largest
            for a in smooths:
                if a >= n:
                    break
                b = n - a
                if b <= a:
                    continue
                if b not in smooths_set:
                    continue
                # Check antichain: b % a != 0
                if b % a != 0:
                    found = [a, b]
                    break
            
            if found is None:
                # Try k=3
                # a+b+c=n, a<b<c, all in smooths, antichain
                # a < n/3
                for i in range(len(smooths)):
                    a = smooths[i]
                    if a * 3 >= n: # a must be < n/3
                        break
                    for j in range(i+1, len(smooths)):
                        b = smooths[j]
                        if a + b >= n:
                            break
                        c = n - a - b
                        if c <= b:
                            continue
                        if c not in smooths_set:
                            continue
                        # Check antichain
                        if b % a == 0 or c % a == 0 or c % b == 0:
                            continue
                        found = [a, b, c]
                        break
                    if found:
                        break
            
            if found is None:
                # Try k=4 (fallback)
                # a+b+c+d=n
                # This is less likely needed
                pass
        
        if found is None:
             # Fallback: use a known valid representation if search failed
             # But for 1..128, k<=3 should always work
             # If not, we raise an error to catch bugs
             raise ValueError(f"No solution found for {n}")
        
        solutions.append(found)
        
    return solutions

_PRECOMPUTED = _precompute_solutions()

def construct():
    return _PRECOMPUTED
