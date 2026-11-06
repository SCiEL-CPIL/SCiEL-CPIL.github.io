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
            # We need a+b=n, a,b in smooths, a < b, a not divide b, b not divide a
            # Since a+b=n, a < n/2
            candidates = [x for x in smooths if x < n]
            for i in range(len(candidates)):
                a = candidates[i]
                b = n - a
                if b <= a: 
                    continue
                if b not in smooths_set:
                    continue
                # Check antichain: neither divides the other
                if b % a == 0:
                    continue
                # a % b is always non-zero if a < b and b > 0, unless a=0 which is not possible
                # So just check b % a != 0 is sufficient for distinct a<b
                found = [a, b]
                break
            
            if found is None:
                # Try k=3
                # a+b+c=n, a<b<c, all in smooths, antichain
                candidates = [x for x in smooths if x < n]
                for i in range(len(candidates)):
                    a = candidates[i]
                    for j in range(i+1, len(candidates)):
                        b = candidates[j]
                        c = n - a - b
                        if c <= b:
                            continue
                        if c not in smooths_set:
                            continue
                        # Check antichain for (a,b,c)
                        # Pairs: (a,b), (a,c), (b,c)
                        if b % a == 0:
                            continue
                        if c % a == 0:
                            continue
                        if c % b == 0:
                            continue
                        found = [a, b, c]
                        break
                    if found:
                        break
            
            if found is None:
                # Try k=4 (just in case, though likely k<=3 is always possible for 1..128)
                candidates = [x for x in smooths if x < n]
                for i in range(len(candidates)):
                    a = candidates[i]
                    for j in range(i+1, len(candidates)):
                        b = candidates[j]
                        for k in range(j+1, len(candidates)):
                            c = candidates[k]
                            d = n - a - b - c
                            if d <= c:
                                continue
                            if d not in smooths_set:
                                continue
                            # Check antichain for (a,b,c,d)
                            # This is getting complex to check all pairs manually in loop, 
                            # but we can write a helper or just check the 6 pairs.
                            if b % a == 0 or c % a == 0 or d % a == 0:
                                continue
                            if c % b == 0 or d % b == 0:
                                continue
                            if d % c == 0:
                                continue
                            found = [a, b, c, d]
                            break
                        if found:
                            break
                    if found:
                        break
        
        if found is None:
            # Fallback: should not happen for 1..128 with smooth numbers
            # But if it does, we need a valid representation. 
            # The problem states "All targets require valid representations".
            # If k=4 fails, try higher k? 
            # Given the density of smooth numbers, k=3 or 4 should always work.
            # For safety, if not found, we could raise an error or try a greedy approach, 
            # but the precomputation should find it.
            # Let's assume k<=4 is sufficient. If not, we might need to expand search.
            # However, 1 is smooth, 2 is smooth... 
            # Most numbers can be formed by 1, 2, 3... but they must be antichain.
            # 1 divides everything, so if 1 is in the set, no other number can be.
            # So 1 can only be used if k=1 and n=1.
            # If n=1, found=[1].
            # If n=2, found=[2].
            # If n=3, found=[3].
            # If n=4, found=[4].
            # If n=5, found=[5].
            # If n=6, 6 is smooth.
            # If n=7, 7 not smooth. 1+6 (1|6 no), 2+5 (2|? no, 5|? no). 2,5 are smooth. 2 does not divide 5. So [2,5] works.
            # It is highly likely k<=3 is always sufficient.
            pass
        
        solutions.append(found)
        
    return solutions

_PRECOMPUTED = _precompute_solutions()

def construct():
    return _PRECOMPUTED
