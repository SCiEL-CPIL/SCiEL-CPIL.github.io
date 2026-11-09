import time

# EVOLVE-BLOCK-START
P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    n = P['n']
    
    # Precompute prime factorizations for all numbers 2..n
    def prime_factors(num):
        factors = {}
        d = 2
        while d * d <= num:
            while num % d == 0:
                factors[d] = factors.get(d, 0) + 1
                num //= d
            d += 1
        if num > 1:
            factors[num] = factors.get(num, 0) + 1
        return factors
    
    candidates = list(range(2, n + 1))
    factorizations = {num: prime_factors(num) for num in candidates}
    
    def divides_product(a, b, c):
        """Check if a divides b*c using prime factorizations."""
        fa = factorizations[a]
        fb = factorizations[b]
        fc = factorizations[c]
        for p, ea in fa.items():
            eb = fb.get(p, 0)
            ec = fc.get(p, 0)
            if ea > eb + ec:
                return False
        return True
    
    # Find all primes (valid base set)
    primes = [num for num in candidates if len(factorizations[num]) == 1 and 2 <= num and factorizations[num].get(num, 0) == 1]
    # Actually, simpler: a prime has exactly one prime factor with exponent 1
    primes = [num for num in candidates if len(factorizations[num]) == 1 and list(factorizations[num].values())[0] == 1]
    
    S = set(primes)
    composites = [num for num in candidates if num not in S]
    
    # Compute conflict score for each composite: number of pairs (b,c) in S where x|b*c or b|x*c or c|x*b
    # But S will grow, so let's use a simpler heuristic: count how many pairs in the full candidate set conflict
    # Better: use the current set S to estimate
    
    # For ordering: count for each composite x, how many pairs (b,c) in current S have any of:
    # x|b*c, b|x*c, c|x*b
    # This is expensive, so use a proxy: count how many pairs in ALL candidates conflict
    # Even simpler: just try composites in increasing order and check
    
    # Actually, let's compute a "difficulty" score: for composite x, count how many pairs (b,c) in [2..n]
    # with b<c, b!=x, c!=x have x|b*c. Lower = safer.
    
    # Precompute this for all composites
    conflict_score = {}
    for x in composites:
        cnt = 0
        fx = factorizations[x]
        for b in candidates:
            if b == x:
                continue
            fb = factorizations[b]
            for c in candidates:
                if c == x or c == b:
                    continue
                fc = factorizations[c]
                # Check if x | b*c
                valid = True
                for p, ex in fx.items():
                    eb = fb.get(p, 0)
                    ec = fc.get(p, 0)
                    if ex > eb + ec:
                        valid = False
                        break
                if valid:
                    cnt += 1
        conflict_score[x] = cnt
    
    # Sort composites by conflict score (ascending)
    sorted_composites = sorted(composites, key=lambda x: conflict_score[x])
    
    # Greedily add composites
    for x in sorted_composites:
        if time.time() > deadline - 5:
            break
        
        # Check if x can be added to S
        valid = True
        S_list = list(S)
        nS = len(S_list)
        
        # Case 1: x | b*c for some b,c in S
        for i in range(nS):
            for j in range(i + 1, nS):
                if divides_product(x, S_list[i], S_list[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Case 2: for each pair (a,c) in S, check a | x*c and a | c*x (same as a|x*c)
            # And x | a*c already covered. Need: a | x*c for all distinct a,c in S
            for i in range(nS):
                a = S_list[i]
                for j in range(nS):
                    if i == j:
                        continue
                    c = S_list[j]
                    if divides_product(a, x, c):
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            S.add(x)
    
    result = sorted(S)
    
    # Verify the result
    n = len(result)
    for i in range(n):
        a = result[i]
        for j in range(i + 1, n):
            b = result[j]
            for k in range(j + 1, n):
                c = result[k]
                if divides_product(a, b, c):
                    # Invalid! Fallback to primes only
                    return sorted(primes)
    
    return result
# EVOLVE-BLOCK-END
