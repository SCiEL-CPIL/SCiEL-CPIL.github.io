import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    n = 210
    max_val = 10210

    # 210 = 2 * 3 * 5 * 7
    # We need a set S of distinct integers in (210, 10210] such that
    # 210 * prod(S) is a perfect square.
    # This is equivalent to: the square-free part of prod(S) is 210.
    # Or: the parity vector of prod(S) (over all primes) equals the parity vector of 210.
    # Parity vector of 210: {2:1, 3:1, 5:1, 7:1} (all other primes: 0)

    # Strategy:
    # 1. Try 1 element: x such that 210*x is square => x = 210*k^2. Smallest > 210 is 840 (k=2).
    # 2. Try 2 elements: x*y = 210*k^2. Minimize max(x,y).
    # 3. Try 3 elements: x*y*z has square-free part 210. Use parity-set grouping.
    # 4. Try more elements if needed.

    best_answer = None
    best_raw = float('inf')

    def update_best(ans, raw):
        nonlocal best_answer, best_raw
        if raw < best_raw:
            # Verify
            prod = 210
            for x in ans:
                prod *= x
            r = math.isqrt(prod)
            if r * r == prod:
                if all(isinstance(x, int) and 210 < x <= max_val for x in ans):
                    if len(set(ans)) == len(ans):
                        best_answer = [int(x) for x in ans]
                        best_raw = raw

    # 1. Single element
    # x = 210 * k^2 > 210 => k >= 2
    for k in range(2, 100):
        x = 210 * k * k
        if x > max_val:
            break
        update_best([x], x - 210)

    # 2. Two elements
    # x * y = 210 * k^2, x != y, x, y > 210, x, y <= max_val
    # Minimize max(x, y)
    for k in range(1, 200):
        if time.time() > deadline - 30:
            break
        val = 210 * k * k
        if val > max_val * max_val:
            break
        sqrt_val = math.isqrt(val)
        # x must be > 210 and y = val/x must be > 210 and <= max_val
        # x <= y => x <= sqrt_val
        # y = val/x <= max_val => x >= val/max_val
        # x > 210
        lo = max(211, math.ceil(val / max_val))
        hi = min(sqrt_val, max_val)
        for x in range(lo, hi + 1):
            if val % x == 0:
                y = val // x
                if y > 210 and y <= max_val and x != y:
                    m = max(x, y)
                    if m - 210 < best_raw:
                        update_best([x, y], m - 210)

    # 3. Three elements using parity-set approach
    # Compute parity set for each number in [211, M]
    # Parity set: frozenset of primes p where exponent of p in n is odd
    
    def prime_parity_set(n):
        s = set()
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    s.add(p)
            p += 1
        if temp > 1:
            s.add(temp)
        return frozenset(s)

    target_parity = frozenset({2, 3, 5, 7})

    # We need x*y*z to have parity set = target_parity
    # parity(x) XOR parity(y) XOR parity(z) = target_parity
    # So parity(x) XOR parity(y) = target_parity XOR parity(z)
    
    # Try increasing M until we find a solution or run out of time
    # For each M, build pairs and group by parity set
    
    for M in range(250, 500):
        if time.time() > deadline - 5:
            break
        
        # Precompute parity sets for [211, M]
        parity_cache = {}
        for i in range(211, M + 1):
            parity_cache[i] = prime_parity_set(i)
        
        # Build pairs grouped by parity set
        # For each pair (x, y) with x < y, compute px = parity(x) XOR parity(y)
        # Group by px
        pairs_by_parity = defaultdict(list)
        
        for x in range(211, M + 1):
            px = parity_cache[x]
            for y in range(x + 1, M + 1):
                py = parity_cache[y]
                # XOR of parity sets: symmetric difference
                pxy = px.symmetric_difference(py)
                m_xy = max(x, y)
                pairs_by_parity[pxy].append((m_xy, x, y))
        
        # For each z, find required pxy = target_parity XOR parity(z)
        found = False
        for z in range(211, M + 1):
            if time.time() > deadline - 3:
                break
            pz = parity_cache[z]
            required_pxy = target_parity.symmetric_difference(pz)
            
            if required_pxy in pairs_by_parity:
                for m_xy, x, y in pairs_by_parity[required_pxy]:
                    if x != z and y != z:
                        m = max(m_xy, z)
                        if m - 210 < best_raw:
                            update_best([x, y, z], m - 210)
                            found = True
                            break
            if found:
                break
        
        # If we found a good solution, we can stop early
        if best_raw <= M - 210:
            # We found a solution with max <= M, no need to check larger M for 3 elements
            # But maybe 4 elements could be better? Unlikely for small M.
            break

    # 4. If still no good solution, try 4 elements
    # This is more complex. Let's try a simple approach:
    # Find 4 numbers whose product has parity set = target_parity.
    # We can pair them: (x1,x2) and (x3,x4) such that parity(x1x2) XOR parity(x3x4) = target
    # Or: find a pair (a,b) with parity pa, and another pair (c,d) with parity pb,
    # such that pa XOR pb = target.
    
    if time.time() < deadline - 10 and best_raw > 400:
        # Try 4 elements
        # Precompute parity for [211, 400]
        M4 = 400
        parity_cache4 = {}
        for i in range(211, M4 + 1):
            parity_cache4[i] = prime_parity_set(i)
        
        # Group pairs by parity
        pairs_by_parity4 = defaultdict(list)
        for x in range(211, M4 + 1):
            px = parity_cache4[x]
            for y in range(x + 1, M4 + 1):
                py = parity_cache4[y]
                pxy = px.symmetric_difference(py)
                m_xy = max(x, y)
                pairs_by_parity4[pxy].append((m_xy, x, y))
        
        # For each pair (a,b), find pair (c,d) with parity = target XOR parity(ab)
        # and all four distinct
        found4 = False
        for pxy1, pair_list1 in pairs_by_parity4.items():
            if time.time() > deadline - 5:
                break
            required_pxy2 = target_parity.symmetric_difference(pxy1)
            if required_pxy2 in pairs_by_parity4:
                for m1, a, b in pair_list1:
                    if time.time() > deadline - 5:
                        break
                    for m2, c, d in pairs_by_parity4[required_pxy2]:
                        if a != c and a != d and b != c and b != d:
                            m = max(m1, m2)
                            if m - 210 < best_raw:
                                update_best([a, b, c, d], m - 210)
                                found4 = True
                                break
                    if found4:
                        break
            if found4:
                break

    # Return best answer
    if best_answer:
        return best_answer
    
    # Fallback
    return [840]
