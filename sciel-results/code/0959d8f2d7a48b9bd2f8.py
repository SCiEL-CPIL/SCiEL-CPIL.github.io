import time
from math import isqrt, gcd
from collections import defaultdict

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _sieve_sfp(limit):
    """Precompute squarefree part for all numbers 1..limit using a sieve."""
    spf = list(range(limit + 1))
    for i in range(2, int(limit ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    sfp = [0] * (limit + 1)
    sfp[1] = 1
    for x in range(2, limit + 1):
        v = x
        r = 1
        while v > 1:
            p = spf[v]
            cnt = 0
            while v % p == 0:
                v //= p
                cnt ^= 1
            if cnt:
                r *= p
        sfp[x] = r
    return sfp

def construct():
    t0 = time.time()
    deadline = t0 + 50.0
    LO = 210
    MAXV = 10210
    
    # Precompute squarefree parts
    sfp = _sieve_sfp(MAXV)
    
    # Target: sfp(product) = sfp(210) = 210
    target_sfp = 210
    
    # Check single elements
    for x in range(LO + 1, MAXV + 1):
        if sfp[x] == target_sfp:
            if _is_square(LO * x):
                return [x]
        if time.time() > deadline:
            break
            
    # Check pairs
    # We need sfp(x) * sfp(y) to have sfp equal to 210.
    # This implies sfp(x) and sfp(y) must be factors of 210 * square.
    # Actually, sfp(a*b) = a*b // gcd(a,b)^2 for squarefree a,b.
    # We iterate through possible sfp values.
    
    # Optimization: Directly check known good pairs or search efficiently.
    # Build index of smallest number for each sfp value
    min_num_for_sfp = {}
    for x in range(LO + 1, MAXV + 1):
        s = sfp[x]
        if s not in min_num_for_sfp:
            min_num_for_sfp[s] = x
        # Store second smallest if needed? 
        # For pairs (a,b) with a!=b, we just need min.
        # For pairs (a,a), we need two distinct numbers.
        
    # To handle a==b case properly, let's store the two smallest numbers for each sfp
    two_smallest_for_sfp = {}
    for x in range(LO + 1, MAXV + 1):
        s = sfp[x]
        if s not in two_smallest_for_sfp:
            two_smallest_for_sfp[s] = [x, None]
        else:
            if two_smallest_for_sfp[s][1] is None:
                two_smallest_for_sfp[s][1] = x
            elif x < two_smallest_for_sfp[s][1]:
                two_smallest_for_sfp[s][1] = x
            # Keep sorted
            if two_smallest_for_sfp[s][0] > two_smallest_for_sfp[s][1]:
                two_smallest_for_sfp[s].reverse()

    best_max = MAXV
    best_pair = None
    
    # Iterate through divisors of 210 to find potential sfp pairs
    # If sfp(x)=a and sfp(y)=b, then a*b must be 210 * k^2.
    # Since a,b are squarefree, a*b is squarefree if gcd(a,b)=1.
    # If gcd(a,b)=g, then sfp(a*b) = a*b/g^2.
    # We need a*b/g^2 = 210.
    
    # Simpler: Iterate through all sfp values present in the range.
    sfp_values = list(two_smallest_for_sfp.keys())
    
    for i in range(len(sfp_values)):
        a = sfp_values[i]
        for j in range(i, len(sfp_values)):
            b = sfp_values[j]
            g = gcd(a, b)
            # sfp(a*b) = (a*b) // (g*g)
            prod_sfp = (a * b) // (g * g)
            if prod_sfp == target_sfp:
                if a == b:
                    if two_smallest_for_sfp[a][1] is not None:
                        x = two_smallest_for_sfp[a][0]
                        y = two_smallest_for_sfp[a][1]
                        if _is_square(LO * x * y):
                            mx = max(x, y)
                            if mx < best_max:
                                best_max = mx
                                best_pair = [x, y]
                else:
                    x = two_smallest_for_sfp[a][0]
                    y = two_smallest_for_sfp[b][0]
                    if _is_square(LO * x * y):
                        mx = max(x, y)
                        if mx < best_max:
                            best_max = mx
                            best_pair = [x, y]
        if time.time() > deadline:
            break

    if best_pair:
        return best_pair
        
    # Fallback
    return [840]
