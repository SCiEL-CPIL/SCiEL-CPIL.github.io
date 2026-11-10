"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    # Try to find minimal number of distinct denominators in [5, 10000000]
    # whose reciprocals sum to exactly 1.
    
    # Theoretical minimum is 3, but with min denominator 5, check if 3 works.
    # 1/a + 1/b + 1/c = 1 with 5 <= a < b < c
    # For 3 terms: 1/a + 1/b + 1/c = 1 => (ab - a - c)(c) ... 
    # Known: minimum with d>=5 is 4 terms.
    
    # Search for 4-term solutions: 1/a + 1/b + 1/c + 1/d = 1, 5 <= a < b < c < d <= 10^7
    # Use: 1/a + 1/b = 1 - 1/c - 1/d = (c*d - c - d)/(c*d)
    # So a*b = c*d*(c*d - c - d) / (c*d - c - d) ... 
    # Better: iterate a, b, solve for c, d.
    # 1/c + 1/d = 1 - 1/a - 1/b = (a*b - a - b)/(a*b)
    # Let N = a*b - a - b, D = a*b. Need N > 0, i.e., a*b > a + b, i.e., (a-1)(b-1) > 1.
    # 1/c + 1/d = N/D => D*(c+d) = N*c*d => N*c*d - D*c - D*d = 0
    # => (N*c - D)(N*d - D) = D^2
    # So we need factor pairs of D^2 such that N*c - D and N*d - D are both positive divisors.
    
    for a in range(5, 50):
        if time.time() > deadline:
            break
        for b in range(a + 1, 200):
            if time.time() > deadline:
                break
            N = a * b - a - b
            D = a * b
            if N <= 0:
                continue
            # 1/c + 1/d = N/D, need c > b, d > c
            # (N*c - D)(N*d - D) = D^2
            # Let u = N*c - D, v = N*d - D, u*v = D^2, u > 0, v > 0
            # c = (u + D) / N, d = (v + D) / N
            # Need (u+D) % N == 0 and (v+D) % N == 0
            # Also c > b => (u+D)/N > b => u+D > b*N
            # And d > c => v > u (since d > c)
            D2 = D * D
            # u must be a divisor of D^2 with u < D (so that v > u, i.e., d > c)
            # Actually u*v = D^2, and we want c < d, so u < v, so u < D.
            # Iterate u over divisors of D^2 less than D.
            # For efficiency, iterate u up to D-1 and check if D^2 % u == 0.
            # But D can be up to ~40000, D^2 up to ~1.6e9, iterating is too slow.
            # Instead, factor D and generate divisors.
            
            # Quick check: is there a solution?
            # For small a, b, this should be fast.
            # Let's just try a few promising (a,b) pairs.
            pass
    
    # Known 4-term solution: 1/5 + 1/7 + 1/9 + 1/... let me just verify known results.
    # Actually, the minimal is 4: e.g., 1/5 + 1/6 + 1/8 + 1/... no.
    # Let me just hardcode a verified 4-term solution if I can find one, or use the DP.
    
    # Use the DP approach but with a better common denominator.
    # The sum of reciprocals of 5..n is about ln(n) - 2.4. For n=100, sum ~ 1.35 - 2.4 < 1.
    # For n=50, sum ~ 3.9 - 2.4 = 1.5 > 1. So a solution exists within [5, 50].
    
    # DP over divisors of a common denominator. Use LCM of 5..40 or similar.
    # But LCM is huge. Instead, use Fraction for exact arithmetic.
    
    # Meet-in-the-middle: split into two sets, find sums that add to 1.
    # For k terms, split into k/2 and k/2.
    
    # For 4 terms: split into 2+2.
    # Precompute all pairs (i,j), 5<=i<j<=50, with their sum s = 1/i + 1/j.
    # Then look for two pairs with disjoint indices whose sums add to 1.
    
    max_den = 50
    pairs = {}  # sum -> list of (i, j)
    for i in range(5, max_den + 1):
        for j in range(i + 1, max_den + 1):
            s = Fraction(1, i) + Fraction(1, j)
            if s >= 1:
                continue
            if s not in pairs:
                pairs[s] = []
            pairs[s].append((i, j))
    
    # Look for two pairs (i,j) and (k,l) with i<j<k<l and s1 + s2 = 1
    found = None
    for s1, list1 in pairs.items():
        s2 = 1 - s1
        if s2 not in pairs:
            continue
        list2 = pairs[s2]
        for (i, j) in list1:
            for (k, l) in list2:
                if i < j < k < l:
                    found = [i, j, k, l]
                    break
            if found:
                break
        if found:
            break
    
    if found:
        return found
    
    # If 4-term not found quickly, fall back to DP with Fraction
    # But let's try a direct 4-term search with slightly larger range
    for a in range(5, 20):
        if time.time() > deadline:
            break
        for b in range(a + 1, 40):
            N = a * b - a - b
            D = a * b
            if N <= 0:
                continue
            # Need 1/c + 1/d = N/D with c > b
            # (N*c - D)(N*d - D) = D^2
            # u = N*c - D, u must divide D^2, u > 0
            # c = (u + D) // N, must be integer and > b
            # d = (D*D // u + D) // N, must be integer and > c
            # Iterate u over divisors of D^2
            D2 = D * D
            # Find divisors of D2
            divs = []
            limit = int(D2 ** 0.5)
            for u in range(1, limit + 1):
                if D2 % u == 0:
                    divs.append(u)
                    if u != D2 // u:
                        divs.append(D2 // u)
            for u in divs:
                if u >= D:  # want u < D so that c < d
                    continue
                v = D2 // u
                c_num = u + D
                d_num = v + D
                if c_num % N != 0 or d_num % N != 0:
                    continue
                c = c_num // N
                d = d_num // N
                if c > b and d > c and d <= 10000000:
                    return [a, b, c, d]
    
    # Fall back to DP
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
