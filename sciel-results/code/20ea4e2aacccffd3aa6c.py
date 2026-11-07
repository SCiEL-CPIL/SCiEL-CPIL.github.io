"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

# EVOLVE-BLOCK-END

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin
    
    def sf(n):
        """Compute squarefree kernel of n."""
        res = 1
        i = 2
        while i * i <= n:
            cnt = 0
            while n % i == 0:
                n //= i
                cnt += 1
            if cnt % 2 == 1:
                res *= i
            i += 1
        if n > 1:
            res *= n
        return res
    
    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n
    
    target_sf = 210  # sf(210) = 210
    
    best_result = [840]  # 210 * 1^2 = 210, but need > 210, so 210*4=840
    best_max = 840 - 210  # 630
    
    # Search for solutions with smaller max value
    # Try increasing max values from 211 upward
    # For each max value M, try to find a subset of {211, ..., M} with sf product = 210
    
    # First, precompute sf for a range
    max_search = 10210
    sf_cache = {}
    
    def get_sf(n):
        if n not in sf_cache:
            sf_cache[n] = sf(n)
        return sf_cache[n]
    
    # Strategy: for each candidate max M from 211 upward, check if there's a subset
    # of {211..M} whose sf product has sf = 210
    # Start with small subsets (size 1, 2, 3, ...)
    
    # Size 1: n where sf(n) == 210
    # n = 210 * k^2, n > 210, so k >= 2 (k=1 gives 210 which is not > 210)
    # k=2: 210*4 = 840
    # k=3: 210*9 = 1890
    # etc.
    
    # Size 2: a, b where sf(a*b) == 210, i.e., sf(sf(a)*sf(b)) == 210
    # Size 3: a, b, c where sf(sf(a)*sf(b)*sf(c)) == 210
    
    # Let's search systematically by increasing max value
    # For efficiency, precompute sf for numbers up to some limit
    
    # Precompute sf for 211 to 10210
    candidates = list(range(211, 10211))
    sfs = [get_sf(n) for n in candidates]
    
    # For size 1
    for i, n in enumerate(candidates):
        if sfs[i] == target_sf:
            if n - 210 < best_max:
                best_max = n - 210
                best_result = [n]
            break  # first one is smallest
    
    # For size 2: find pairs (a,b) with a < b, sf(a*b) == 210
    # sf(a*b) = sf(sf(a) * sf(b))
    # We want the smallest max(a,b)
    
    # Use a set/dict approach: for each b, check if there exists a < b with sf(sf(a)*sf(b)) == 210
    # Precompute: for each possible sf value, which numbers have it
    
    # Let's try a different approach: iterate over max value M from 211 to 10210
    # and for each M, check if M can be part of a valid subset with all elements <= M
    
    # Actually, let's just do a direct search for small subsets
    # Size 2:
    # For each pair, check sf(product) == 210
    # To be efficient, group by sf value
    
    # Map from sf value to list of (index, value)
    from collections import defaultdict
    sf_to_indices = defaultdict(list)
    for i, n in enumerate(candidates):
        sf_to_indices[sfs[i]].append((i, n))
    
    # For size 2: for each b in candidates, for each a < b, check sf(a*b) == 210
    # This is O(n^2) which is ~10000^2 = 100M, too slow
    # Instead: for each sf_a, sf_b such that sf(sf_a * sf_b) == 210, find pairs
    
    # Let's enumerate possible sf_a, sf_b combinations
    # sf_a and sf_b are squarefree numbers
    # sf(sf_a * sf_b) == 210 means the product sf_a * sf_b, when made squarefree, gives 210
    
    # Alternative: just iterate over pairs but limit the search
    # Since we want to minimize max, let's try increasing max values
    
    # For a given max M, the candidates are 211..M
    # Check size 1, 2, 3, ... subsets
    
    # Let's try a smarter approach:
    # For size 2: a*b must have sf = 210
    # a*b = 210 * k^2 for some integer k
    # So a*b = 210, 210*4, 210*9, ... = 210, 840, 1890, 3780, 6570, ...
    # a and b are both > 210 and <= 10210
    
    # For each product P = 210 * k^2, find factor pairs (a,b) with 210 < a < b <= 10210
    # and a*b = P
    
    # k=1: P=210, min factor > 210 impossible since a*b=210 and a>210 means b<1
    # k=2: P=840, factors: 210*4, 120*7, etc. a>210: a=210? No, a>210. 
    #   840 = 210*4, 168*5, 120*7, 84*10, 70*12, 60*14, 48*17.5 no...
    #   840 = 2^3 * 3 * 5 * 7
    #   Factor pairs with both > 210: 210*4=840, but 210 is not > 210. 
    #   Next: is there a factor > 210 with its complement also > 210? 
    #   840/211 ≈ 3.98, so if a=211, b=840/211 not integer.
    #   Actually for a*b=840 with a,b>210: a >= 211, b = 840/a <= 840/211 ≈ 3.98 < 210. Impossible.
    
    # k=3: P=1890, a,b>210: a>=211, b=1890/a <= 1890/211 ≈ 8.96 < 210. Impossible.
    
    # k=4: P=3360, a>=211, b<=3360/211≈15.9 < 210. Impossible.
    
    # k=5: P=5250, a>=211, b<=5250/211≈24.9 < 210. Impossible.
    
    # k=6: P=7560, a>=211, b<=7560/211≈35.8 < 210. Impossible.
    
    # k=7: P=10290, a>=211, b<=10290/211≈48.8 < 210. Impossible.
    
    # k=8: P=13440, a>=211, b<=13440/211≈63.7 < 210. Impossible.
    
    # k=9: P=17010, a>=211, b<=17010/211≈80.6 < 210. Impossible.
    
    # k=10: P=21000, a>=211, b<=21000/211≈99.5 < 210. Impossible.
    
    # k=11: P=25410, a>=211, b<=25410/211≈120.4 < 210. Impossible.
    
    # k=12: P=30240, a>=211, b<=30240/211≈143.3 < 210. Impossible.
    
    # k=13: P=35490, a>=211, b<=35490/211≈168.2 < 210. Impossible.
    
    # k=14: P=41160, a>=211, b<=41160/211≈195.1 < 210. Impossible.
    
    # k=15: P=47250, a>=211, b<=47250/211≈224.0. So b can be up to 224.
    #   Need a*b=47250, a>210, b>210, a<=b.
    #   a in (210, 224], b=47250/a in [224, 225)
    #   Check: 47250/211≈223.9, 47250/224≈210.9
    #   So a in [211,224], b=47250/a must be integer and >210
    #   47250 = 2 * 3^3 * 5^3 * 7
    #   Let's check: 47250/225 = 210, but 225>224. 47250/224 = 210.9375, not integer.
    #   47250/222 = 212.8..., 47250/220 = 214.77...
    #   Let me just check divisibility.
    
    # This is getting complex. Let me just write code to search.
    
    # Direct search approach with time limit
    # Precompute sf for all candidates
    # Then search for subsets of size 1, 2, 3, ... with increasing max
    
    # For efficiency, let's search by increasing max value M
    # For each M, check if any subset of {211..M} works
    
    # Size 1: already found 840
    # Size 2: search for pairs
    
    # Let's do a more targeted search
    # For size 2: a*b = 210 * k^2
    # a > 210, b > 210, a <= b
    # max(a,b) = b, minimize b
    # b >= a > 210, so b >= 211
    # a*b = 210*k^2, so b = 210*k^2/a
    # For b to be integer, a must divide 210*k^2
    # And b > 210, so 210*k^2/a > 210, i.e., k^2/a > 1, i.e., a < k^2
    # Also a <= b means a <= 210*k^2/a, i.e., a^2 <= 210*k^2, i.e., a <= k*sqrt(210)
    
    # Let's just iterate over k and find valid (a,b) pairs
    # b = 210*k^2/a, need a | 210*k^2, a > 210, b > 210, a <= b
    
    # Minimize b = 210*k^2/a, so maximize a (subject to a <= b and a | 210*k^2)
    # a <= b = 210*k^2/a means a^2 <= 210*k^2 means a <= k*sqrt(210) ≈ 14.49*k
    
    # So a is in (210, 14.49*k] and a divides 210*k^2
    
    # Let's search over k
    best_pair = None
    best_pair_max = 10**9
    
    for k in range(1, 200):
        P = 210 * k * k
        if P > 10210 * 10210:
            break
        # Find divisors a of P with 210 < a <= P/211 (so that b=P/a > 210)
        # and a <= sqrt(P) (so a <= b)
        a_max = min(int(math.isqrt(P)), P // 211)
        # a must be > 210
        a_min = 211
        if a_max < a_min:
            continue
        # Find divisors of P in [a_min, a_max]
        # P = 210 * k^2
        # Iterate over possible a values
        # To be efficient, iterate over divisors
        # But P can be large, so let's just check a range
        # Actually, let's factor P and enumerate divisors
        # Or just check if any a in range divides P
        # Range size: a_max - a_min, which can be large
        
        # Better: iterate over divisors of P
        # P = 2 * 3 * 5 * 7 * k^2
        # Let's just check a few candidates
        
        # Actually, let's just iterate a from a_min to a_max and check P % a == 0
        # But this could be slow if the range is large
        
        # Alternative: for each prime factor, build divisors
        # Let's just do a simple loop for now, with time check
        if time.time() > deadline - 5:
            break
        
        # Check a values that divide P
        # a must divide P = 210 * k^2
        # Let's iterate over possible a
        for a in range(a_min, a_max + 1):
            if P % a == 0:
                b = P // a
                if b > 210 and b <= 10210 and a <= b:
                    if b < best_pair_max:
                        best_pair_max = b
                        best_pair = (a, b)
    
    if best_pair and best_pair_max - 210 < best_max:
        best_max = best_pair_max - 210
        best_result = [best_pair[0], best_pair[1]]
    
    # Size 3: a*b*c = 210 * k^2
    # This is more complex. Let's skip for now and return best found.
    
    # Verify the result
    if best_result:
        prod = 1
        for x in best_result:
            prod *= x
        if not is_perfect_square(210 * prod):
            # Fallback
            best_result = [840]
            best_max = 630
    
    return list(best_result)
