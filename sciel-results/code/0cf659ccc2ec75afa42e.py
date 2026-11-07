import math
import time
from collections import defaultdict

def construct():
    deadline = time.time() + 85  # 85s safety margin
    
    def sf_kernel(n):
        """Return squarefree kernel of n."""
        if n <= 1:
            return 1
        result = 1
        f = n
        d = 2
        while d * d <= f:
            if f % d == 0:
                exp = 0
                while f % d == 0:
                    f //= d
                    exp += 1
                if exp % 2 == 1:
                    result *= d
            d += 1 if d == 2 else 2
        if f > 1:
            result *= f
        return result
    
    def is_perfect_square(n):
        if n < 0:
            return False
        x = math.isqrt(n)
        return x * x == n
    
    # Precompute squarefree kernels up to 10210
    max_val = 10210
    sf = [0] * (max_val + 1)
    sf[1] = 1
    for i in range(2, max_val + 1):
        sf[i] = sf_kernel(i)
    
    # Build map from sf value to sorted list of numbers with that sf
    sf_to_nums = defaultdict(list)
    for n in range(211, max_val + 1):
        sf_to_nums[sf[n]].append(n)
    
    # Also build sf_to_min for quick lookup
    sf_to_min = {}
    for s, nums in sf_to_nums.items():
        sf_to_min[s] = nums[0]
    
    # Precompute prime factor sets for each squarefree kernel value
    # Map: frozenset of primes -> squarefree kernel value
    # And: squarefree kernel value -> frozenset of primes
    sf_to_primes = {}
    primes_to_sf = {}
    for s in sf_to_nums.keys():
        n = s
        primes = frozenset()
        d = 2
        tmp = s
        while d * d <= tmp:
            if tmp % d == 0:
                primes = primes | {d}
                while tmp % d == 0:
                    tmp //= d
            d += 1 if d == 2 else 2
        if tmp > 1:
            primes = primes | {tmp}
        sf_to_primes[s] = primes
        primes_to_sf[primes] = s
    
    # Target: squarefree kernel of 210 = 210
    target = 210
    target_primes = sf_to_primes[target]
    
    best_max = max_val + 1
    best_list = None
    
    # Size 1: find n in [211, 10210] with sf[n] == 210
    if target in sf_to_min:
        n = sf_to_min[target]
        best_max = n
        best_list = [n]
    
    # Size 2: need sf[a] and sf[b] such that sf[a] XOR sf[b] (in prime set sense) = target
    # i.e., sf_to_primes[sf[a]] ^ sf_to_primes[sf[b]] = target_primes
    # So sf_to_primes[sf[b]] = target_primes ^ sf_to_primes[sf[a]]
    
    # Build list of (a, sf[a], primes_of_sf[a]) for a in [211, max_val]
    # For size 2, iterate over a, compute required primes for b, look up
    # Sort by a to prioritize smaller values
    
    for a in range(211, max_val + 1):
        if a >= best_max:
            break
        sa = sf[a]
        pa = sf_to_primes[sa]
        required_pb = target_primes ^ pa
        if required_pb in primes_to_sf:
            sb = primes_to_sf[required_pb]
            if sb in sf_to_min:
                b = sf_to_min[sb]
                if b != a and b < best_max:
                    m = max(a, b)
                    if m < best_max:
                        best_max = m
                        best_list = [a, b]
    
    # Size 3: For pair (a,b), need c such that target_primes ^ pa ^ pb ^ pc = empty set
    # So pc = target_primes ^ pa ^ pb
    # Look up if pc corresponds to a valid sf value with a number c
    
    # Precompute all valid (sf_value, min_num) pairs for quick lookup
    valid_sf_min = {s: sf_to_min[s] for s in sf_to_min if s in primes_to_sf}
    
    search_limit = min(best_max, 6000)
    
    for a in range(211, min(search_limit, 4000)):
        if time.time() > deadline - 5:
            break
        sa = sf[a]
        pa = sf_to_primes[sa]
        for b in range(a + 1, min(search_limit, 4000)):
            if time.time() > deadline - 5:
                break
            sb = sf[b]
            pb = sf_to_primes[sb]
            required_pc = target_primes ^ pa ^ pb
            if required_pc in primes_to_sf:
                sc = primes_to_sf[required_pc]
                if sc in valid_sf_min:
                    c = valid_sf_min[sc]
                    if c != a and c != b and c < best_max:
                        m = max(a, b, c)
                        if m < best_max:
                            best_max = m
                            best_list = [a, b, c]
    
    # Size 4: For triple (a,b,c), need d such that target_primes ^ pa ^ pb ^ pc ^ pd = empty
    # pd = target_primes ^ pa ^ pb ^ pc
    # Use meet-in-the-middle: precompute all pair XORs, then look up
    
    if time.time() < deadline - 20 and best_max > 220:
        # Build all valid pairs (a,b) with a < b and their XOR
        # Then for each such pair, look up if there's another pair with XOR = target ^ pair_xor
        pair_limit = min(best_max, 1500)
        pair_xor_map = defaultdict(list)  # xor_value -> list of (a, b)
        
        for a in range(211, pair_limit):
            if time.time() > deadline - 10:
                break
            sa = sf[a]
            pa = sf_to_primes[sa]
            for b in range(a + 1, pair_limit):
                if time.time() > deadline - 10:
                    break
                sb = sf[b]
                pb = sf_to_primes[sb]
                xor_ab = pa ^ pb
                pair_xor_map[xor_ab].append((a, b))
        
        # Now for each pair (a,b), we need another pair (c,d) such that:
        # target ^ pa ^ pb ^ pc ^ pd = empty
        # => pc ^ pd = target ^ pa ^ pb
        # => pc ^ pd = target ^ xor_ab
        # So look up pairs with XOR = target ^ xor_ab
        
        for xor_ab, pairs in pair_xor_map.items():
            if time.time() > deadline - 10:
                break
            required_xor = target_primes ^ xor_ab
            if required_xor in pair_xor_map:
                for (a, b) in pairs:
                    for (c, d) in pair_xor_map[required_xor]:
                        # Ensure all four are distinct
                        if len({a, b, c, d}) == 4:
                            m = max(a, b, c, d)
                            if m < best_max:
                                best_max = m
                                best_list = [a, b, c, d]
                                break
                    if time.time() > deadline - 10:
                        break
    
    # Size 5: For quadruple (a,b,c,d), need e such that target ^ pa ^ pb ^ pc ^ pd ^ pe = empty
    # pe = target ^ pa ^ pb ^ pc ^ pd
    # Try small range
    
    if time.time() < deadline - 15 and best_max > 220:
        small_limit = min(best_max, 800)
        for a in range(211, small_limit):
            if time.time() > deadline - 10:
                break
            sa = sf[a]
            pa = sf_to_primes[sa]
            for b in range(a + 1, small_limit):
                if time.time() > deadline - 10:
                    break
                sb = sf[b]
                pb = sf_to_primes[sb]
                for c in range(b + 1, small_limit):
                    if time.time() > deadline - 10:
                        break
                    sc = sf[c]
                    pc = sf_to_primes[sc]
                    for d in range(c + 1, small_limit):
                        if time.time() > deadline - 10:
                            break
                        sd = sf[d]
                        pd = sf_to_primes[sd]
                        required_pe = target_primes ^ pa ^ pb ^ pc ^ pd
                        if required_pe in primes_to_sf:
                            se = primes_to_sf[required_pe]
                            if se in valid_sf_min:
                                e = valid_sf_min[se]
                                if e != a and e != b and e != c and e != d and e < best_max:
                                    m = max(a, b, c, d, e)
                                    if m < best_max:
                                        best_max = m
                                        best_list = [a, b, c, d, e]
    
    if best_list is None:
        best_list = [840]
        best_max = 840
    
    # Verify the result
    prod = 210
    for x in best_list:
        prod *= x
    if not is_perfect_square(prod):
        # Fallback to known valid solution
        best_list = [840]
        best_max = 840
    
    # Ensure all values are Python ints
    result = [int(x) for x in best_list]
    return result
