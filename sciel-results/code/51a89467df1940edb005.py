"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from math import gcd
from collections import defaultdict

P = {'n': 210, 'maximum_window': 10210, 'maximum_size': 100}

def construct():
    """
    Find 1..100 distinct integers strictly larger than 210 and at most 10210.
    Their product multiplied by 210 must be a perfect square.
    Minimize max(selected integers) - 210.
    """
    start_time = time.time()
    deadline = start_time + 85.0

    n = 210
    max_val = 10210
    max_size = 100

    # 210 = 2 * 3 * 5 * 7
    # We need product S to have square-free part exactly 210
    # i.e., parity vector of product must match parity vector of 210

    # Precompute square-free parts for all numbers in range
    def compute_sf_parts(lo, hi):
        """Compute square-free part for each number in [lo, hi]."""
        sf = {}
        for x in range(lo, hi + 1):
            temp = x
            result = 1
            d = 2
            while d * d <= temp:
                if temp % (d * d) == 0:
                    while temp % (d * d) == 0:
                        temp //= (d * d)
                elif temp % d == 0:
                    result *= d
                    temp //= d
                d += 1 if d == 2 else 2
            if temp > 1:
                result *= temp
            sf[x] = result
        return sf

    # Check if a set of numbers has the right property
    def is_valid_solution(nums):
        """Check if 210 * prod(nums) is a perfect square."""
        prod = 210
        for x in nums:
            prod *= x
        # Check perfect square using integer sqrt
        s = math.isqrt(prod)
        return s * s == prod

    # Strategy: Try increasing upper bounds, find minimal max
    # For each upper bound U, search for a valid set with all elements in (210, U]

    # Known: [840] works, giving max=840, raw=630
    best_solution = [840]
    best_max = 840

    # Try to find better solutions by searching for small sets
    # Key: use square-free part matching
    # sf(a*b) = sf(a) * sf(b) / gcd(sf(a),sf(b))^2 ... no
    # Actually: sf(a*b) in terms of prime parity: XOR of parity vectors
    
    # Let me use a different approach: for each number, compute its "square-free kernel"
    # Two numbers a, b: sf(a*b) has prime p with odd exponent iff (exp_p(a) + exp_p(b)) is odd
    # i.e., iff exp_p(a) mod 2 != exp_p(b) mod 2
    # So sf(a*b) = product of primes p where parity differs between a and b
    
    # For the product of a set S to have sf = 210 = 2*3*5*7:
    # The XOR of all parity vectors must equal the parity vector of 210

    # Let me represent each number by its parity vector (as a frozenset of primes with odd exponent)
    def parity_set(x):
        """Return frozenset of primes with odd exponent in x."""
        primes = set()
        temp = x
        d = 2
        while d * d <= temp:
            cnt = 0
            while temp % d == 0:
                cnt += 1
                temp //= d
            if cnt % 2 == 1:
                primes.add(d)
            d += 1 if d == 2 else 2
        if temp > 1:
            primes.add(temp)
        return frozenset(primes)

    # Parity set of 210 = {2, 3, 5, 7}
    target = frozenset({2, 3, 5, 7})

    # For a set S, the XOR of all parity sets must equal target
    # XOR of frozensets = symmetric difference

    # Let's search for solutions with small max
    # Try max = 211, 212, ... up to 840
    
    # For efficiency, precompute parity sets for numbers up to some limit
    # and use a meet-in-the-middle or greedy approach

    # Approach: For each upper bound U (from 211 upward), check if there exists
    # a subset of {211, ..., U} whose parity XOR equals target.
    
    # This is a subset XOR problem. For small U, we can enumerate.
    # But the space is huge. Let's be smarter.

    # Key observation: If we have two numbers with the same parity set, their XOR is empty set.
    # So we can pair them up and they cancel out.
    # We need an odd number of "contributions" to each prime in target.

    # Better approach: Try to find the minimum U such that there's a valid set.
    # Start from U = 211 and go up.

    # For a given U, the numbers available are 211..U.
    # We need a non-empty subset whose parity XOR = target.
    
    # Since we want to minimize max, we search U from 211 upward.
    # For each U, we can try:
    #   1. Single element: is there x in (210, U] with parity_set(x) = target?
    #      That means sf(x) = 210, i.e., x = 210*k^2. First such x > 210 is 840.
    #   2. Two elements: find a, b in (210, U] with parity_set(a) ^ parity_set(b) = target
    #      i.e., parity_set(b) = parity_set(a) ^ target
    #   3. Three elements: etc.

    # Let's try pairs first since they're efficient to search.

    # Precompute parity sets up to a reasonable limit
    limit = min(840, max_val)
    parity_map = {}
    for x in range(211, limit + 1):
        if time.time() > deadline - 5:
            break
        ps = parity_set(x)
        parity_map[x] = ps

    # For pairs: for each a, check if there exists b > a with parity_set(b) = parity_set(a) ^ target
    # Build a map from parity set to list of numbers
    ps_to_nums = defaultdict(list)
    for x in range(211, limit + 1):
        if x not in parity_map:
            continue
        ps = parity_map[x]
        ps_to_nums[ps].append(x)

    # Sort numbers in each group
    for ps in ps_to_nums:
        ps_to_nums[ps].sort()

    # For each a, compute needed_b_ps = parity_set(a) ^ target
    # Then look for b > a with that parity set
    found_pair = False
    best_pair_max = float('inf')
    
    for a in range(211, limit + 1):
        if a not in parity_map:
            continue
        if time.time() > deadline - 5:
            break
        pa = parity_map[a]
        needed = pa ^ target
        if needed in ps_to_nums:
            # Find smallest b > a in ps_to_nums[needed]
            nums = ps_to_nums[needed]
            # Binary search for first element > a
            lo, hi = 0, len(nums)
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] <= a:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < len(nums):
                b = nums[lo]
                if b > a and b <= limit:
                    cur_max = max(a, b)
                    if cur_max < best_pair_max:
                        best_pair_max = cur_max
                        found_pair = True
                        # Check validity
                        if is_valid_solution([a, b]):
                            if cur_max < best_max:
                                best_max = cur_max
                                best_solution = [a, b]

    # If we found a pair better than 840, use it
    # Otherwise try triples or keep [840]

    # Also check: maybe a triple with max < best_max exists
    # For triples: need a, b, c with pa ^ pb ^ pc = target
    # i.e., pc = pa ^ pb ^ target
    # This is more complex. Let's try if we have time.

    if time.time() < deadline - 10:
        # Try to find a triple with max < best_max
        triple_limit = min(best_max - 1, limit)
        if triple_limit > 212:
            # Build map for numbers up to triple_limit
            ps_to_nums_t = defaultdict(list)
            parity_map_t = {}
            for x in range(211, triple_limit + 1):
                if time.time() > deadline - 10:
                    break
                ps = parity_set(x)
                parity_map_t[x] = ps
                ps_to_nums_t[ps].append(x)
            
            # For each pair (a, b), compute needed = pa ^ pb ^ target
            # Look for c > b with that parity
            found_triple = False
            best_triple_max = float('inf')
            
            # To limit work, only try pairs where max(a,b) < best_max
            for a in range(211, triple_limit + 1):
                if a not in parity_map_t:
                    continue
                if time.time() > deadline - 10:
                    break
                pa = parity_map_t[a]
                for b in range(a + 1, triple_limit + 1):
                    if b not in parity_map_t:
                        continue
                    if time.time() > deadline - 10:
                        break
                    pb = parity_map_t[b]
                    needed = pa ^ pb ^ target
                    if needed in ps_to_nums_t:
                        nums = ps_to_nums_t[needed]
                        # Find c > b
                        lo, hi = 0, len(nums)
                        while lo < hi:
                            mid = (lo + hi) // 2
                            if nums[mid] <= b:
                                lo = mid + 1
                            else:
                                hi = mid
                        if lo < len(nums):
                            c = nums[lo]
                            if c > b and c <= triple_limit:
                                cur_max = max(a, b, c)
                                if cur_max < best_triple_max:
                                    best_triple_max = cur_max
                                    if is_valid_solution([a, b, c]):
                                        if cur_max < best_max:
                                            best_max = cur_max
                                            best_solution = [a, b, c]
                                            found_triple = True
                                        break  # Found a valid triple, move to next a
                    if time.time() > deadline - 10:
                        break

    # Final validation
    if is_valid_solution(best_solution):
        # Ensure all constraints met
        assert all(210 < x <= max_val for x in best_solution)
        assert len(set(best_solution)) == len(best_solution)
        assert 1 <= len(best_solution) <= max_size
        return [int(x) for x in best_solution]
    else:
        # Fallback to known good solution
        return [840]

# EVOLVE-BLOCK-END
