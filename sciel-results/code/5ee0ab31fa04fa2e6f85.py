"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    """
    Find 1..100 distinct integers strictly larger than 210 and at most 10210.
    Their product multiplied by 210 must be a perfect square.
    Minimize max(selected integers) - 210.
    """
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 210
    max_val = 10210
    max_size = 100
    
    # 210 = 2 * 3 * 5 * 7
    # For single number: x = 210 * t^2, x > 210 means t >= 2, x = 840, raw = 630
    best_solution = [840]
    best_max = 840
    
    def is_valid_solution(nums):
        """Check if 210 * prod(nums) is a perfect square."""
        prod = 210
        for x in nums:
            prod *= x
        s = math.isqrt(prod)
        return s * s == prod
    
    def get_divisors(x):
        """Get all divisors of x."""
        divs = set()
        i = 1
        while i * i <= x:
            if x % i == 0:
                divs.add(i)
                divs.add(x // i)
            i += 1
        return sorted(divs)
    
    # Phase 1: Search for pairs using a * b = 210 * t^2
    # For each t, compute 210*t^2, find all factor pairs (a,b) with a*b = 210*t^2
    # both in (210, best_max)
    
    # t can be at most sqrt(max_val / 210) for single number, but for pairs
    # a * b = 210 * t^2, with a, b > 210
    # So t^2 > 210 * 210 / 210 = 210, t > sqrt(210) ~ 14.5
    # Actually a * b = 210 * t^2, a > 210, b > 210
    # So 210 * t^2 > 210 * 210 = 44100, t^2 > 210, t >= 15
    # Also a, b <= best_max - 1 to improve, so 210 * t^2 <= (best_max-1)^2
    # t^2 <= (best_max-1)^2 / 210
    
    t_max = int(math.isqrt((best_max - 1) ** 2 / n)) + 1
    t_max = min(t_max, 200)  # safety cap
    
    found_better = False
    for t in range(1, t_max + 1):
        if time.time() > deadline - 5:
            break
        prod = n * t * t
        # Find factor pairs (a, b) with a * b = prod, a > 210, b > 210, a != b
        # a <= b to avoid duplicates
        a_max = int(math.isqrt(prod))
        for a in range(211, a_max + 1):
            if prod % a != 0:
                continue
            b = prod // a
            if b <= a:
                continue
            if b > best_max - 1:
                continue
            if b > max_val:
                continue
            # a and b are valid
            cur_max = b  # since b > a
            if cur_max < best_max:
                best_max = cur_max
                best_solution = [a, b]
                found_better = True
    
    # Phase 2: If no pair found, try triples
    # For triples: a * b * c = 210 * t^2
    # This is harder. Let's try a different approach.
    # Use parity/square-free part approach
    
    if not found_better and time.time() < deadline - 10:
        # Use square-free part approach for triples
        # sf(x) = product of primes with odd exponent in x
        # Need sf(a) * sf(b) * sf(c) = 210 (as square-free parts)
        
        def sf_part(x):
            """Return the square-free part of x."""
            result = 1
            temp = x
            d = 2
            while d * d <= temp:
                cnt = 0
                while temp % d == 0:
                    cnt += 1
                    temp //= d
                if cnt % 2 == 1:
                    result *= d
                d += 1 if d == 2 else 2
            if temp > 1:
                result *= temp
            return result
        
        # Precompute sf for numbers up to best_max - 1
        limit = best_max - 1
        sf_map = {}
        sf_to_nums = {}
        for x in range(211, limit + 1):
            if time.time() > deadline - 10:
                break
            sf = sf_part(x)
            sf_map[x] = sf
            if sf not in sf_to_nums:
                sf_to_nums[sf] = []
            sf_to_nums[sf].append(x)
        
        # For triples: sf(a) * sf(b) * sf(c) should be 210
        # i.e., sf(a) * sf(b) * sf(c) = 210
        # So sf(c) = 210 / (sf(a) * sf(b))
        # sf(a) * sf(b) must divide 210
        
        # 210 = 2 * 3 * 5 * 7
        # Divisors of 210: 1, 2, 3, 5, 6, 7, 10, 14, 15, 21, 30, 35, 42, 70, 105, 210
        divs_210 = [1, 2, 3, 5, 6, 7, 10, 14, 15, 21, 30, 35, 42, 70, 105, 210]
        
        for a in range(211, limit + 1):
            if a not in sf_map:
                continue
            if time.time() > deadline - 10:
                break
            sa = sf_map[a]
            for b in range(a + 1, limit + 1):
                if b not in sf_map:
                    continue
                if time.time() > deadline - 10:
                    break
                sb = sf_map[b]
                prod_sf = sa * sb
                if prod_sf > 210 or 210 % prod_sf != 0:
                    continue
                sc_needed = 210 // prod_sf
                if sc_needed in sf_to_nums:
                    # Find c > b with sf = sc_needed
                    for c in sf_to_nums[sc_needed]:
                        if c > b and c <= limit:
                            cur_max = c
                            if cur_max < best_max:
                                best_max = cur_max
                                best_solution = [a, b, c]
                                break
                    if best_max <= b:
                        break
    
    # Final validation
    if is_valid_solution(best_solution):
        assert all(210 < x <= max_val for x in best_solution)
        assert len(set(best_solution)) == len(best_solution)
        assert 1 <= len(best_solution) <= max_size
        return [int(x) for x in best_solution]
    else:
        return [840]

# EVOLVE-BLOCK-END
