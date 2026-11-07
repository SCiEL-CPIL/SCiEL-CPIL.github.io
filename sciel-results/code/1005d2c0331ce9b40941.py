import math
import time
from functools import lru_cache

def _sqfree_kernel(n):
    """Return the squarefree kernel (product of primes with odd exponent) of n."""
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            cnt = 0
            while temp % d == 0:
                temp //= d
                cnt += 1
            if cnt % 2 == 1:
                result *= d
        d += 1
    if temp > 1:
        result *= temp
    return result

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    best_val = float('inf')
    best_list = None

    # Precompute squarefree kernels for all numbers in range
    # We'll search for the smallest M such that there exist distinct ints in [211, M]
    # whose product * 210 is a perfect square.

    # Strategy: for increasing M from 211 upward, check if we can find a subset
    # of {211, ..., M} whose product has squarefree kernel 210.
    #
    # Equivalent: product of selected numbers must have squarefree part 210.
    # Since 210 is squarefree, the XOR (mod 2) of the prime-exponent parity vectors
    # of selected numbers must equal the vector for 210 = {2:1, 3:1, 5:1, 7:1}.
    #
    # But the numbers can be large and have many prime factors. Instead, we can
    # think in terms of the squarefree kernel directly: the product of kernels
    # of selected numbers, reduced to squarefree, must equal 210.
    #
    # A simpler approach: just try small sets (1, 2, 3, 4 elements) and for each
    # candidate max M, check if a valid subset exists with all elements <= M.

    # For k=1: need x such that 210*x is a perfect square, x > 210.
    # 210*x = s^2 => x = s^2/210. Since 210 is squarefree, x = 210*t^2.
    # x > 210 => t >= 2, x = 840.
    x1 = 840
    if x1 <= 10210:
        best_val = x1 - 210
        best_list = [x1]

    # For k=2: need a*b such that 210*a*b is a square, 211 <= a < b <= M.
    # 210*a*b = s^2 => a*b = 210*t^2 (since 210 is squarefree, a*b must be 210 times a square)
    # Actually: 210*a*b must be a square. Let a*b = P. Then 210*P = square.
    # Since 210 is squarefree, P must be 210 * (square). So a*b = 210*t^2.
    # We want to minimize max(a,b), so minimize b (with a < b).
    
    # Precompute: for each t, 210*t^2 = a*b with 211 <= a < b.
    # To minimize b, we want a as large as possible (close to sqrt(210*t^2)) but a < b.
    
    # Let's search t from 1 upward, and for each, find the factorization a*b = 210*t^2
    # with a < b, a >= 211, b <= 10210, minimizing b.
    
    for t in range(1, 300):
        if time.time() > deadline:
            break
        prod = 210 * t * t
        if prod > 10210 * 10210:
            break
        # Find a in [211, sqrt(prod)] such that a divides prod and b = prod/a > a
        a_max = min(math.isqrt(prod), 10210)
        for a in range(a_max, 210, -1):
            if prod % a == 0:
                b = prod // a
                if a < b and b <= 10210:
                    val = b - 210
                    if val < best_val:
                        best_val = val
                        best_list = [a, b]
                    break  # for this t, we found the best (largest a, smallest b)
    
    # For k=3: need a*b*c such that 210*a*b*c is a square.
    # 210*a*b*c = s^2 => a*b*c = 210*t^2.
    # We want to minimize max(a,b,c). Let's try to find triples with small max.
    # 
    # Approach: iterate c from 211 upward (as the max), and for each c,
    # find a < b < c such that a*b*c = 210*t^2 for some t.
    # Equivalently, a*b = 210*t^2 / c, so c must divide 210*t^2.
    
    # Alternative: iterate over possible products P = a*b*c where P = 210*t^2,
    # and factor P into three distinct factors all in [211, M].
    
    # Let's try: for c from 211 to some limit, check if there exist a, b with
    # 211 <= a < b < c and a*b*c = 210*t^2.
    # This means a*b = 210*t^2/c, so c | 210*t^2.
    
    # For efficiency, let's precompute which c values can work.
    # c must divide 210*t^2 for some t. Since 210 = 2*3*5*7, 
    # c can have prime factors only from those that appear in 210*t^2.
    # Actually t can be any integer, so c can be any integer that divides 210*t^2
    # for some t. This means c's squarefree part must divide 210.
    
    # So c must have squarefree kernel dividing 210.
    # Let's precompute valid c values.
    
    # Actually, let's just try small c values and check.
    # For c in [211, ...], we need a*b = 210*t^2/c to be an integer with
    # 211 <= a < b < c.
    
    # Let's iterate c from 211 upward, and for each, find t such that
    # 210*t^2 is divisible by c, then check if 210*t^2/c can be factored
    # into a*b with 211 <= a < b < c.
    
    # To be efficient, for a given c, t must be such that c | 210*t^2.
    # Let g = gcd(c, 210). Then c/g must divide t^2.
    # Let c' = c // g. We need c' | t^2, so c' must be a perfect square times... 
    # Actually, we need t^2 to be divisible by c'. This means c' must be a square
    # times a divisor... No, t^2 is always a perfect square, so c' must divide
    # some perfect square. This means the squarefree part of c' must be 1, i.e.,
    # c' must be a perfect square. Wait no - t^2 can be any perfect square, so
    # c' must divide t^2, which means c' must be of the form where each prime
    # exponent in c' is <= the exponent in t^2. Since t^2 has even exponents,
    # we need c' to have the property that for each prime p, v_p(c') <= v_p(t^2) = 2*v_p(t).
    # This is always satisfiable: just take t to be a multiple of the square root
    # of the "odd part" of c'. Specifically, let c' = s * u^2 where s is squarefree.
    # Then we need s | t^2, which requires s = 1 (since s is squarefree and t^2
    # has even exponents). Wait, that's not right either.
    
    # Let me reconsider. We need c | 210 * t^2. Let c = 2^a * 3^b * 5^c * 7^d * r
    # where r is coprime to 210. Then 210*t^2 = 2^1 * 3^1 * 5^1 * 7^1 * t^2.
    # For c | 210*t^2, we need:
    #   a <= 1 + 2*v2(t), b <= 1 + 2*v3(t), c_exp <= 1 + 2*v5(t), d <= 1 + 2*v7(t)
    #   and r | t^2 (since r is coprime to 210).
    # For r | t^2, r must be a perfect square (since r is coprime to 210, and
    # t^2 has even exponents for all primes not in 210).
    # So the squarefree part of r must be 1, meaning r must be a perfect square.
    # And for the 2,3,5,7 parts, we can always choose t to satisfy the constraints.
    
    # So the condition is: the squarefree part of c (after removing factors of 2,3,5,7
    # with appropriate exponents) must be a perfect square... This is getting complex.
    
    # Simpler: just try c from 211 upward, and for each c, try small t values
    # and check if c divides 210*t^2.
    
    # But this could be slow. Let's limit the search.
    
    # Actually, let's try a different approach for k=3:
    # We want a*b*c = 210*t^2 with 211 <= a < b < c, minimizing c.
    # For a given c, we need a*b = 210*t^2/c. Let's denote N = 210*t^2/c.
    # We need N to be an integer, and N = a*b with 211 <= a < b < c.
    # So N must be >= 211*212 and < c*c (since a < c and b < c, a*b < c^2).
    # Also N must be factorable into two distinct factors both >= 211 and < c.
    
    # Let's iterate c from 211, and for each, find the smallest t such that
    # c | 210*t^2, then check if N = 210*t^2/c can be factored appropriately.
    
    # To find t such that c | 210*t^2:
    # Let's precompute for each c the minimal t.
    # c | 210*t^2 iff for each prime p: v_p(c) <= v_p(210) + 2*v_p(t).
    # So v_p(t) >= ceil((v_p(c) - v_p(210)) / 2) if v_p(c) > v_p(210), else 0.
    # t_min = product over p of p^max(0, ceil((v_p(c) - v_p(210))/2)).
    
    def _min_t_for_c(c):
        """Find the minimal t such that c divides 210*t^2."""
        temp = c
        t = 1
        # Factor out 2, 3, 5, 7 first
        for p, exp_210 in [(2,1), (3,1), (5,1), (7,1)]:
            cnt = 0
            while temp % p == 0:
                temp //= p
                cnt += 1
            if cnt > exp_210:
                # Need 2*v_p(t) >= cnt - exp_210
                needed = cnt - exp_210
                v_t = (needed + 1) // 2  # ceil(needed/2)
                t *= p ** v_t
        # Remaining temp is coprime to 210
        if temp > 1:
            # temp must be a perfect square for t to exist (since t^2 must cover temp)
            # Actually, we need temp | t^2, and gcd(temp, 210) = 1.
            # Factor temp: for each prime p in temp with exponent e, we need 2*v_p(t) >= e,
            # so v_p(t) >= ceil(e/2).
            d = 2
            temp2 = temp
            while d * d <= temp2:
                if temp2 % d == 0:
                    cnt = 0
                    while temp2 % d == 0:
                        temp2 //= d
                        cnt += 1
                    v_t = (cnt + 1) // 2
                    t *= d ** v_t
                d += 1
            if temp2 > 1:
                t *= temp2  # exponent 1, need v_t >= 1
        return t
    
    # Now iterate c from 211 upward
    for c in range(211, 10211):
        if time.time() > deadline:
            break
        if c - 210 >= best_val:
            break
        
        t_min = _min_t_for_c(c)
        if t_min > 500:  # too large, unlikely to help
            continue
        
        # Try t from t_min upward
        for t in range(t_min, t_min + 50):
            num = 210 * t * t
            if num % c != 0:
                continue
            N = num // c
            # Need N = a*b with 211 <= a < b < c
            if N < 211 * 212:
                continue
            if N >= c * c:
                break
            a_max = min(math.isqrt(N), c - 1)
            if a_max < 211:
                continue
            for a in range(211, a_max + 1):
                if N % a == 0:
                    b = N // a
                    if a < b < c:
                        val = c - 210
                        if val < best_val:
                            best_val = val
                            best_list = [a, b, c]
                        break
            else:
                continue
            break  # found a valid pair for this c
    
    # For k=4: similar approach but more complex. Let's try if we have time.
    if time.time() < deadline - 20:
        # For k=4: a*b*c*d = 210*t^2, 211 <= a < b < c < d, minimize d.
        # This is harder. Let's try a limited search.
        # For d from 214 upward, check if a*b*c = 210*t^2/d for some t.
        pass
    
    if best_list is None:
        return [840]
    
    return list(best_list)
