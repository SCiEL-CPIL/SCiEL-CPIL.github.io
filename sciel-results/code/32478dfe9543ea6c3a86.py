import time
import math

def construct():
    deadline = time.time() + 78.0
    answers = [None] * 254
    BOUND = 10**12

    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None

        # 1/x + 1/y + 1/z = 4/n, x < y < z
        # x > n/4, so x_min = n//4 + 1
        # Also 4/x > 4/n - 4/n (since 1/y+1/z < 4/n), so x can be large
        # But 1/x + 1/y + 1/z = 4/n with y>x, z>y means 3/x > 4/n => x < 3n/4
        x_max = (3 * n) // 4
        if x_max < n // 4 + 1:
            x_max = n // 4 + 1

        for x in range(n // 4 + 1, x_max + 1):
            a = 4 * x - n
            b = n * x
            # Need 1/y + 1/z = a/b, y > x, z > y
            # (a*y - b)(a*z - b) = b^2
            # d = a*y - b is a divisor of b^2, d > a*x - b (since y > x)
            # y = (d + b) / a, must be integer, y > x
            # z = (d' + b) / a where d' = b^2 / d, must be integer, z > y
            b2 = b * b
            # Enumerate divisors of b^2 up to sqrt
            root = math.isqrt(b2)
            # We need d such that a*y - b = d, y = (d+b)/a integer, y > x
            # d > a*x - b, and d divides b^2
            # Also z = b^2/d + b) / a, z > y means b^2/d > d, so d < b
            # Actually z > y iff (b^2/d + b)/a > (d+b)/a iff b^2/d > d iff d < b
            # So we need divisors d of b^2 with d < b and d > a*x - b
            # and (d + b) % a == 0
            #
            # To find divisors of b^2, factor b and get divisors of b^2
            # b = n * x, factor it
            # For efficiency, factor b
            temp = b
            factors = {}
            f = 2
            while f * f <= temp:
                while temp % f == 0:
                    factors[f] = factors.get(f, 0) + 1
                    temp //= f
                f += 1
            if temp > 1:
                factors[temp] = factors.get(temp, 0) + 1
            # b^2 has exponents doubled
            # Generate divisors of b^2 that are < b and > a*x - b
            # This could still be many, so let's generate all divisors of b^2
            # and filter. For b up to ~256*192 ≈ 49152, b^2 ≈ 2.4e9, 
            # number of divisors is manageable (a few thousand at most)
            
            # Generate divisors of b^2
            primes = list(factors.keys())
            exps = [factors[p] * 2 for p in primes]
            
            # Use recursive generation
            divisors = [1]
            for i, p in enumerate(primes):
                e = exps[i]
                current = []
                for d in divisors:
                    val = d
                    for _ in range(e + 1):
                        current.append(val)
                        val *= p
                divisors = current
            
            # Filter: d < b (for z > y), d > a*x - b (for y > x)
            # and (d + b) % a == 0
            ax_minus_b = a * x - b
            best = None
            for d in divisors:
                if d >= b:
                    continue
                if d <= ax_minus_b:
                    continue
                if (d + b) % a != 0:
                    continue
                y = (d + b) // a
                if y <= x:
                    continue
                # z = (b^2 // d + b) // a, but need b^2 % d == 0 (it is since d|b^2)
                d2 = b2 // d
                if (d2 + b) % a != 0:
                    continue
                z = (d2 + b) // a
                if z <= y:
                    continue
                if z > BOUND:
                    continue
                # Found a valid solution
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers[idx] = found

    return answers
