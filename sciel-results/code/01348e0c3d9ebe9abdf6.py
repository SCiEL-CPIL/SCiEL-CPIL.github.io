"""Program for the bounded-LCM maximum clique task; returns the optimal 24-element set (divisors of 504)."""
# EVOLVE-BLOCK-START
import math

def construct():
    # The maximum cardinality of a subset S of [1,512] with lcm(a,b) <= 512
    # for all pairs equals max_{n <= 512} tau(n) = 24, achieved by n = 504
    # (504 = 2^3 * 3^2 * 7). Any two divisors of 504 have lcm dividing 504 <= 512.
    # We return all 24 divisors of 504.
    N = 512
    M = 504  # most highly composite number <= 512 with tau = 24
    
    # Enumerate divisors of M
    divisors = []
    for d in range(1, int(math.isqrt(M)) + 1):
        if M % d == 0:
            divisors.append(d)
            if d != M // d:
                divisors.append(M // d)
    divisors.sort()
    
    # Verify: every divisor is in [1, N] and pairwise lcm <= N
    for a in divisors:
        assert 1 <= a <= N
    valid = True
    for i in range(len(divisors)):
        for j in range(i + 1, len(divisors)):
            g = math.gcd(divisors[i], divisors[j])
            if divisors[i] // g * divisors[j] > N:
                valid = False
                break
        if not valid:
            break
    
    if valid and len(divisors) > 0:
        return divisors
    
    # Fallback (should not be reached): divisors of 360 also give 24
    divisors2 = []
    for d in range(1, int(math.isqrt(360)) + 1):
        if 360 % d == 0:
            divisors2.append(d)
            if d != 360 // d:
                divisors2.append(360 // d)
    divisors2.sort()
    if divisors2:
        return divisors2
    
    return [1]
# EVOLVE-BLOCK-END
