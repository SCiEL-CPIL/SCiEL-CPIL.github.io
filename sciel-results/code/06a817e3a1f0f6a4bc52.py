"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    TARGET = Fraction(5, 31)
    MAX_DEN = 10_000_000
    MAX_TERMS = 256

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def lcm(a, b):
        return a * b // gcd(a, b)

    # Number-theoretic approach for 3-term Egyptian fractions:
    # Find distinct a, b, c such that 1/a + 1/b + 1/c = 5/31
    # WLOG assume a <= b <= c, so c is the largest denominator.
    # 1/b + 1/c = 5/31 - 1/a = (5a - 31)/(31a)
    # Let N = 5a - 31, D = 31a
    # 1/b + 1/c = N/D
    # (b+c)/(bc) = N/D
    # D(b+c) = Nbc
    # Nbc - Db - Dc = 0
    # Multiply by N: N^2 bc - DNb - DNc = 0
    # (Nb - D)(Nc - D) = D^2
    # So for each divisor p of D^2, let q = D^2/p
    # Then Nb - D = p, Nc - D = q
    # b = (p + D) / N, c = (q + D) / N
    # We need b and c to be positive integers, distinct from a, and b <= c.

    def find_3term_solutions(max_c_bound=None):
        """Find all 3-term Egyptian fraction solutions for 5/31 with max denominator <= max_c_bound."""
        solutions = []
        # a must be such that 5a - 31 > 0, i.e., a >= 7
        # Also 1/a < 5/31, so a > 31/5 = 6.2, so a >= 7
        # Upper bound on a: since a <= b <= c and 1/a + 1/b + 1/c = 5/31,
        # we have 3/a >= 5/31, so a <= 93/5 = 18.6, so a <= 18
        for a in range(7, 19):
            N = 5 * a - 31
            D = 31 * a
            if N <= 0:
                continue
            # D^2 can be large, but we only need divisors p where:
            # b = (p + D) / N must be integer, so (p + D) % N == 0
            # c = (D^2/p + D) / N must be integer
            # Also b >= a (since a <= b) and c >= b
            # And if max_c_bound, c <= max_c_bound
            
            # Enumerate divisors of D^2
            D2 = D * D
            # Get divisors of D^2
            divisors = set()
            for i in range(1, int(math.isqrt(D2)) + 1):
                if D2 % i == 0:
                    divisors.add(i)
                    divisors.add(D2 // i)
            
            for p in divisors:
                q = D2 // p
                # b = (p + D) / N, c = (q + D) / N
                if (p + D) % N != 0:
                    continue
                b = (p + D) // N
                if (q + D) % N != 0:
                    continue
                c = (q + D) // N
                # Check b <= c
                if b > c:
                    continue
                # Check distinctness: a, b, c all distinct
                if a == b or a == c or b == c:
                    continue
                # Check positive
                if b < 1 or c < 1:
                    continue
                # Check a <= b
                if a > b:
                    continue
                # Check bound on c
                if max_c_bound is not None and c > max_c_bound:
                    continue
                # Verify the solution
                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == TARGET:
                    solutions.append((a, b, c))
        
        return solutions

    # Search for 3-term solutions with increasing max denominator bound
    best_solution = None
    best_max = MAX_DEN

    # First, try to find any 3-term solution
    # We know from the inspiration that [8, 31, 248] works with max 248
    # Let's search from small max_c upward to find the optimal 3-term solution
    
    # But first, let's also consider 2-term solutions (though unlikely for 5/31)
    # 1/a + 1/b = 5/31 => (a+b)/(ab) = 5/31 => 31(a+b) = 5ab
    # 5ab - 31a - 31b = 0
    # 25ab - 155a - 155b = 0
    # (5a - 31)(5b - 31) = 31^2 = 961
    # 961 = 31 * 31, so divisors: 1, 31, 961
    # 5a - 31 = 1 => a = 32/5 (not integer)
    # 5a - 31 = 31 => a = 62/5 (not integer)
    # 5a - 31 = 961 => a = 992/5 (not integer)
    # No 2-term solution.

    # Search for 3-term solutions with max denominator from 31 upward
    # The minimum possible max denominator is at least 31 (since 1/31 is a term)
    # Actually, we need to check if there's a solution with all denominators < 248
    
    found = False
    for max_c in range(31, 250):
        if time.time() - start_time > DEADLINE:
            break
        sols = find_3term_solutions(max_c)
        if sols:
            # Pick the one with smallest max denominator
            for (a, b, c) in sols:
                if c <= max_c and c < best_max:
                    best_max = c
                    best_solution = [a, b, c]
            found = True
            break  # We found a solution with this max_c, which is minimal

    if not found:
        # If no 3-term solution found in range, try broader search or fallback
        # Try all 3-term solutions without bound
        sols = find_3term_solutions(None)
        if sols:
            for (a, b, c) in sols:
                if c < best_max:
                    best_max = c
                    best_solution = [a, b, c]

    if not best_solution:
        # Fallback: greedy
        used = set()
        r = TARGET
        best_solution = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d in used:
                d += 1
                while d in used:
                    d += 1
            best_solution.append(d)
            used.add(d)
            r -= Fraction(1, d)
            if len(best_solution) > MAX_TERMS:
                break

    # Verify
    if best_solution:
        s = sum(Fraction(1, d) for d in best_solution)
        if s != TARGET or len(set(best_solution)) != len(best_solution) or any(d < 1 or d > MAX_DEN for d in best_solution) or len(best_solution) > MAX_TERMS:
            # Absolute fallback
            best_solution = [8, 31, 248]

    return best_solution
# EVOLVE-BLOCK-END
