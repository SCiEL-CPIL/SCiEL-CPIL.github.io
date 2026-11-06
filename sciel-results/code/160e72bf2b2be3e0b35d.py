"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from collections import defaultdict

def construct():
    deadline = time.time() + 55
    target = Fraction(5, 31)
    
    # Known fallback
    best_max = 248
    best_solution = [8, 31, 248]
    
    # Verify fallback
    s = sum(Fraction(1, d) for d in best_solution)
    if s != target:
        best_solution = [8, 31, 248]
        best_max = 248
    
    # Meet-in-the-middle approach:
    # For a given M, find distinct denominators in [1, M] summing to target.
    # Split into two halves: find all sums of k terms, then check complement.
    
    # Strategy: For each M from lower bound upward, check if solution exists.
    # Use pair sums for 2-term, triple sums for 3-term, etc.
    
    # Lower bound: need at least 3 terms (2-term impossible as shown).
    # Minimum M: try from 7 upward.
    
    # Precompute helper: all pairs (a,b) with 1<=a<b<=M, store sum -> list of pairs
    # Then for 3 terms: for each c, check if target - 1/c is in pair_sums with a,b < c
    # For 4 terms: split into two pairs, check if sum1 + sum2 = target
    
    # Let's try 3-term solutions first for small M
    # 1/a + 1/b + 1/c = 5/31, a < b < c <= M
    # For each c from 7 to M:
    #   1/a + 1/b = 5/31 - 1/c = (5c-31)/(31c)
    #   Solve (31c*a - (5c-31))(31c*b - (5c-31)) = (5c-31)^2
    
    # Actually, let me use a more direct approach for 3 terms:
    # Fix c, then solve 1/a + 1/b = p/q where p/q = (5c-31)/(31c)
    # (qa-p)(qb-p) = p^2
    
    def find_three_terms(M):
        """Find a<b<c<=M with 1/a+1/b+1/c = 5/31."""
        from math import gcd, isqrt
        for c in range(7, M + 1):
            if time.time() > deadline:
                return None
            p = 5 * c - 31
            q = 31 * c
            if p <= 0:
                continue
            g = gcd(p, q)
            p_r, q_r = p // g, q // g
            p2 = p_r * p_r
            sqrt_p2 = isqrt(p2)
            for u in range(1, sqrt_p2 + 1):
                if p2 % u != 0:
                    continue
                for pair in [(u, p2 // u), (p2 // u, u)]:
                    u1, u2 = pair
                    if (u1 + p_r) % q_r != 0:
                        continue
                    a = (u1 + p_r) // q_r
                    if (u2 + p_r) % q_r != 0:
                        continue
                    b = (u2 + p_r) // q_r
                    if a < 1 or b < 1:
                        continue
                    if a > b:
                        a, b = b, a
                    if a == b:
                        continue
                    if a >= b or b >= c:
                        continue
                    # Verify
                    if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                        return [a, b, c]
        return None
    
    # Try 3-term for M from 7 to 100
    for M in range(7, 101):
        if time.time() > deadline:
            break
        sol = find_three_terms(M)
        if sol is not None:
            if M < best_max:
                best_max = M
                best_solution = sol
            break
    
    # If no 3-term found with small M, try 4-term with meet-in-the-middle
    # 1/a + 1/b + 1/c + 1/d = 5/31, a < b < c < d <= M
    # Precompute all pair sums for denominators in [1, M]
    # Then check if two disjoint pairs sum to target
    
    def find_four_terms(M):
        """Find a<b<c<d<=M with 1/a+1/b+1/c+1/d = 5/31 using meet-in-the-middle."""
        if time.time() > deadline:
            return None
        # Precompute all pairs (a,b) with 1<=a<b<=M
        # Store: sum -> list of (a,b)
        pair_sums = defaultdict(list)
        for a in range(1, M):
            for b in range(a + 1, M + 1):
                s = Fraction(1, a) + Fraction(1, b)
                pair_sums[s].append((a, b))
        
        # For each pair (a,b), check if target - (1/a+1/b) is a sum of two
        # distinct denominators c,d not in {a,b} with c<d
        for a in range(1, M - 1):
            if time.time() > deadline:
                return None
            for b in range(a + 1, M):
                s_ab = Fraction(1, a) + Fraction(1, b)
                rem = target - s_ab
                if rem <= 0:
                    continue
                if rem in pair_sums:
                    for (c, d) in pair_sums[rem]:
                        if c > b and d > c:  # ensures a<b<c<d
                            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == target:
                                return [a, b, c, d]
        return None
    
    # Try 4-term for M from 7 to best_max-1
    if best_max > 10:
        for M in range(7, min(best_max, 80)):
            if time.time() > deadline:
                break
            sol = find_four_terms(M)
            if sol is not None:
                if M < best_max:
                    best_max = M
                    best_solution = sol
                break
    
    # Try 5-term with meet-in-the-middle: pairs + triples
    # Precompute all pair sums and all triple sums
    # Check if pair_sum + triple_sum = target with disjoint denominators
    
    def find_five_terms(M):
        """Find 5 distinct denominators <= M summing to 5/31."""
        if time.time() > deadline:
            return None
        # Precompute pair sums: sum -> list of (a,b)
        pair_sums = defaultdict(list)
        for a in range(1, M - 1):
            for b in range(a + 1, M):
                s = Fraction(1, a) + Fraction(1, b)
                pair_sums[s].append((a, b))
        
        # For each triple (c,d,e) with c<d<e<=M:
        #   Check if target - (1/c+1/d+1/e) is in pair_sums with a<b<c
        for c in range(1, M - 2):
            if time.time() > deadline:
                return None
            for d in range(c + 1, M - 1):
                for e in range(d + 1, M + 1):
                    s_cde = Fraction(1, c) + Fraction(1, d) + Fraction(1, e)
                    rem = target - s_cde
                    if rem <= 0:
                        continue
                    if rem in pair_sums:
                        for (a, b) in pair_sums[rem]:
                            if a < b and b < c:
                                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) + Fraction(1, e) == target:
                                    return [a, b, c, d, e]
        return None
    
    # Try 5-term for M from 7 to min(best_max-1, 50)
    if best_max > 15:
        for M in range(7, min(best_max, 50)):
            if time.time() > deadline:
                break
            sol = find_five_terms(M)
            if sol is not None:
                if M < best_max:
                    best_max = M
                    best_solution = sol
                break
    
    # Try 6-term with meet-in-the-middle: 3+3
    # Precompute all triple sums
    # Check if two disjoint triples sum to target
    
    def find_six_terms(M):
        """Find 6 distinct denominators <= M summing to 5/31."""
        if time.time() > deadline:
            return None
        # Precompute triple sums: sum -> list of (a,b,c)
        triple_sums = defaultdict(list)
        for a in range(1, M - 2):
            for b in range(a + 1, M - 1):
                for c in range(b + 1, M + 1):
                    s = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
                    triple_sums[s].append((a, b, c))
        
        # For each triple (a,b,c), check if target - sum is in triple_sums
        # with d<e<f and c<d
        for a in range(1, M - 4):
            if time.time() > deadline:
                return None
            for b in range(a + 1, M - 3):
                for c in range(b + 1, M - 2):
                    s_abc = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
                    rem = target - s_abc
                    if rem <= 0:
                        continue
                    if rem in triple_sums:
                        for (d, e, f) in triple_sums[rem]:
                            if d > c and e > d and f > e:
                                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) + Fraction(1, e) + Fraction(1, f) == target:
                                    return [a, b, c, d, e, f]
        return None
    
    # Try 6-term for M from 7 to min(best_max-1, 40)
    if best_max > 20:
        for M in range(7, min(best_max, 40)):
            if time.time() > deadline:
                break
            sol = find_six_terms(M)
            if sol is not None:
                if M < best_max:
                    best_max = M
                    best_solution = sol
                break
    
    # Final verification
    if best_solution:
        s = sum(Fraction(1, d) for d in best_solution)
        if s != target:
            best_solution = [8, 31, 248]
            best_max = 248
    
    return best_solution
# EVOLVE-BLOCK-END
