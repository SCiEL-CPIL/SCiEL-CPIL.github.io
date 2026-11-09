"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    # Deadline guard: evaluator kills at 90s; keep a safety margin.
    start_time = time.time()
    DEADLINE = 85.0  # seconds, leaving 5s margin
    
    TARGET = Fraction(5, 31)
    MAX_DEN = 10_000_000
    MAX_TERMS = 256
    
    # We want to minimize the largest denominator in an Egyptian fraction expansion
    # of 5/31 using 1..256 distinct denominators in [1, 10000000].
    
    # Strategy:
    # 1. Try small sets of "anchor" denominators (the larger ones) and then
    #    complete with greedy or small denominators.
    # 2. The key insight: to minimize the max denominator, we want to use many
    #    small denominators. The greedy algorithm tends to produce large denominators.
    #    Instead, we should try to find a set of small distinct denominators whose
    #    reciprocals sum to 5/31.
    
    # Approach: Use a bounded search / backtracking with pruning.
    # Since we can use up to 256 terms, and denominators must be distinct,
    # we can try to find a solution where all denominators are small.
    
    # Let's think about this differently. The sum of reciprocals of 1..n is
    # approximately log(n) + gamma. We need sum = 5/31 ≈ 0.1613.
    # If we use denominators from some set, the sum of 1/d for d in some range.
    
    # Actually, let's try a direct approach:
    # - Try to find a subset of {2, 3, 4, ..., M} (distinct) such that the sum
    #   of reciprocals equals exactly 5/31, minimizing M.
    # - This is a subset-sum problem with rational numbers.
    
    # Since the target is small (5/31), we don't need many small denominators.
    # 1/2 = 0.5 > 5/31, so we can't use 1 or 2 if we want the sum to be exactly 5/31
    # with positive terms... wait, no. We need the sum to BE 5/31. So we can use
    # denominators larger than 31 (since 1/31 < 5/31). Actually 1/31 ≈ 0.032, and
    # 5/31 ≈ 0.161. So we need several terms.
    
    # Let me try a smarter approach:
    # Use a greedy-like method but with optimization: try different first denominators
    # and see which gives the smallest maximum.
    
    # Better: try to find a representation using denominators from a small range.
    # Let's try denominators from [4, M] for various M, and use backtracking.
    
    # Given the time constraint, let's use a heuristic search:
    # 1. Start with a greedy solution to get an upper bound on max denominator.
    # 2. Then try to improve by swapping in smaller denominators.
    
    def greedy_complete(remaining, used, max_den_so_far):
        """Complete the Egyptian fraction using greedy, returning (denoms, max_den) or None."""
        denoms = []
        r = remaining
        while r > 0:
            if r.numerator == 0:
                break
            # Greedy: largest d such that 1/d <= r
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d in used:
                # Find next available
                d += 1
                while d in used:
                    d += 1
            if d > MAX_DEN:
                return None
            denoms.append(d)
            used.add(d)
            r -= Fraction(1, d)
            max_den_so_far = max(max_den_so_far, d)
            if len(denoms) + len(used) > MAX_TERMS:
                return None
        if r != 0:
            return None
        return denoms, max_den_so_far
    
    # Initial greedy solution
    used = set()
    r = TARGET
    initial_denoms = []
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d in used:
            d += 1
            while d in used:
                d += 1
        initial_denoms.append(d)
        used.add(d)
        r -= Fraction(1, d)
        if len(initial_denoms) > MAX_TERMS:
            break
    
    best_max = max(initial_denoms) if initial_denoms else MAX_DEN
    best_solution = initial_denoms[:]
    
    # Try to improve by trying different starting denominators
    # For each possible first denominator d1 in a reasonable range,
    # try to complete with greedy and see if we get a smaller max.
    
    # The first denominator should be at least ceil(31/5) = 7 (since 1/7 ≈ 0.143 < 5/31 ≈ 0.161)
    # Actually 1/6 ≈ 0.167 > 5/31, so d1 >= 7.
    
    # Let's try d1 from 7 to some upper bound, and for each, try to complete
    # the remaining fraction with a set of denominators that are all <= current best_max.
    
    # To make this efficient, use a bounded greedy: at each step, choose the smallest
    # available denominator that keeps the remainder completable within the bound.
    
    def bounded_greedy(remaining, used, max_allowed):
        """Try to complete using denominators <= max_allowed."""
        denoms = []
        r = remaining
        while r > 0:
            if time.time() - start_time > DEADLINE:
                return None
            # Smallest d not in used such that 1/d <= r and d <= max_allowed
            # d >= ceil(1/r)
            d_min = (r.denominator + r.numerator - 1) // r.numerator
            if d_min < 1:
                d_min = 1
            found = False
            for d in range(d_min, min(max_allowed, MAX_DEN) + 1):
                if d in used:
                    continue
                # Check if 1/d <= r
                if Fraction(1, d) > r:
                    continue
                # Try this d
                new_r = r - Fraction(1, d)
                used.add(d)
                denoms.append(d)
                if new_r == 0:
                    return denoms
                if len(denoms) >= MAX_TERMS:
                    used.discard(d)
                    denoms.pop()
                    continue
                # Recurse
                result = bounded_greedy(new_r, used, max_allowed)
                if result is not None:
                    return denoms + result
                used.discard(d)
                denoms.pop()
                found = True
            if not found and denoms:
                return None
            if not found and not denoms:
                return None
        return denoms if denoms else []
    
    # This recursive approach might be too slow. Let's use a simpler heuristic:
    # Try different max bounds and see if we can find a solution.
    
    # Actually, let's try a different approach: use the fact that we can use up to 256 terms.
    # With many terms, we can use small denominators. The sum of 1/d for d in a set S
    # can be adjusted by including/excluding terms.
    
    # Let's try: use denominators from a range [a, b] and find a subset that sums to 5/31.
    # This is hard. Instead, let's try to construct a solution with all denominators <= some bound B.
    
    # Simple approach: try B from small to large, and for each B, try to find a subset
    # of {4, 5, ..., B} whose reciprocals sum to 5/31.
    
    # For efficiency, use a greedy-with-backtracking limited in depth.
    
    # Let's try a practical approach:
    # 1. Get initial greedy solution (max ~ some value)
    # 2. Try to reduce the max by replacing large denominators with combinations of smaller ones
    
    # For now, let's just try a few first-denominator choices and greedy completion,
    # tracking the best max denominator.
    
    # Try first denominator from 7 to 20
    for d1 in range(7, 25):
        if time.time() - start_time > DEADLINE:
            break
        used = {d1}
        r = TARGET - Fraction(1, d1)
        if r <= 0:
            continue
        # Try to complete with greedy, but prefer smaller denominators
        denoms = [d1]
        current_r = r
        success = True
        while current_r > 0:
            if time.time() - start_time > DEADLINE:
                success = False
                break
            # Choose smallest available d such that 1/d <= current_r
            d_min = (current_r.denominator + current_r.numerator - 1) // current_r.numerator
            if d_min < 1:
                d_min = 1
            chosen = None
            for d in range(d_min, best_max + 1):
                if d in used:
                    continue
                if Fraction(1, d) > current_r:
                    continue
                chosen = d
                break
            if chosen is None:
                # No valid d within current best_max, try greedy without bound
                d = (current_r.denominator + current_r.numerator - 1) // current_r.numerator
                while d in used:
                    d += 1
                chosen = d
                if chosen > best_max:
                    success = False
                    break
            used.add(chosen)
            denoms.append(chosen)
            current_r -= Fraction(1, chosen)
            if len(denoms) > MAX_TERMS:
                success = False
                break
        if success and current_r == 0 and len(denoms) <= MAX_TERMS:
            max_d = max(denoms)
            if max_d < best_max:
                best_max = max_d
                best_solution = denoms[:]
    
    # Verify solution
    if best_solution:
        s = sum(Fraction(1, d) for d in best_solution)
        if s != TARGET or len(set(best_solution)) != len(best_solution) or any(d < 1 or d > MAX_DEN for d in best_solution) or len(best_solution) > MAX_TERMS:
            # Fallback to initial greedy
            best_solution = initial_denoms[:]
    
    if not best_solution:
        # Absolute fallback
        best_solution = [8]
        r = TARGET - Fraction(1, 8)
        used = {8}
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            while d in used:
                d += 1
            best_solution.append(d)
            used.add(d)
            r -= Fraction(1, d)
    
    return best_solution
# EVOLVE-BLOCK-END
