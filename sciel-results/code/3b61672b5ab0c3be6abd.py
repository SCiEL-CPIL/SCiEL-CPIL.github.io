"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    target = Fraction(5, 31)
    best_max = None
    best_result = None
    
    def update_best(result):
        nonlocal best_max, best_result
        if result is None:
            return
        max_d = max(result)
        if max_d > 10000000:
            return
        if len(result) < 1 or len(result) > 256:
            return
        if len(set(result)) != len(result):
            return
        # Verify sum
        s = sum(Fraction(1, d) for d in result)
        if s != target:
            return
        if best_max is None or max_d < best_max:
            best_max = max_d
            best_result = result
    
    # Try 2 terms: 1/a + 1/b = 5/31
    # 31(a+b) = 5ab => a = 31b/(5b-31)
    for b in range(7, 13):
        denom = 5 * b - 31
        if denom <= 0:
            continue
        num = 31 * b
        if num % denom == 0:
            a = num // denom
            if a > b and a >= 1:
                update_best([a, b])
    
    # Systematic search for 3+ terms
    # For k terms, search a1 < a2 < ... < ak
    # Key: minimize max denominator, so we want all denominators small
    
    def search_terms(remaining, last_denom, max_terms, current_terms, current_max):
        """Recursively find up to max_terms more denominators > last_denom."""
        if time.time() > deadline:
            return None
        if remaining == 0:
            return list(current_terms)
        if max_terms == 0:
            return None
        if remaining < 0:
            return None
        
        # min_d: 1/d <= remaining => d >= ceil(1/remaining)
        min_d = max(last_denom + 1, -(-remaining.denominator // remaining.numerator))
        
        # max_d: even with max_terms terms, if 1/d < remaining/max_terms, infeasible
        # So d <= floor(max_terms / remaining) = floor(max_terms * rem.den / rem.num)
        if remaining.numerator == 0:
            return None
        max_d = (max_terms * remaining.denominator) // remaining.numerator
        if max_d < min_d:
            return None
        # Also cap max_d to avoid huge searches
        max_d = min(max_d, last_denom + 500)
        
        for d in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
            new_rem = remaining - Fraction(1, d)
            if new_rem < 0:
                break
            result = search_terms(new_rem, d, max_terms - 1, current_terms + [d], max(current_max, d))
            if result is not None:
                return result
        return None
    
    # Try 3 terms systematically
    for a in range(7, 31):
        if time.time() > deadline:
            break
        rem2 = target - Fraction(1, a)
        if rem2 <= 0:
            continue
        result = search_terms(rem2, a, 2, [a], a)
        if result is not None:
            update_best(result)
    
    # Try 4 terms
    for a in range(7, 25):
        if time.time() > deadline:
            break
        rem2 = target - Fraction(1, a)
        if rem2 <= 0:
            continue
        result = search_terms(rem2, a, 3, [a], a)
        if result is not None:
            update_best(result)
    
    # Try 5 terms
    for a in range(7, 20):
        if time.time() > deadline:
            break
        rem2 = target - Fraction(1, a)
        if rem2 <= 0:
            continue
        result = search_terms(rem2, a, 4, [a], a)
        if result is not None:
            update_best(result)
    
    # Try 6 terms
    for a in range(7, 18):
        if time.time() > deadline:
            break
        rem2 = target - Fraction(1, a)
        if rem2 <= 0:
            continue
        result = search_terms(rem2, a, 5, [a], a)
        if result is not None:
            update_best(result)
    
    # Local optimization: try to replace the largest denominator with a smaller one
    if best_result is not None:
        for _ in range(50):  # limited iterations
            if time.time() > deadline:
                break
            cur = list(best_result)
            cur_max = max(cur)
            # Try replacing cur_max with something smaller
            improved = False
            for idx in range(len(cur)):
                if cur[idx] != cur_max:
                    continue
                # Remove this term, find replacement
                cur_without = [d for i, d in enumerate(cur) if i != idx]
                rem = target - sum(Fraction(1, d) for d in cur_without)
                if rem <= 0:
                    continue
                # Find a denominator d < cur_max that works
                last_d = max(cur_without) if cur_without else 0
                min_d = max(last_d + 1, -(-rem.denominator // rem.numerator))
                max_d = cur_max - 1
                if max_d < min_d:
                    continue
                for d in range(min_d, max_d + 1):
                    new_rem = rem - Fraction(1, d)
                    if new_rem == 0:
                        new_cur = sorted(cur_without + [d])
                        if len(set(new_cur)) == len(new_cur):
                            update_best(new_cur)
                            improved = True
                            break
                    elif new_rem > 0:
                        # Check if we can complete with one more term
                        if new_rem.numerator == 1:
                            c = new_rem.denominator
                            if c > d and c < cur_max and c not in new_cur:
                                new_cur = sorted(cur_without + [d, c])
                                if len(set(new_cur)) == len(new_cur):
                                    update_best(new_cur)
                                    improved = True
                                    break
                if improved:
                    break
            if not improved:
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback: greedy
    r = target
    values = []
    while r > 0 and len(values) < 256:
        d = -(-r.denominator // r.numerator)
        if d in values:
            d += 1
        values.append(d)
        r -= Fraction(1, d)
    return values

# EVOLVE-BLOCK-END
