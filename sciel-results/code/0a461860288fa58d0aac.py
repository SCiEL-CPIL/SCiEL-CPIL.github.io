# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for the 90s wall-clock limit

    def verify(result):
        if not result:
            return False
        total_num, total_den = 0, 1
        for d in result:
            total_num = total_num * d + total_den
            total_den = total_den * d
            g = gcd(total_num, total_den)
            total_num //= g
            total_den //= g
        if total_num != total_den:
            return False
        if len(set(result)) != len(result):
            return False
        if not all(5 <= d <= 10000000 for d in result):
            return False
        return True

    # Pre-verified known solutions (smallest count first)
    # These are known Egyptian fraction representations of 1 with denominators >= 5
    known_solutions = [
        [5, 5, 5, 5, 5],  # Wait, these must be distinct! Let me fix this.
    ]
    
    # Actually, I need distinct denominators. Let me use a known minimal solution.
    # Known: 1 = 1/5 + 1/6 + 1/8 + 1/12 + 1/20 + 1/30 + 1/60 + 1/120 + 1/240
    # But that's 9 terms. Let me try to find smaller.
    
    # Actually, let me just search from small term counts with better pruning.
    
    def search(remaining_num, remaining_den, min_den, terms_left, current, deadline):
        if time.time() > deadline:
            return None
        
        if terms_left == 0:
            if remaining_num == 0:
                return list(current)
            return None
        
        if remaining_num == 0:
            return None
        
        if terms_left == 1:
            if remaining_den % remaining_num == 0:
                d = remaining_den // remaining_num
                if d >= min_den and d >= 5 and d <= 10000000:
                    current.append(d)
                    result = list(current)
                    current.pop()
                    return result
            return None
        
        # Pruning: max achievable with terms_left terms starting from min_den
        # Use harmonic series approximation for speed
        # H(n) ≈ ln(n) + gamma + 1/(2n) - 1/(12n^2)
        # Sum from min_den to min_den+terms_left-1 ≈ ln((min_den+terms_left-1)/min_den)
        import math
        if min_den > 0 and terms_left > 0:
            max_sum_approx = math.log((min_den + terms_left) / min_den)
            remaining_approx = remaining_num / remaining_den
            if remaining_approx > max_sum_approx * 1.01:  # 1% tolerance
                return None
        
        # Lower bound on d
        d_lower = max(min_den, (remaining_den + remaining_num - 1) // remaining_num)
        
        # Upper bound on d:
        # remaining <= terms_left / d  => d <= terms_left / remaining
        # d <= floor(terms_left * remaining_den / remaining_num)
        d_upper = (terms_left * remaining_den) // remaining_num
        
        if d_upper < d_lower:
            return None
            
        # Additional pruning: if terms_left terms all at d_upper, their sum must be >= remaining
        # 1/d_upper + 1/(d_upper+1) + ... + 1/(d_upper + terms_left - 1) >= remaining
        # Approximate: terms_left / d_upper >= remaining
        # This is already implied by d_upper = terms_left * remaining_den // remaining_num
        
        for d in range(d_lower, d_upper + 1):
            if time.time() > deadline:
                return None
            
            # 1/d <= remaining?
            if d * remaining_num < remaining_den:
                continue
            
            new_num = remaining_num * d - remaining_den
            new_den = remaining_den * d
            
            if new_num < 0:
                continue
            
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            if terms_left - 1 == 0:
                if new_num == 0:
                    current.append(d)
                    result = list(current)
                    current.pop()
                    return result
                continue
            
            current.append(d)
            result = search(new_num, new_den, d + 1, terms_left - 1, current, deadline)
            if result is not None:
                return result
            current.pop()
        
        return None

    # Try small term counts first (minimum possible is likely 5-6 terms)
    for num_terms in [5, 6, 7, 8, 9, 10, 12, 15, 20]:
        if time.time() > deadline - 5:
            break
        result = search(1, 1, 5, num_terms, [], deadline)
        if result and verify(result):
            return result

    # Fallback: greedy with better termination
    remaining_num, remaining_den = 1, 1
    result = []
    min_d = 5
    for _ in range(256):
        if remaining_num == 0:
            break
        if time.time() > deadline - 2:
            break
        d = max(min_d, (remaining_den + remaining_num - 1) // remaining_num)
        while d * remaining_num < remaining_den:
            d += 1
        if d > 10000000:
            break
        result.append(d)
        remaining_num = remaining_num * d - remaining_den
        remaining_den = remaining_den * d
        g = gcd(remaining_num, remaining_den)
        if g > 1:
            remaining_num //= g
            remaining_den //= g
        min_d = d + 1
    
    if remaining_num == 0 and verify(result):
        return result
    
    # Last resort: return a known valid solution if we have one
    # 1 = 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/12 + 1/14 + 1/15 + 1/18 + 1/20 + 1/24 + 1/28 + 1/30 + 1/35 + 1/40 + 1/42 + 1/45 + 1/50 + 1/56 + 1/60 + 1/63 + 1/70 + 1/72 + 1/75 + 1/80 + 1/84 + 1/90 + 1/96 + 1/100 + 1/105 + 1/112 + 1/120 + 1/126 + 1/135 + 1/140 + 1/144 + 1/150 + 1/156 + 1/160 + 1/168 + 1/175 + 1/180 + 1/189 + 1/192 + 1/196 + 1/200 + 1/210 + 1/216 + 1/224 + 1/225 + 1/231 + 1/240 + 1/252 + 1/264 + 1/270 + 1/280 + 1/288 + 1/294 + 1/300 + 1/308 + 1/315 + 1/320 + 1/336 + 1/342 + 1/348 + 1/350 + 1/360 + 1/375 + 1/384 + 1/390 + 1/396 + 1/400 + 1/408 + 1/420 + 1/432 + 1/440 + 1/448 + 1/450 + 1/462 + 1/480 + 1/495 + 1/504 + 1/512 + 1/525 + 1/540 + 1/546 + 1/560 + 1/576 + 1/588 + 1/600 + 1/616 + 1/630 + 1/640 + 1/648 + 1/660 + 1/672 + 1/675 + 1/686 + 1/700 + 1/720 + 1/735 + 1/750 + 1/756 + 1/768 + 1/770 + 1/784 + 1/800 + 1/810 + 1/825 + 1/840 + 1/858 + 1/864 + 1/875 + 1/880 + 1/896 + 1/900 + 1/918 + 1/924 + 1/936 + 1/945 + 1/952 + 1/960 + 1/972 + 1/980 + 1/990 + 1/1000
    # This is too many. Let me just raise an error if we can't find one.
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
