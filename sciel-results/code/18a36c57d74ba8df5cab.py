"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    
    target = Fraction(5, 31)
    max_den = 10000000
    max_terms = 256
    
    best = None
    best_max = float('inf')
    
    def greedy_complete(rem, used_set, current_vals):
        """Pure greedy completion. Return list or None."""
        if rem == 0:
            return list(current_vals)
        if rem < 0:
            return None
        vals = list(current_vals)
        used = set(used_set)
        r = rem
        while r > 0:
            d_min = (r.denominator + r.numerator - 1) // r.numerator
            if d_min > max_den:
                return None
            d = d_min
            while d in used and d <= max_den:
                d += 1
            if d > max_den:
                return None
            vals.append(d)
            used.add(d)
            r -= Fraction(1, d)
            if len(vals) > max_terms:
                return None
        return vals
    
    # Strategy 1: Try various first denominators with greedy completion
    for d1 in range(8, 60):
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        vals = greedy_complete(r1, {d1}, [d1])
        if vals is not None:
            mx = max(vals)
            if mx < best_max:
                best_max = mx
                best = vals
    
    # Strategy 2: Try pairs with greedy completion
    for d1 in range(8, 35):
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        for d2 in range(d1 + 1, min(300, d1 + 200)):
            r2 = r1 - Fraction(1, d2)
            if r2 <= 0:
                continue
            vals = greedy_complete(r2, {d1, d2}, [d1, d2])
            if vals is not None:
                mx = max(vals)
                if mx < best_max:
                    best_max = mx
                    best = vals
    
    # Strategy 3: At each step, try greedy and greedy+1, greedy+2 (beam width 3)
    # This allows slight deviations that might lead to smaller final denominators
    def beam_complete(rem, used_set, current_vals, beam_width=3, max_depth=10):
        """Beam search: at each step, try the smallest few valid denominators."""
        if rem == 0:
            return list(current_vals)
        if rem < 0:
            return None
        if len(current_vals) >= max_terms:
            return None
        
        d_min = (rem.denominator + rem.numerator - 1) // rem.numerator
        if d_min > max_den:
            return None
        
        # Generate candidate denominators: d_min, d_min+1, ..., up to beam_width
        candidates = []
        d = d_min
        while len(candidates) < beam_width and d <= max_den:
            if d not in used_set:
                candidates.append(d)
            d += 1
        
        if not candidates:
            return None
        
        # Try each candidate, recursively
        best_result = None
        best_mx = float('inf')
        for c in candidates:
            new_rem = rem - Fraction(1, c)
            if new_rem < 0:
                continue
            new_used = used_set | {c}
            new_vals = current_vals + [c]
            result = beam_complete(new_rem, new_used, new_vals, beam_width, max_depth)
            if result is not None:
                mx = max(result)
                if mx < best_mx:
                    best_mx = mx
                    best_result = result
        
        return best_result
    
    # Try beam search from various starting points
    for d1 in range(8, 30):
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        vals = beam_complete(r1, {d1}, [d1], beam_width=3, max_depth=8)
        if vals is not None:
            mx = max(vals)
            if mx < best_max:
                best_max = mx
                best = vals
    
    # Fallback
    if best is None:
        r, values = target - Fraction(1, 8), [8]
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        best = values
    
    return best
# EVOLVE-BLOCK-END
