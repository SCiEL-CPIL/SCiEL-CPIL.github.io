def construct():
    num, den = 1, 1
    min_denom = 5
    result = []
    max_denom = 10000000
    max_terms = 256
    
    while num > 0:
        if len(result) >= max_terms:
            raise RuntimeError("Exceeded max terms")
        
        # Find smallest d >= min_denom such that 1/d <= num/den
        # d >= den / num
        d_candidate = (den + num - 1) // num
        if d_candidate < min_denom:
            d_candidate = min_denom
            
        if d_candidate > max_denom:
            raise RuntimeError("Denominator exceeds max limit")
            
        result.append(d_candidate)
        
        # Update sum: num/den - 1/d_candidate
        new_num = num * d_candidate - den
        new_den = den * d_candidate
        
        if new_num == 0:
            break
            
        # Simplify to keep numbers small
        from math import gcd
        g = gcd(new_num, new_den)
        num = new_num // g
        den = new_den // g
        
        min_denom = d_candidate + 1
        
    if num != 0:
        raise RuntimeError("Did not reach zero sum")
        
    return result
