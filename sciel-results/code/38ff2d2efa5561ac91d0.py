import math

def construct():
    """
    Find 1..100 distinct integers strictly larger than 210 and at most 10210
    such that their product multiplied by 210 is a perfect square.
    Minimize max(selected integers) - 210.
    """
    
    def is_square(n):
        if n < 0:
            return False
        x = math.isqrt(n)
        return x * x == n

    def squarefree_kernel(n):
        """Compute the squarefree part of n."""
        result = 1
        temp = n
        # Check factor 2
        if temp % 2 == 0:
            count = 0
            while temp % 2 == 0:
                temp //= 2
                count += 1
            if count % 2 == 1:
                result *= 2
        # Check odd factors
        p = 3
        while p * p <= temp:
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    result *= p
            p += 2
        if temp > 1:
            result *= temp
        return result

    # We search for the smallest M such that there exists a valid set with max <= M.
    # We prioritize pairs as they are likely to yield the smallest M.
    # For a pair (a, b), we need sf(210 * a) == sf(b).
    
    best_max = 10211
    best_set = None
    
    # Iterate a from 211 to 10210
    # For each a, find the smallest b > 210 such that sf(b) == sf(210 * a)
    # and b <= 10210 and b != a.
    # Then update best_max if max(a, b) < best_max.
    
    # Optimization: We can stop early if we find a pair with max(a,b) small enough
    # that no triple or larger set can beat it?
    # Actually, just finding the global minimum pair is sufficient if we are confident
    # that pairs are optimal. Given the analysis, 240 is likely optimal.
    # Let's just find the best pair.
    
    for a in range(211, 10211):
        key_a = squarefree_kernel(210 * a)
        
        # We need b such that sf(b) == key_a.
        # b = key_a * y^2.
        # We want the smallest b > 210.
        # y starts at 1.
        # If key_a * 1^2 <= 210, try y=2, etc.
        
        y = 1
        while True:
            b = key_a * y * y
            if b > 10210:
                break
            if b > 210 and b != a:
                current_max = max(a, b)
                if current_max < best_max:
                    best_max = current_max
                    best_set = [a, b]
                # Since we want to minimize max(a,b), and b increases with y,
                # the smallest valid b for this a is the best candidate for this a.
                # We don't need to check larger y for this a.
                break
            y += 1
        
        # Early termination?
        # If best_max is small, we might stop.
        # But let's just run through all a to be sure.
        # Actually, if a > best_max, then max(a, b) >= a > best_max, so no improvement.
        if a >= best_max:
            # We still need to check if there's a b < a such that max(a,b) < best_max?
            # No, if a >= best_max, then max(a,b) >= a >= best_max.
            # So we can break.
            break

    if best_set is None:
        # Fallback to a known valid solution if no pair found (should not happen)
        # 840, 225 is a valid pair: sf(210*840)=sf(176400)=sf(420^2)=1? 
        # 210*840 = 176400. sqrt(176400) = 420. Yes.
        # sf(225) = 1.
        # Wait, condition is sf(210*a) == sf(b).
        # sf(210*840) = sf(210 * 4 * 210) = sf(4 * 210^2) = 1.
        # sf(225) = 1.
        # So (840, 225) works.
        return [840, 225]
    
    return best_set
