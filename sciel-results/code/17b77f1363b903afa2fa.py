import math

def construct():
    """
    Difference-sparse construction: U is a subset of a short arithmetic progression
    with a large common difference. We optimize the gap size to balance the linear
    growth of the difference set against the quadratic growth of the sumset.
    """
    best_score = 0
    best_set = [0, 1, 3, 7, 15]
    
    def calc_score(u):
        """Calculate the score for a given set u."""
        if not u:
            return 0
        max_u = u[-1]
        
        # Use bitmasks for efficient computation
        sum_mask = 0
        diff_mask = 0
        
        for i in u:
            for j in u:
                sum_mask |= (1 << (i + j))
                diff_mask |= (1 << (i - j + max_u))
        
        num_sums = bin(sum_mask).count('1')
        num_diffs = bin(diff_mask).count('1')
        
        if num_sums == 0:
            return 0
            
        ratio = num_diffs / num_sums
        if ratio <= 1:
            return 0
            
        denom = math.log2(2 * max_u + 1)
        if denom == 0:
            return 0
            
        return 1 + math.log2(ratio) / denom

    # Strategy: Try arithmetic progressions with various parameters
    # U = {a*i : i in S} where S is a subset of {0, 1, ..., k}
    # The key insight: if we use a large gap 'a', the differences scale with 'a'
    # but so do the sums. The ratio |U-U|/|U+U| depends on the structure.
    
    # For an arithmetic progression {0, a, 2a, ..., ka}:
    # U+U = {0, a, ..., 2ka} -> size 2k+1
    # U-U = {-ka, ..., ka} -> size 2k+1
    # Ratio = 1, which gives score = 1. Not good.
    
    # We need a "difference-sparse" set. Let's try sets where the differences
    # are more spread out than the sums.
    
    # Consider U = {0, a, 2a, ..., (k-1)a, ka + b} for some offset b
    # Or better: use a set with a large gap in the middle.
    
    # Actually, the key is to make |U-U| large relative to |U+U|.
    # For a set with max element M, |U-U| <= 2M+1 and |U+U| <= 2M+1.
    # To maximize the ratio, we want |U-U| close to 2M+1 and |U+U| as small as possible.
    
    # A set with large gaps can have |U+U| much smaller than 2M+1 while |U-U| 
    # remains close to 2M+1.
    
    # Let's try: U = {0, a, 2a, ..., (n-1)a} where a is large
    # This gives |U+U| = 2n-1, |U-U| = 2n-1, ratio = 1. Bad.
    
    # Instead, try: U = {0, a, 2a, ..., (n-2)a, M} where M >> (n-1)a
    # Then U+U has elements from {0, a, ..., (2n-3)a} plus {M, M+a, ..., M+(n-2)a}
    # |U+U| ≈ (2n-2) + (n-1) = 3n-3 (if no overlaps)
    # U-U has elements from {-(n-2)a, ..., (n-2)a} plus {M-ja : j=0..n-2} and {-(M-ja) : j=0..n-2}
    # |U-U| ≈ (2n-3) + 2(n-1) = 4n-5 (if no overlaps)
    # Ratio ≈ (4n-5)/(3n-3) ≈ 4/3 for large n
    
    # Let's try this construction with various n and gap sizes
    
    for n in range(3, 20):
        for a in range(1, 50):
            # Try M = k * a + offset for various k
            for k in range(n, n + 5):
                for offset in range(0, a):
                    M = k * a + offset
                    if M <= (n-1) * a:
                        continue
                    u = [i * a for i in range(n-1)] + [M]
                    u.sort()
                    s = calc_score(u)
                    if s > best_score:
                        best_score = s
                        best_set = u
    
    # Also try the original exponential-like sets
    for base in range(2, 6):
        for n in range(3, 15):
            u = [base ** i for i in range(n)]
            s = calc_score(u)
            if s > best_score:
                best_score = s
                best_set = u
    
    return best_set
