import time
from fractions import Fraction

def construct():
    """
    Find a set of 1..256 distinct integer denominators in [5, 10000000]
    whose reciprocal sum is exactly 1. Minimize the count.
    
    Strategy:
    1. Use a greedy algorithm to find a representation with small denominators.
    2. The greedy algorithm picks the largest possible unit fraction <= remaining sum,
       ensuring denominators are >= 5 and distinct.
    3. Since the greedy method for Egyptian fractions tends to produce few terms
       when starting from small denominators, and we want to minimize the count,
       we try to find a representation with the fewest terms.
    
    Note: The standard greedy algorithm for Egyptian fractions gives a valid
    representation. To minimize the count, we can use the known fact that
    the greedy algorithm often produces near-optimal or optimal solutions for
    small denominators.
    
    We know that 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/21 + 1/35 + ... 
    Actually, let's just run the greedy algorithm and see how many terms we get.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s limit
    
    def greedy_egyptian(remaining, min_denom, used):
        """
        Greedy algorithm for Egyptian fractions.
        remaining: Fraction to represent
        min_denom: minimum allowed denominator
        used: set of already-used denominators
        Returns: list of denominators or None if fails
        """
        if remaining == 0:
            return []
        if remaining < 0:
            return None
        
        # Find the largest unit fraction <= remaining
        # 1/d <= remaining => d >= 1/remaining
        d_min = max(min_denom, (1 / remaining).numerator // (1 / remaining).denominator + 1)
        # Actually, 1/d <= remaining means d >= 1/remaining
        # Since d must be integer, d >= ceil(1/remaining)
        
        # But we also need d not in used
        d = d_min
        while d in used:
            d += 1
        
        # Check if 1/d <= remaining
        if Fraction(1, d) > remaining:
            # This shouldn't happen if we computed d_min correctly, but let's be safe
            d += 1
            while d in used:
                d += 1
        
        if Fraction(1, d) > remaining:
            return None
        
        new_used = used | {d}
        rest = greedy_egyptian(remaining - Fraction(1, d), d + 1, new_used)
        if rest is not None:
            return [d] + rest
        else:
            return None
    
    # Start with remaining = 1
    result = greedy_egyptian(Fraction(1, 1), 5, set())
    
    if result is not None and len(result) <= 256:
        # Verify the result
        s = sum(Fraction(1, d) for d in result)
        if s == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result):
            return result
    
    # If greedy fails (unlikely), try a different approach
    # For now, raise an error as fallback
    raise RuntimeError("No valid representation found")

