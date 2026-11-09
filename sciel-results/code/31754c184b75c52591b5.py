"""Cap set greedy construction with sparse-digit (zero-heavy) priority."""
import time

# EVOLVE-BLOCK-START

_DEADLINE_SECONDS = 320.0
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.monotonic()
    return time.monotonic() - _start_time

def _remaining():
    return _DEADLINE_SECONDS - _elapsed()

def priority(vector, n):
    """
    Returns a priority score for a vector in {0,1,2}^n.
    Higher priority means the vector is considered earlier in the greedy selection.
    
    Strategy:
    1. Prioritize vectors with more zeros (lower sum of digits).
       Vectors with many zeros are structurally less prone to forming 3-term APs
       with each other because the midpoint must also respect the zero-coordinates.
       This effectively builds the cap set in shells of increasing Hamming weight.
    2. Break ties using lexicographic order (converted to a rank) to ensure
       deterministic and varied selection.
    """
    # Calculate sum of digits (Hamming weight in a generalized sense for {0,1,2})
    # Smaller sum = more zeros = higher priority.
    digit_sum = sum(vector)
    
    # Calculate lexicographic rank for tie-breaking.
    # We want earlier lex vectors to have slightly higher priority (or lower, as long as consistent).
    # Let's use the rank as a small penalty.
    rank = 0
    for v in vector:
        rank = rank * 3 + v
        
    # The maximum possible rank is 3^n - 1.
    # We normalize the rank to a small range so it doesn't dominate the sum.
    # However, for large n, 3^n is huge. We can just use the rank directly 
    # but scaled down, or simply use the sum as the primary key and rank as secondary.
    # To combine into a single float:
    # Primary: -digit_sum (so smaller sum is larger value)
    # Secondary: -rank (so smaller rank is larger value)
    
    # We need to ensure the secondary term doesn't interfere with the primary.
    # The difference between digit_sum levels is 1.
    # The range of rank is [0, 3^n - 1].
    # If we just return -digit_sum - rank, a vector with sum 0 and high rank might
    # be worse than a vector with sum 1 and low rank if rank difference > 1.
    # We want strict ordering by sum first.
    
    # Let's scale the rank down by a factor much larger than the max possible sum difference?
    # No, we want sum to be the primary key.
    # Priority = -digit_sum * M - rank
    # Where M is large enough such that any difference in digit_sum outweighs any difference in rank.
    # Max rank is ~3^n. This is too big for a float if n is large?
    # n is usually small enough (e.g., 10-20) that 3^n fits in a float, but precision might be an issue.
    # Alternatively, we can just return a tuple? No, interface requires float.
    
    # Let's use a logarithmic or normalized approach for the tie-breaker.
    # Or simply: -digit_sum - (rank / (3**n))
    # This ensures that the difference in digit_sum (integer) is always >= 1,
    # while the rank term is in [0, 1).
    # Thus, any vector with a smaller sum will always have a strictly higher priority
    # than any vector with a larger sum, regardless of rank.
    
    # Calculate 3**n. For large n, this might overflow or lose precision.
    # If n is very large, 3**n is huge. 
    # However, the evaluator calls this for 3^n vectors. If n is large, the time will be dominated by the call count.
    # Let's assume n is manageable (e.g., < 30).
    
    # To be safe against precision issues with large n, we can compute the denominator once?
    # No, n is an argument.
    
    # Optimization: 
    # We can avoid calculating 3**n if we just use a very small epsilon for the tie breaker.
    # But we need strict separation.
    # Let's try: -digit_sum - rank * 1e-12
    # This might not separate ranks if rank is large.
    
    # Better: Use the fact that we just need a total order.
    # If we return -digit_sum, ties are broken arbitrarily by the sort algorithm (stable sort? usually yes, but input order matters).
    # To make it robust, let's mix the rank in a way that is unique enough.
    
    # Let's stick to the normalized rank approach.
    # For n up to ~40, 3**n is around 10^19, which is within float64 precision for integers up to 2^53.
    # 3^33 is approx 5.5 * 10^15. 3^34 is 1.6 * 10^16.
    # So for n < 33, 3**n is an exact integer in float.
    # For larger n, we might lose precision in the rank calculation itself if we do rank = rank*3+v in a loop? No, that's integer math.
    # But dividing by 3**n in float might lose precision.
    
    # Alternative: 
    # Priority = -digit_sum
    # And rely on the fact that the evaluator might sort stably? 
    # The prompt says "added greedily in descending priority". It doesn't specify tie-breaking.
    # To be safe, we should provide a unique priority.
    
    # Let's use a hash-based tie breaker for robustness and speed.
    # But hash can be expensive.
    
    # Let's try the normalized rank again.
    # If n is large, 3**n is huge. 
    # We can compute log(3**n) = n * log(3).
    # But we need the actual value to normalize rank.
    
    # Let's assume n is small enough that 3**n is computable.
    # If n > 40, the number of vectors 3^n is > 10^19, which is impossible to iterate over in 300s.
    # So n will be small (likely < 20).
    
    base = 3 ** n
    # If base is too large, we might have issues. But as argued, n will be small.
    tie_breaker = rank / base if base > 0 else 0.0
    
    return -digit_sum - tie_breaker

# EVOLVE-BLOCK-END
