import zlib

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Count occurrences of 0, 1, 2
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Score based on balance: maximize product c0*c1*c2
    # This rewards vectors with a uniform distribution of values.
    balance_score = float(c0 * c1 * c2)
    
    # Tie-breaker using CRC32 to ensure deterministic but pseudo-random order
    # among vectors with the same balance score.
    # This prevents fallback to lexicographic order if the greedy sort is stable
    # and the input is lexicographic.
    b = bytes(vector)
    h = zlib.crc32(b)
    
    # Scale the hash to a small range to act as a tie-breaker
    # 1e-6 ensures it doesn't override the integer balance score difference
    # (which is at least 1 for distinct products, though products can be close).
    # Actually, c0*c1*c2 can be large.
    # For n=20, max product is (6*7*7) ~ 294.
    # So 1e-6 is very small.
    tie_breaker = (h % 1000000) * 1e-6
    
    return balance_score + tie_breaker
# EVOLVE-BLOCK-END
