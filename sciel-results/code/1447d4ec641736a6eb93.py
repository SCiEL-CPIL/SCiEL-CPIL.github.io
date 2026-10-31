import zlib

def priority(vector, n):
    # Calculate counts using C-implemented methods for speed
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Ideal count per symbol
    ideal = n / 3.0
    
    # Deviation from ideal balance
    deviation = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    
    # We want to minimize deviation, so we negate it for "higher is better"
    # Scale it up to dominate the hash tie-breaker
    balance_score = -deviation * 1000.0
    
    # Fast deterministic hash for tie-breaking using zlib.crc32
    # Convert tuple to bytes to satisfy zlib.crc32
    h = zlib.crc32(bytes(vector))
    
    # Normalize hash to [0, 1)
    hash_score = h / 4294967296.0
    
    return float(balance_score + hash_score)
