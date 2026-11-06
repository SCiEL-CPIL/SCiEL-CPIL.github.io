import zlib
import time

# Track coordinate frequencies to estimate conflict risk
_coord_freq = [0] * 3  # count of each value (0, 1, 2) seen so far
_added_count = 0
_start_time = None

def priority(vector, n):
    global _coord_freq, _added_count, _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we've used too much time, just use hash
    elapsed = time.time() - _start_time
    if elapsed > 280:  # Safety margin before 300s limit
        data = bytes(vector)
        return float(zlib.crc32(data))
    
    # Count coordinate frequencies in this vector
    local_freq = [0, 0, 1]  # Initialize with 1 for each value to avoid division by zero
    for x in vector:
        local_freq[x] += 1
    
    # Estimate conflict risk: vectors that share common coordinate values
    # with already-added vectors are more likely to form 3-APs
    # Use a soft penalty based on frequency alignment
    conflict_risk = 0.0
    for i in range(3):
        # If this vector has many of value i, and many added vectors also have value i,
        # then there's higher risk of forming 3-APs
        conflict_risk += local_freq[i] * _coord_freq[i]
    
    # Normalize conflict risk
    if _added_count > 0:
        conflict_risk /= _added_count
    
    # Hash for deterministic ordering
    data = bytes(vector)
    h = zlib.crc32(data)
    
    # Weight the conflict penalty strongly enough to affect ordering
    # but keep hash as primary driver for diversity
    penalty = conflict_risk * 10**8
    
    # Update frequency tracking
    for x in vector:
        _coord_freq[x] += 1
    _added_count += 1
    
    return float(h) - penalty
