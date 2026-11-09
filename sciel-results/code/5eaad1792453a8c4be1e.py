"""Find the maximum interval in [1,1008] with no modular inverse pair (including self-inverse)."""
# EVOLVE-BLOCK-START
def construct():
    mod = 1009
    n = 1008  # domain [1, 1008]
    
    # Precompute modular inverses
    inv = [0] * (mod + 1)
    for x in range(1, mod):
        inv[x] = pow(x, mod - 2, mod)
    
    # Self-inverse elements 1 and 1008 are always forbidden because x*x % 1009 == 1.
    # So the interval must be within [2, 1007].
    
    # An index x is "forbidden" if it is the larger element of an inverse pair {x, inv[x]}.
    # i.e., inv[x] < x.
    # If an interval contains no such x, it cannot contain both elements of any pair.
    # Proof: For any pair {a, b} with a < b, inv[b] = a < b, so b is forbidden.
    # If the interval avoids all forbidden indices, it avoids b. Thus it cannot contain both a and b.
    
    forbidden = []
    for x in range(2, 1008):
        if inv[x] < x:
            forbidden.append(x)
            
    # The forbidden indices split the domain [2, 1007] into segments.
    # We need to find the longest segment that does not contain any forbidden index.
    # The segments are defined by the gaps between consecutive forbidden indices,
    # and the edges [2, first_forbidden-1] and [last_forbidden+1, 1007].
    
    if not forbidden:
        return [2, 1006]
        
    forbidden.sort()
    
    best_len = 0
    best_start = 2
    
    # Check gap before first forbidden
    prev = 1 # Boundary
    for f in forbidden:
        # Interval [prev+1, f-1] is valid
        start = prev + 1
        end = f - 1
        if start <= end:
            length = end - start + 1
            if length > best_len:
                best_len = length
                best_start = start
        prev = f
        
    # Check gap after last forbidden
    start = prev + 1
    end = 1007
    if start <= end:
        length = end - start + 1
        if length > best_len:
            best_len = length
            best_start = start
            
    if best_len == 0:
        # Fallback, though there should always be a gap of at least 1 unless forbidden covers all
        return [2, 1]
        
    return [best_start, best_len]
# EVOLVE-BLOCK-END
