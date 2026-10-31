import hashlib

def priority(vector, n):
    # Convert vector to bytes for hashing
    # vector is a tuple of ints in {0, 1, 2}
    # We can pack them into a bytes object
    # n is the dimension
    # For efficiency, we can just hash the tuple directly, but 
    # hashing a tuple of ints is fast enough for n up to ~10-15.
    # However, to be safe and consistent with the "inspiration" which 
    # used a 64-bit slice of SHA256, we will do that.
    
    # Create a bytes representation
    # Each element is 0, 1, or 2. We can map to bytes 0, 1, 2.
    data = bytes(vector)
    
    # Hash it
    hash_obj = hashlib.sha256(data).digest()
    
    # Take the first 8 bytes (64 bits) and convert to an integer
    # This provides a 64-bit pseudo-random priority
    # We return it as a float to match the interface
    # Using a 64-bit integer as a float is safe (float has 53 bits of precision, 
    # but we only need relative ordering. If we convert to float, we might lose 
    # precision for large ints. 
    # Better: return the integer part? No, interface requires float.
    # We can scale it down or just return the int as a float.
    # Python floats can represent integers exactly up to 2^53.
    # 2^64 is larger. So we might lose precision.
    # To avoid precision loss affecting order, we can take a 48-bit slice 
    # or just use the first 53 bits.
    # Or, we can return a float in [0, 1) derived from the hash.
    
    # Let's take the first 8 bytes, convert to int, then normalize to [0, 1)
    # h = int.from_bytes(hash_obj[:8], 'big')
    # return h / (1 << 64)
    
    # This ensures a uniform distribution in [0, 1) and avoids precision issues 
    # with large integers if the evaluator sorts them.
    
    h = int.from_bytes(hash_obj[:8], 'big')
    return h / (1 << 64)
