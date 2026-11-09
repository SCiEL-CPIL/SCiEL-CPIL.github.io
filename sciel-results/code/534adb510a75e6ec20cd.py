import numpy as np

def priority(vector, n):
    # Convert to numpy array for fast operations
    v = np.array(vector)
    
    # Hamming weight
    w = np.sum(v != 0)
    
    # If zero vector, very low priority
    if w == 0:
        return -1000.0
        
    # Deterministic hash for tie-breaking
    # Use a simple hash based on the vector content
    # We want to break ties in a way that might avoid 3-APs
    # A simple way is to use the vector itself as a key for a hash
    # But we need it to be fast.
    
    # Let's try a simple linear combination with random-looking coefficients
    # But it must be deterministic.
    
    # Alternative: Just use -w. 
    # Does tie-breaking matter?
    # If we don't break ties, it's lexicographic.
    # Lexicographic might be bad.
    
    # Let's try a "balanced" priority.
    # Sum of coordinates?
    s = np.sum(v) % 3
    
    # Maybe prioritize vectors with sum 0?
    # But we established that weight is more important.
    
    # Let's use a hash of the vector to add a small perturbation.
    # Python's hash is salted per process, so we need a stable hash.
    # We can use zlib.crc32 or similar.
    
    import zlib
    # Convert vector to bytes
    b = bytes(v)
    h = zlib.crc32(b)
    
    # Normalize hash to [0, 1)
    perturbation = (h / 2**32) * 0.1
    
    return -w + perturbation
