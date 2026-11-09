# EVOLVE-BLOCK-START
import hashlib
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0

def priority(vector, n):
    # Check deadline to avoid being killed
    if time.time() > _DEADLINE - 1.0:
        # Return a very low priority to stop adding vectors quickly if we are out of time
        # The evaluator handles the kill, but this ensures we don't hang in loops if called repeatedly
        return -1e9

    # Convert vector to a string representation suitable for hashing
    # Using tuple directly in repr is safe and fast enough for hashing
    v_str = repr(vector)
    
    # Compute MD5 hash
    # MD5 is deterministic and provides a 128-bit digest
    # We take the first 8 bytes (64 bits) to convert to an integer
    # This creates a uniform pseudo-random permutation of the vectors
    h = hashlib.md5(v_str.encode('utf-8')).digest()
    
    # Convert to integer
    # int.from_bytes is efficient
    score_int = int.from_bytes(h[:8], 'big')
    
    # Normalize to a float in [0, 1) or similar range
    # The evaluator sorts by priority descending.
    # We want the hash to determine the order.
    # A larger integer means higher priority.
    # We can just return the integer as a float.
    # 2^64 is approx 1.8e19, which fits in a double precision float with enough precision 
    # to distinguish most 64-bit integers (double has 53 bits of mantissa, so it might collide 
    # for very large numbers, but the relative order will be largely preserved or at least 
    # pseudo-random).
    # To be safer with precision, we can normalize to [0, 1].
    # But if we normalize, we lose the full 64-bit entropy in the float representation.
    # However, the evaluator likely just sorts.
    # Let's return the integer as a float. Python floats can handle integers up to 2^53 exactly.
    # 2^64 is larger than 2^53. So distinct 64-bit hashes might map to the same float.
    # To ensure uniqueness and full entropy usage in sorting, we should ideally return something 
    # that preserves order better.
    # Actually, if two hashes collide in float representation, they get the same priority.
    # The evaluator then breaks ties arbitrarily or by index.
    # Given 3^n vectors, for n up to ~20, 3^20 is ~3e9, which is well within 2^53.
    # So for reasonable n, the float conversion of the 64-bit int is safe and unique.
    
    return float(score_int)
# EVOLVE-BLOCK-END
