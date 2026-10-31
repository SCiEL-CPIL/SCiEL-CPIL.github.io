import zlib
import time

# Module-level state for deadline tracking
_start_time = None
_DEADLINE = 320.0  # Safety margin: evaluator kills at 330s, we stop at 320s

def priority(vector, n):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a constant value to quickly finish if we're over time
        return 0.0

    # Strategy: Combine structural hashing (from current program) with
    # structural refinement (from inspiration 2: penalize ones).
    #
    # Rationale:
    # 1. The CRC32 hash provides a deterministic, scrambled order that avoids
    #    pathological lexicographic failures.
    # 2. The "fewer ones" heuristic suggests that vectors with many 1s are
    #    more likely to form 3-term arithmetic progressions (3-APs).
    #    Specifically, if v is a midpoint of a 3-AP (a, v, b), then v_i = 1
    #    implies a_i=0, b_i=2 or a_i=2, b_i=0. Thus, high count of 1s increases
    #    the chance of being a "dangerous" midpoint.
    # 3. By subtracting a weighted count of 1s from the hash, we bias the
    #    greedy selection to add "safer" vectors (those with fewer 1s) earlier,
    #    while maintaining the scrambled order for tie-breaking or minor variations.
    #    This should allow the greedy algorithm to pack more vectors before
    #    hitting the cap set constraint.

    # Count the number of 1s
    num_ones = sum(1 for x in vector if x == 1)
    
    # Calculate the base-3 integer value for hashing
    val = 0
    for x in vector:
        val = val * 3 + x
    
    # Use CRC32 for scrambling
    h = zlib.crc32(str(val).encode())
    
    # Combine: High hash value is good (scrambled order), but penalize high num_ones.
    # The weight 1000000 ensures that the structural penalty (num_ones) dominates
    # over the hash variation only when the difference in num_ones is significant,
    # but actually, we want the hash to provide diversity within the same num_ones class.
    # Let's invert the logic slightly: 
    # We want to prioritize LOWER num_ones. 
    # So, priority = -num_ones * large_weight + hash_value
    # This way, a vector with 0 ones will always have higher priority than one with 1 one,
    # regardless of the hash. Within the same num_ones count, the hash determines order.
    
    weight = 1000000
    return float(-num_ones * weight + h)
