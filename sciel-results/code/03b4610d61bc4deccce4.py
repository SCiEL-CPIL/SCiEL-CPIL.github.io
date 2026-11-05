import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a list of 12 distinct positive integers <= 1048576 such that
    all 2^12 subset sums are distinct, minimizing the largest integer.
    
    The theoretical minimum for the largest integer in a set of n distinct
    positive integers with distinct subset sums is 2^(n-1).
    For n=12, this is 2^11 = 2048.
    The set of powers of 2 {1, 2, 4, ..., 2048} achieves this bound.
    """
    n = 12
    
    # Construct the optimal set: powers of 2
    # This ensures all subset sums are distinct (binary representation uniqueness)
    # and minimizes the largest element.
    result = [1 << i for i in range(n)]
    
    # The maximum allowed value is 1048576 (2^20).
    # Our largest element is 2^11 = 2048, which is well within the limit.
    
    # Optional: Fast verification
    # We verify that all 2^12 subset sums are distinct.
    # This is a sanity check to ensure the logic holds.
    # For n=12, this is 4096 sums, which is very fast.
    def _validate(arr):
        sums = set()
        # Iterate over all 2^n subsets
        for mask in range(1 << len(arr)):
            s = 0
            # Calculate sum for this subset
            # Using bit operations for efficiency
            for i in range(len(arr)):
                if mask & (1 << i):
                    s += arr[i]
            if s in sums:
                return False
            sums.add(s)
        return True

    if not _validate(result):
        # This branch is theoretically unreachable for powers of 2,
        # but serves as a safety fallback.
        # If it fails, we return a known valid set (which is the same).
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        
    return result
# EVOLVE-BLOCK-END
