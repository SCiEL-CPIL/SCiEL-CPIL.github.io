"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50s margin within 60s limit

    # Precompute modular inverses for denominators 1 to 10 modulo 101
    # inv[x] = x^{-1} mod 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)

    # best_mask[r] stores the bitmask of the smallest subset that sums to r mod 101
    # best_len[r] stores the length of that subset
    best_mask = [0] * 101
    best_len = [1000] * 101  # Initialize with a large number

    # Iterate over all non-empty subsets of {1, ..., 10}
    # mask ranges from 1 to (1 << 10) - 1
    for mask in range(1, 1 << 10):
        # Check deadline
        if time.time() > deadline:
            break

        terms = []
        s = 0
        # Calculate sum of inverses and length for this mask
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                s = (s + inv[x]) % 101
                terms.append(x)
        
        length = len(terms)
        
        # If this subset is shorter than the current best for this residue, update
        if length < best_len[s]:
            best_len[s] = length
            best_mask[s] = mask
            # Optional: store the terms directly to avoid reconstruction later
            # But storing mask is fine, we can reconstruct.
            # Actually, let's store the list to be safe and simple, or just the mask.
            # Storing mask is memory efficient. We'll reconstruct at the end.

    # If the deadline was hit, we might have incomplete coverage.
    # However, with 1023 subsets, the loop finishes in microseconds.
    # The deadline is a safety net.

    # Construct the result
    result = []
    for r in range(101):
        if best_len[r] == 1000:
            # This should not happen as coverage is expected.
            # Fallback: return an empty list or raise error?
            # Task says "Every residue must be represented".
            # If not found, we have a problem. But mathematically it should be covered.
            # For robustness, if not found, we could do a random search, but it's not needed here.
            result.append([])
        else:
            # Reconstruct terms from mask
            mask = best_mask[r]
            terms = [x for x in range(1, 11) if mask & (1 << (x - 1))]
            result.append(terms)
            
    return result
# EVOLVE-BLOCK-END
