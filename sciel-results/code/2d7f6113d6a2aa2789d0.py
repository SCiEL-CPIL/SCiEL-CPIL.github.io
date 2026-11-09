"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    MOD = 101
    # Precompute modular inverses of 1..10 mod 101
    inv = [0] + [pow(x, -1, MOD) for x in range(1, 11)]
    
    # We need to find for each residue r, the minimum size subset S of {1..10}
    # such that sum(inv[x] for x in S) == r mod 101.
    # Then the objective is the maximum of these minimum sizes.
    
    # Initialize min_size and best_mask for each residue
    min_size_for_res = [11] * MOD
    best_mask_for_res = [0] * MOD
    
    # Iterate through all non-empty subsets of {1..10}
    # There are 2^10 - 1 = 1023 such subsets.
    for mask in range(1, 1 << 10):
        s = 0
        sz = 0
        # Calculate sum of inverses and size for this subset
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                s = (s + inv[x]) % MOD
                sz += 1
        
        r = s
        # If this subset is smaller than the current best for residue r, update
        if sz < min_size_for_res[r]:
            min_size_for_res[r] = sz
            best_mask_for_res[r] = mask

    # The optimal max row length is the max of the min sizes
    # We don't strictly need to return this value, but it's good for verification
    # best_k = max(min_size_for_res)
    
    # Construct the result lists
    result = []
    for r in range(MOD):
        m = best_mask_for_res[r]
        # Assert that a solution was found
        assert m != 0, f"Residue {r} not covered"
        
        # Convert mask to list of denominators
        row = [x for x in range(1, 11) if m & (1 << (x - 1))]
        result.append(row)
        
        # Verification
        assert len(row) >= 1
        assert len(set(row)) == len(row)
        s = sum(inv[x] for x in row) % MOD
        assert s == r, f"Row {r}: sum {s} != {r}"

    return result
# EVOLVE-BLOCK-END
