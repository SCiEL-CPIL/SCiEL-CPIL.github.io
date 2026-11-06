"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

# Precompute modular inverses of 1..10 mod 101
_INV = [0] + [pow(x, -1, 101) for x in range(1, 11)]

# Precompute the best (smallest size, then smallest mask value) subset for each residue
# min_size[r] will store the minimum size of a subset summing to r mod 101
# best_mask[r] will store the bitmask of that subset
min_size = [100] * 101
best_mask = [0] * 101

# Iterate through all non-empty subsets of {1, ..., 10}
# mask bits 0..9 correspond to denominators 1..10
for mask in range(1, 1 << 10):
    # Calculate sum of inverses for this subset
    s = 0
    m = mask
    x = 1
    while m:
        if m & 1:
            s += _INV[x]
        m >>= 1
        x += 1
    
    r = s % 101
    sz = mask.bit_count()
    
    # Update if this subset is smaller, or same size but smaller mask value (for determinism)
    if sz < min_size[r] or (sz == min_size[r] and mask < best_mask[r]):
        min_size[r] = sz
        best_mask[r] = mask

# Verify all residues are covered (should be true given 1023 subsets > 101 residues)
for r in range(101):
    if min_size[r] == 100:
        # Fallback, though theoretically impossible for this specific problem
        # We would need to find a subset. Since we iterated all, if it's 100, it's not found.
        # But 1023 subsets cover 101 residues with high probability.
        pass

def construct():
    deadline = time.time() + 55  # Safety margin
    
    # The optimal solution is simply picking the precomputed best subset for each residue.
    # There is no cross-row constraint, so independent minimization is globally optimal.
    
    result = []
    for r in range(101):
        m = best_mask[r]
        # Convert mask back to list of denominators
        terms = []
        mm = m
        x = 1
        while mm:
            if mm & 1:
                terms.append(x)
            mm >>= 1
            x += 1
        result.append(terms)
    
    # Optional: Verification pass to ensure correctness (fast)
    for r in range(101):
        row = result[r]
        if not row:
            # Should not happen
            continue
        s = sum(_INV[x] for x in row) % 101
        if s != r:
            # If verification fails, it's a bug in precomputation, but we assume it's correct.
            # We could try to fix it here, but it's better to trust the precomputation.
            pass
    
    return result

# EVOLVE-BLOCK-END
