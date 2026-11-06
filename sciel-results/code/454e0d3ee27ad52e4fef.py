"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [pow(x, -1, 101) for x in range(1, 11)]
    
    # Precompute all non-empty subsets: (size, residue, mask)
    all_subsets = []
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        size = len(terms)
        residue = sum(inv[x - 1] for x in terms) % 101
        all_subsets.append((size, residue, mask))
    
    # Group subsets by residue
    by_residue = {}
    for size, residue, mask in all_subsets:
        by_residue.setdefault(residue, []).append((size, mask))
    
    # For each residue, find the minimum subset size needed
    min_size_for_residue = {}
    for r in range(101):
        if r in by_residue:
            min_size_for_residue[r] = min(s for s, m in by_residue[r])
        else:
            min_size_for_residue[r] = 10  # fallback
    
    # The global maximum of these minima is our target max row length
    target_max_len = max(min_size_for_residue.values())
    
    # Now, for each residue, pick a subset of size <= target_max_len
    # To minimize the largest row length, we want all rows to have size <= target_max_len
    # Since target_max_len is the max of the minimums, each residue has at least one subset of size <= target_max_len
    
    result = []
    for r in range(101):
        # Find a subset of size <= target_max_len for this residue
        found = False
        for size, mask in sorted(by_residue.get(r, []), key=lambda x: x[0]):
            if size <= target_max_len:
                terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
                result.append(terms)
                found = True
                break
        if not found:
            # Should not happen, but fallback
            result.append([1])
    
    return result
# EVOLVE-BLOCK-END
