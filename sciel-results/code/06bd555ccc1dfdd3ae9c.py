"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_START = time.time()
_DEADLINE = 82.0  # safety margin under the 90s kill limit


def _elapsed():
    return time.time() - _START


def construct():
    """
    Constructs 101 lists of denominators (1-10) such that the sum of their
    modular inverses (mod 101) equals the list index.
    Minimizes the maximum length of any list.
    Precomputes all 2^10 - 1 non-empty subsets, finds the minimum max-size k
    that covers all 101 residues, then assigns the smallest valid subset per residue.
    """
    n = 10
    mod = 101
    
    # Precompute modular inverses
    inv = [0] * (n + 1)
    for i in range(1, n + 1):
        inv[i] = pow(i, -1, mod)
    
    # Precompute all non-empty subsets: (mask, sum_mod, size)
    # Use dynamic programming over masks for speed
    all_subsets = []
    subset_sum = [0] * (1 << n)
    subset_size = [0] * (1 << n)
    
    for mask in range(1, 1 << n):
        # Get lowest set bit
        lsb = mask & (-mask)
        bit_idx = lsb.bit_length() - 1  # 0-indexed
        prev_mask = mask ^ lsb
        subset_sum[mask] = (subset_sum[prev_mask] + inv[bit_idx + 1]) % mod
        subset_size[mask] = subset_size[prev_mask] + 1
        all_subsets.append((mask, subset_sum[mask], subset_size[mask]))
    
    # Sort by size, then by mask for determinism
    all_subsets.sort(key=lambda x: (x[2], x[0]))
    
    # For each residue, find the smallest (size, mask) subset that sums to it
    # This automatically gives us the minimum max-size k needed
    best_for_residue = [None] * mod  # (size, mask) for each residue
    
    for mask, s, size in all_subsets:
        if best_for_residue[s] is None:
            best_for_residue[s] = (size, mask)
        else:
            # Since sorted by (size, mask), the first one found is optimal
            break
    
    # Actually, we need to be more careful. Let's re-scan properly.
    # Since all_subsets is sorted by (size, mask), the first subset we see for
    # each residue is the one with minimum size, and among those, minimum mask.
    best_for_residue = [None] * mod
    for mask, s, size in all_subsets:
        if best_for_residue[s] is None:
            best_for_residue[s] = (size, mask)
    
    # Construct the result rows
    rows = []
    for r in range(mod):
        if best_for_residue[r] is None:
            # Should not happen if problem is valid
            rows.append([1])  # fallback
        else:
            size, mask = best_for_residue[r]
            terms = []
            for i in range(1, n + 1):
                if (mask >> (i - 1)) & 1:
                    terms.append(i)
            rows.append(terms)
    
    return rows
# EVOLVE-BLOCK-END
