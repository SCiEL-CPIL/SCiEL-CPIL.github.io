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
    all_subsets = []
    for mask in range(1, 1 << n):
        s = 0
        size = 0
        for i in range(1, n + 1):
            if (mask >> (i - 1)) & 1:
                s = (s + inv[i]) % mod
                size += 1
        all_subsets.append((mask, s, size))
    
    # Sort by size, then by mask for determinism
    all_subsets.sort(key=lambda x: (x[2], x[0]))
    
    # Find minimum k such that subsets of size <= k cover all residues
    # For each residue, find the smallest (size, mask) subset that sums to it
    best_for_residue = [None] * mod  # (size, mask) for each residue
    
    # Group subsets by size
    by_size = {}
    for mask, s, size in all_subsets:
        if size not in by_size:
            by_size[size] = []
        by_size[size].append((mask, s))
    
    # Try increasing k
    max_k = n
    for k in range(1, max_k + 1):
        # Check if subsets of size <= k cover all residues
        covered = [False] * mod
        for size in range(1, k + 1):
            if size not in by_size:
                continue
            for mask, s in by_size[size]:
                if not covered[s]:
                    covered[s] = True
                    # Record this as candidate (we'll pick smallest later)
                    if best_for_residue[s] is None or size < best_for_residue[s][0]:
                        best_for_residue[s] = (size, mask)
        
        if all(covered):
            break
    
    # Now assign: for each residue, pick the subset with smallest size, then smallest mask
    # Re-scan all_subsets sorted by (size, mask) to get the best assignment
    final_assignment = [None] * mod
    for mask, s, size in all_subsets:
        if final_assignment[s] is None:
            final_assignment[s] = (mask, size)
    
    # Construct the result rows
    rows = []
    for r in range(mod):
        if final_assignment[r] is None:
            # Should not happen if problem is valid
            rows.append([1])  # fallback
        else:
            mask, size = final_assignment[r]
            terms = []
            for i in range(1, n + 1):
                if (mask >> (i - 1)) & 1:
                    terms.append(i)
            rows.append(terms)
    
    return rows
# EVOLVE-BLOCK-END
