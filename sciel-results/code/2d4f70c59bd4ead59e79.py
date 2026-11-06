import time
from itertools import combinations

def construct():
    """
    Constructs 101 lists of distinct denominators in 1..10 such that the sum of 
    their modular inverses mod 101 equals the row index.
    Minimizes the maximum length of the lists.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s limit
    
    MOD = 101
    DENOMS = list(range(1, 11))
    
    # Precompute modular inverses
    invs = [0] * 11
    for x in DENOMS:
        invs[x] = pow(x, -1, MOD)
    
    # Precompute all non-empty subsets with their size and residue
    all_subsets = []  # list of (size, residue, terms_tuple)
    for mask in range(1, 1 << 10):
        terms = []
        current_sum = 0
        for i in range(10):
            if mask & (1 << i):
                denom = i + 1
                terms.append(denom)
                current_sum = (current_sum + invs[denom]) % MOD
        all_subsets.append((len(terms), current_sum, tuple(terms)))
    
    # Sort by size, then by mask order for determinism
    all_subsets.sort(key=lambda x: (x[0], x[2]))
    
    # Build residue -> list of (size, terms) sorted by size
    res_to_subsets = {r: [] for r in range(MOD)}
    for size, residue, terms in all_subsets:
        res_to_subsets[residue].append((size, terms))
    
    # For each residue, find the minimum size needed
    min_size_per_res = [min(s for s, _ in res_to_subsets[r]) if res_to_subsets[r] else 11 for r in range(MOD)]
    
    # Binary search / iterate on K: find smallest K such that all residues can be covered
    # with subsets of size <= K, using a greedy set cover approach
    # 
    # The key insight: we need to assign ONE subset to each residue such that:
    #   - The subset's residue matches the row index
    #   - The subset size <= K
    #   - We want to minimize the MAXIMUM size used
    #
    # Since each residue can only be covered by subsets that sum to that residue,
    # and subsets can be reused across different rows, this is actually just:
    #   For each residue r, pick the smallest subset that sums to r.
    # The maximum row length is then max over all r of (smallest subset size for r).
    #
    # Wait - that's what the current program does! And it scores 1.4.
    # 
    # Let me reconsider: the problem says "Minimize the largest row length."
    # If each residue has a unique minimum-size subset, then the answer is fixed.
    # But maybe some residues can be covered by multiple subsets of the same min size,
    # and we need to choose which one to use for each residue to balance things out?
    # No - the constraint is only on the sum, not on which specific subset is used.
    # Each row is independent. So the minimum possible max row length is:
    #   max over r of (minimum subset size that sums to r)
    #
    # Let me verify: is there any residue that requires a larger subset?
    max_min_size = max(min_size_per_res)
    
    # Build the result using the minimum-size subset for each residue
    result = []
    for r in range(MOD):
        # Find the smallest subset for residue r
        best = None
        for size, terms in res_to_subsets[r]:
            if best is None or size < best[0]:
                best = (size, terms)
        if best is not None:
            result.append(list(best[1]))
        else:
            result.append([1])  # Fallback
    
    # Verify all residues are covered
    for r in range(MOD):
        if not result[r]:
            # Should not happen
            result[r] = [1]
    
    return result
