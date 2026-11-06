"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find 101 nonempty lists of distinct denominators from 1..10
    such that the sum of modular inverses modulo 101 equals the row index.
    Minimize the maximum row length.
    """
    start_time = time.time()
    deadline = start_time + 80  # Safety margin: evaluator kills at 90s
    
    # Precompute modular inverses for denominators 1 to 10 modulo 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)
    
    # Generate all non-empty subsets of {1, ..., 10}
    # There are 2^10 - 1 = 1023 such subsets
    all_subsets = []
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for x in range(1, 11):
            if mask >> (x - 1) & 1:
                terms.append(x)
                s = (s + inv[x]) % 101
        all_subsets.append((s, terms))
    
    # Group subsets by their residue
    # residue_to_subsets[r] = list of (length, terms) for residue r
    residue_to_subsets = [[] for _ in range(101)]
    for residue, terms in all_subsets:
        residue_to_subsets[residue].append((len(terms), terms))
    
    # For each residue, sort subsets by length (shorter first)
    for r in range(101):
        residue_to_subsets[r].sort(key=lambda x: x[0])
    
    # Check if any residue has no representation
    for r in range(101):
        if not residue_to_subsets[r]:
            # This shouldn't happen, but handle gracefully
            pass
    
    # Strategy: Try increasing max_length L starting from 1
    # For a given L, we need to assign to each residue r a subset of length <= L
    # such that all residues are covered.
    # Since each subset covers exactly one residue, this is just:
    # for each residue r, pick the shortest subset (length <= L) that sums to r.
    # The question is: what is the minimum L such that every residue has at least
    # one subset of length <= L?
    
    # Compute the minimum subset length needed for each residue
    min_len_for_residue = [float('inf')] * 101
    for r in range(101):
        if residue_to_subsets[r]:
            min_len_for_residue[r] = residue_to_subsets[r][0][0]
    
    # The maximum of min_len_for_residue over all residues is the answer
    max_min_len = max(min_len_for_residue)
    
    # Now construct the result: for each residue, use the shortest subset
    result = [None] * 101
    for r in range(101):
        if residue_to_subsets[r]:
            result[r] = residue_to_subsets[r][0][1]
        else:
            # Fallback: should not happen
            result[r] = [1]
    
    # Verify the solution
    for r in range(101):
        if not result[r]:
            raise ValueError(f"Row {r} is empty")
        # Check distinct denominators
        if len(result[r]) != len(set(result[r])):
            raise ValueError(f"Row {r} has duplicate denominators")
        # Check denominators in range 1..10
        for x in result[r]:
            if x < 1 or x > 10:
                raise ValueError(f"Row {r} has denominator {x} out of range")
        # Check sum of inverses
        s = sum(inv[x] for x in result[r]) % 101
        if s != r:
            raise ValueError(f"Row {r}: expected residue {r}, got {s}")
    
    return result
# EVOLVE-BLOCK-END
