"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    p = 101
    max_denom = 10
    
    # Precompute modular inverses
    inv = [pow(x, -1, p) for x in range(1, max_denom + 1)]
    
    # Precompute all non-empty subsets of {1..10} with their residue and size
    # There are 2^10 - 1 = 1023 subsets
    all_subsets = []  # list of (residue, size, terms_tuple)
    for mask in range(1, 1 << max_denom):
        terms = tuple(x for x in range(1, max_denom + 1) if (mask >> (x - 1)) & 1)
        residue = sum(inv[x - 1] for x in terms) % p
        size = len(terms)
        all_subsets.append((residue, size, terms))
    
    # Find the minimum k such that all 101 residues can be covered
    # For each k, check if all residues are achievable by some subset of size <= k
    # Then for each residue, pick the smallest-size subset that achieves it
    
    # First, determine the minimal k
    for k in range(1, max_denom + 1):
        covered = set()
        for residue, size, terms in all_subsets:
            if size <= k:
                covered.add(residue)
        if len(covered) == p:
            min_k = k
            break
    else:
        min_k = max_denom
    
    # Now, for each residue, find the best (smallest size, then lexicographic) subset
    # We want to minimize the maximum row length, which is min_k
    # But we also want each row to be as short as possible
    
    best_for_residue = {}
    for residue, size, terms in all_subsets:
        if size <= min_k:
            if residue not in best_for_residue:
                best_for_residue[residue] = (size, terms)
            else:
                # Prefer smaller size, then lexicographically smaller
                if size < best_for_residue[residue][0]:
                    best_for_residue[residue] = (size, terms)
                elif size == best_for_residue[residue][0] and terms < best_for_residue[residue][1]:
                    best_for_residue[residue] = (size, terms)
    
    # Check if all residues are covered
    if len(best_for_residue) < p:
        # Fallback: use any subset
        for residue, size, terms in all_subsets:
            if residue not in best_for_residue:
                best_for_residue[residue] = (size, terms)
    
    # Build the result
    result = []
    for r in range(p):
        if r in best_for_residue:
            result.append(list(best_for_residue[r][1]))
        else:
            # Should not happen, but fallback
            result.append([1])
    
    return result
# EVOLVE-BLOCK-END
