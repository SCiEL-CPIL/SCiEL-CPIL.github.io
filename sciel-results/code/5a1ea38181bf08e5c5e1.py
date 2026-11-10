"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin within 90s limit
    
    p = 101
    # Precompute inverses for 1..10 modulo 101
    inverses = [pow(x, -1, p) for x in range(1, 11)]
    
    # Generate all non-empty subsets of {1, ..., 10} with their sums and lengths
    # subset: list of denominators
    # There are 2^10 - 1 = 1023 subsets
    subsets = []
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for i in range(10):
            if mask & (1 << i):
                denom = i + 1
                terms.append(denom)
                s += inverses[i]
        residue = s % p
        length = len(terms)
        subsets.append((length, residue, terms))
    
    # Sort subsets by length (shortest first)
    subsets.sort(key=lambda x: x[0])
    
    # Group subsets by length for efficient access
    # subsets_by_length[k] = list of (residue, terms) for subsets of length k
    subsets_by_length = defaultdict(list)
    for length, residue, terms in subsets:
        subsets_by_length[length].append((residue, terms))
    
    # Find the minimum k such that we can assign all 101 residues
    # using only subsets of size <= k, with each residue getting exactly one subset
    # and no subset used twice (since each subset has a unique residue sum, 
    # but multiple subsets can have the same residue)
    
    # Actually, we need to find a minimum-maximum-length assignment.
    # For a given k, we have all subsets of size <= k available.
    # We need to assign one subset to each of the 101 residues such that
    # the subset's residue matches the target residue.
    # Multiple subsets can map to the same residue, so we just need at least
    # one available subset for each residue.
    
    # Wait - the constraint is that denominators are distinct within a row,
    # but can recur across rows. So each row just needs one valid subset.
    # The question is: for a given k, is there at least one subset of size <= k
    # for each residue r in 0..100?
    
    # Let's check: for each k from 1 to 10, check if all 101 residues are covered
    # by subsets of size <= k.
    
    best_k = 10
    best_assignment = None
    
    for k in range(1, 11):
        # Check if all residues are covered by subsets of size <= k
        covered = {}
        for length in range(1, k + 1):
            for residue, terms in subsets_by_length[length]:
                if residue not in covered:
                    covered[residue] = terms
        
        if len(covered) == p:
            best_k = k
            best_assignment = covered
            break
    
    if best_assignment is None:
        # Should not happen, but fallback
        best_assignment = {}
        for length, residue, terms in subsets:
            if residue not in best_assignment:
                best_assignment[residue] = terms
        best_k = max(len(v) for v in best_assignment.values())
    
    # Build the result: 101 lists, row r contains the assigned denominators
    result = [None] * p
    for r in range(p):
        if r in best_assignment:
            result[r] = best_assignment[r]
        else:
            # Fallback - should not happen
            result[r] = [1]
    
    return result
# EVOLVE-BLOCK-END
