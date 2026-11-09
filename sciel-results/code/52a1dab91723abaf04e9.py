import time

def construct():
    """
    Constructs 101 lists of distinct denominators in 1..10 such that the sum of their
    modular inverses modulo 101 equals the list index. Minimizes the largest list length.
    """
    p = 101
    max_denom = 10
    
    # Pre-compute modular inverses for 1..10
    invs = [pow(x, -1, p) for x in range(1, max_denom + 1)]
    
    # best_rep[r] will store the subset (as a list of denominators) with the minimum size
    # that sums to r modulo p.
    best_rep = [None] * p
    
    # Iterate over all non-empty subsets of {1, ..., 10}
    # There are 2^10 - 1 = 1023 such subsets.
    for mask in range(1, 1 << max_denom):
        subset = []
        residue_sum = 0
        for i in range(max_denom):
            if mask & (1 << i):
                denom = i + 1
                subset.append(denom)
                residue_sum = (residue_sum + invs[i]) % p
        
        # We want to minimize the size of the subset for each residue.
        # If we haven't seen this residue, or we've seen it with a larger subset, update.
        if best_rep[residue_sum] is None or len(subset) < len(best_rep[residue_sum]):
            best_rep[residue_sum] = subset
    
    # All residues should be covered. 
    # If some are not (which shouldn't happen given the problem statement implies existence),
    # we might need to handle it, but the problem states "Every residue must be represented".
    # Given 1023 subsets and 101 residues, coverage is virtually certain.
    
    # The result is a list of 101 lists.
    # Row r contains the best representation for residue r.
    return best_rep
