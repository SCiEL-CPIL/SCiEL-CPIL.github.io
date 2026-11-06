import time

def construct():
    P = 101
    D = list(range(1, 11))
    
    # Precompute inverses
    inv = [0] * 11
    for x in D:
        inv[x] = pow(x, -1, P)
        
    # Generate all subsets with their residue and size
    # subset_data: list of (residue, size, list_of_terms)
    subset_data = []
    
    for mask in range(1, 1 << 10):
        terms = []
        res = 0
        size = 0
        for x in D:
            if mask & (1 << (x - 1)):
                terms.append(x)
                res = (res + inv[x]) % P
                size += 1
        subset_data.append((res, size, terms))
        
    # Sort by size to find the minimum k
    # We want to check if residues from sizes 1..k cover all 0..100
    # Let's group by size or just sort
    subset_data.sort(key=lambda x: x[1])
    
    # We need to cover all 101 residues
    # Let's iterate k from 1 to 10
    # For a given k, we look at all subsets with size <= k
    # We need to see if the set of their residues is all 101
    
    # To speed up, we can just check the union incrementally
    covered = set()
    best_solution = None
    best_k = 11
    
    # We need to pick specific subsets for the final answer
    # Let's store the best subset for each residue for the current k
    # Actually, we can just find the minimal k first, then pick witnesses.
    
    # Let's just iterate k and check coverage
    # Since N is small, we can just re-check or update incrementally.
    # Let's update incrementally.
    
    # However, we need to return the actual lists.
    # Let's find the minimal k first.
    
    current_max_size = 0
    residues_covered = set()
    
    # We iterate through subsets sorted by size.
    # We add them to the pool.
    # When all 101 residues are covered, we have found the minimal k.
    # But we need to be careful: we are adding subsets one by one.
    # The "k" is the size of the current subset being added.
    # If we add a subset of size 4 and it completes the coverage, then k=4.
    
    # We need to store which subset covers which residue.
    # If multiple subsets cover the same residue, we can keep any.
    # But we want to minimize the max size.
    # If we process in increasing size order, the first time a residue is covered,
    # it is covered by a subset of the smallest possible size?
    # Not necessarily. A residue might be covered by size 2, but we might have
    # already "used" that slot? No, the slots are independent.
    # Each row r corresponds to residue r. We just need *one* subset for row r.
    # So, for each residue r, we want the subset with the *smallest* size that maps to r.
    # If we do that, the max row length is $\max_r \min \{|A| : f(A)=r\}$.
    # This is exactly what we want to minimize.
    
    # So the strategy is:
    # 1. For each residue r in 0..100, find the minimum size of a subset A such that f(A)=r.
    # 2. Let k* be the maximum of these minimum sizes.
    # 3. The answer is k*.
    # 4. Construct the output by picking, for each r, a subset of size min_size[r].
    
    min_size_for_residue = [float('inf')] * 101
    best_subset_for_residue = [None] * 101
    
    for res, size, terms in subset_data:
        if size < min_size_for_residue[res]:
            min_size_for_residue[res] = size
            best_subset_for_residue[res] = terms
            
    # Check if all residues are covered
    # Since there are 1023 subsets and 101 residues, it's very likely all are covered.
    # But let's verify.
    for r in range(101):
        if min_size_for_residue[r] == float('inf'):
            # This shouldn't happen for k=10, but just in case
            # If it happens, we have a problem.
            pass
            
    # The maximum length is max(min_size_for_residue)
    max_len = max(min_size_for_residue)
    
    # Construct the result
    result = []
    for r in range(101):
        result.append(best_subset_for_residue[r])
        
    return result
