def construct():
    # Calculate the minimal subset size and corresponding subset for each residue
    min_size_for_residue = [11] * 101
    best_subset_for_residue = [None] * 101
    
    # Iterate through all non-empty subsets of denominators 1..10
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        residue = sum(pow(x, -1, 101) for x in terms) % 101
        size = len(terms)
        
        # Keep the smallest subset found for this residue
        if size < min_size_for_residue[residue]:
            min_size_for_residue[residue] = size
            best_subset_for_residue[residue] = terms
            
    # Construct the result list
    # Since we picked the minimal size for each residue, the max row length is minimized
    result = []
    for r in range(101):
        if best_subset_for_residue[r] is not None:
            result.append(best_subset_for_residue[r])
        else:
            # Fallback, though all residues should be covered for p=101
            result.append([])
            
    return result
