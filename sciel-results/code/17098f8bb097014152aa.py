import itertools
import time

def construct():
    # Modular inverses of 1..10 mod 101
    invs = [pow(i, -1, 101) for i in range(1, 11)]
    
    # Precompute all subsets of denominators 1..10 with their residues
    all_subsets = []
    for mask in range(1, 1 << 10):
        terms = tuple(i + 1 for i in range(10) if (mask >> i) & 1)
        residue = sum(invs[i - 1] for i in terms) % 101
        all_subsets.append((residue, len(terms), terms))
    
    # Group subsets by residue
    subsets_by_residue = {r: [] for r in range(101)}
    for residue, size, terms in all_subsets:
        subsets_by_residue[residue].append((size, terms))
    
    # For each residue, sort by size then lexicographically
    for r in range(101):
        subsets_by_residue[r].sort()
    
    def try_max_size(K):
        """Try to find an assignment of subsets to residues such that:
        - Each residue gets exactly one subset
        - Each subset has size <= K
        - Subsets may share denominators across different residues (allowed)
        - Denominators within a single subset are distinct (guaranteed by construction)
        
        Returns a dict mapping residue -> terms, or None if no solution exists.
        """
        # Since denominators may recur across different rows, this is simply:
        # For each residue, pick a subset of size <= K.
        # The constraint is just that each residue must have at least one subset of size <= K.
        
        res_map = {}
        for r in range(101):
            found = False
            for size, terms in subsets_by_residue[r]:
                if size <= K:
                    res_map[r] = terms
                    found = True
                    break
            if not found:
                return None
        
        return res_map

    # Try to find a solution with max subset size 2, then 3, etc.
    # Start from K=1 since that's the theoretical minimum
    solution = None
    for K in range(1, 11):
        solution = try_max_size(K)
        if solution is not None:
            break
            
    if solution is None:
        # Fallback: should not happen since K=10 covers all subsets
        result = []
        for r in range(101):
            result.append(list(subsets_by_residue[r][0][1]))
        return result

    result = []
    for r in range(101):
        result.append(list(solution[r]))
        
    return result
