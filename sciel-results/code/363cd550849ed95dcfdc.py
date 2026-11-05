"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def _polish_result(result, subsets, P, deadline):
    """Try to reduce individual row lengths by finding smaller valid subsets."""
    # Build a lookup: for each residue, list all subsets that produce it, sorted by size
    residue_to_subsets = {}
    for r, s, t in subsets:
        if r not in residue_to_subsets:
            residue_to_subsets[r] = []
        residue_to_subsets[r].append((s, t))
    
    # Sort each list by size
    for r in residue_to_subsets:
        residue_to_subsets[r].sort(key=lambda x: x[0])
    
    # Try to replace each row with a smaller subset if one exists
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for r in range(101):
            current_size = len(result[r])
            # Look for a subset of this residue with size < current_size
            for s, t in residue_to_subsets.get(r, []):
                if s < current_size:
                    result[r] = t
                    improved = True
                    break
                elif s == current_size:
                    break  # No point continuing, sizes are sorted
                else:
                    break
    
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s/90s limit
    
    P = 101
    DENOMS = list(range(1, 11))
    
    # Precompute inverses
    inv = {x: pow(x, -1, P) for x in DENOMS}
    
    # Generate all non-empty subsets with their residue and size
    # For each mask from 1 to 2^10 - 1
    subsets = []  # list of (residue, size, terms_list)
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x-1)) & 1]
        residue = sum(inv[x] for x in terms) % P
        size = len(terms)
        subsets.append((residue, size, terms))
    
    # Try increasing max size k, find the smallest k such that all residues are covered
    for k in range(1, 11):
        if time.time() > deadline:
            break
        # Filter subsets with size <= k
        available = [(r, s, t) for (r, s, t) in subsets if s <= k]
        
        # Check if all 101 residues are covered
        covered = set()
        for r, s, t in available:
            covered.add(r)
        
        if len(covered) == 101:
            # All residues covered with max size k
            # Build the result: for each residue, pick the smallest subset
            # Sort by size then by residue to prioritize smaller subsets
            available_sorted = sorted(available, key=lambda x: (x[1], x[0]))
            result = [None] * 101
            for r, s, t in available_sorted:
                if result[r] is None:
                    result[r] = t
            
            # Verify all are set
            all_set = all(result[r] is not None for r in range(101))
            if all_set:
                # Local polish: try to reduce individual row lengths
                if time.time() < deadline - 5.0:
                    result = _polish_result(result, subsets, P, deadline)
                return result
    
    # Fallback: use any subset (should not happen, but safe)
    representations = {}
    for r, s, t in subsets:
        if r not in representations:
            representations[r] = t
    
    # If some residues are missing, we need to handle it
    # But with 1023 subsets and 101 residues, this should be covered
    result = [None] * 101
    for r in range(101):
        if r in representations:
            result[r] = representations[r]
        else:
            # Should not happen, but provide a default
            result[r] = [1]
    
    return result
# EVOLVE-BLOCK-END
