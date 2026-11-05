"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
def construct():
    return _generate_odd_merge(N)

def _generate_odd_merge(n):
    """
    Generates a Batcher's Odd-Merge sorting network for n wires.
    """
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Split indices into even and odd positions
    even_indices = [i for i in range(0, n, 2)]
    odd_indices = [i for i in range(1, n, 2)]
    
    n_even = len(even_indices)
    n_odd = len(odd_indices)
    
    # Recursively generate networks for the even and odd subsets
    comps_even = _generate_odd_merge(n_even)
    comps_odd = _generate_odd_merge(n_odd)
    
    # Map the local indices (0..n_even-1 and 0..n_odd-1) to the global indices
    mapped_even = [(even_indices[i], even_indices[j]) for i, j in comps_even]
    mapped_odd = [(odd_indices[i], odd_indices[j]) for i, j in comps_odd]
    
    # Merge step: Compare E_i with O_i
    merge_comps = []
    for i in range(n // 2):
        e_idx = even_indices[i]
        o_idx = odd_indices[i]
        if e_idx < o_idx:
            merge_comps.append((e_idx, o_idx))
        else:
            merge_comps.append((o_idx, e_idx))
            
    return mapped_even + mapped_odd + merge_comps
# EVOLVE-BLOCK-END
