"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
def _bitonic_sort(n):
    """
    Generate Batcher's bitonic sort network for n wires (n must be power of 2).
    Returns list of (i, j) comparator pairs.
    """
    if n <= 1:
        return []
    
    half = n // 2
    
    # Recursively sort first half in ascending order
    left = _bitonic_sort(half)
    left_mapped = [(a, b) for (a, b) in left]
    
    # Recursively sort second half in descending order
    right = _bitonic_sort(half)
    # For descending order, swap comparator directions
    right_mapped = [(b, a) for (a, b) in right]
    right_mapped = [(a + half, b + half) for (a, b) in right_mapped]
    
    # Bitonic merge
    merge = _bitonic_merge(n, half)
    
    return left_mapped + right_mapped + merge

def _bitonic_merge(n, half):
    """
    Merge two sorted halves (first ascending, second descending) into
    a fully sorted sequence.
    """
    if half <= 1:
        return []
    
    quarter = half // 2
    
    # Compare-exchange pairs (i, i + half) for i = 0..half-1
    merge_step = [(i, i + half) for i in range(half)]
    
    # Recursively merge the two halves
    left = _bitonic_merge(half, quarter)
    left_mapped = [(a, b) for (a, b) in left]
    
    right = _bitonic_merge(half, quarter)
    right_mapped = [(a + half, b + half) for (a, b) in right]
    
    return merge_step + left_mapped + right_mapped

def construct():
    """
    Construct a sorting network for N=18 wires using Batcher's bitonic sort.
    We use a 32-wire network and prune to 18 wires.
    """
    N_p2 = 32
    net = _bitonic_sort(N_p2)
    
    # Prune to first 18 wires
    pruned = [(i, j) for (i, j) in net if i < N and j < N]
    
    return pruned
# EVOLVE-BLOCK-END
