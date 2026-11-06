"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""

N = 16

# EVOLVE-BLOCK-START

def _oes(n):
    """Batcher OES for n wires."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    h = n // 2
    left = _oes(h)
    right = [(a+h, b+h) for a, b in _oes(h)]
    merge = _ome(n, h)
    return left + right + merge

def _ome(n, h):
    """Odd-even merge of two sorted halves of size h, total n=2h wires."""
    if h == 1:
        return [(0, 1)]
    hh = h // 2
    # Even-indexed: wires 0,2,4,...,h-2 and h,h+2,...,2h-2
    even1 = list(range(0, h, 2))
    even2 = list(range(h, 2*h, 2))
    # Odd-indexed: wires 1,3,5,...,h-1 and h+1,h+3,...,2h-1
    odd1 = list(range(1, h, 2))
    odd2 = list(range(h+1, 2*h, 2))
    
    # Recursively merge even and odd subsequences
    # The even subsequence merge operates on 2*hh = h wires
    # We need to map the recursive result back to the actual wires.
    
    def _ome_sub(n_sub, wires1, wires2):
        """Merge two sorted subsequences on given wires."""
        if len(wires1) == 1:
            return [(wires1[0], wires2[0])]
        hs = len(wires1) // 2
        w1_even = wires1[0::2]
        w2_even = wires2[0::2]
        w1_odd = wires1[1::2]
        w2_odd = wires2[1::2]
        comps = _ome_sub(hs, w1_even, w2_even)
        comps += _ome_sub(hs, w1_odd, w2_odd)
        # Adjacent comparisons
        all_w = sorted(wires1 + wires2)
        for i in range(0, len(all_w), 2):
            comps.append((all_w[i], all_w[i+1]))
        return comps
    
    comps = _ome_sub(h, even1, even2)
    comps += _ome_sub(h, odd1, odd2)
    # Final adjacent comparisons
    for i in range(0, n, 2):
        comps.append((i, i+1))
    return comps

def construct():
    return _oes(16)

# EVOLVE-BLOCK-END
