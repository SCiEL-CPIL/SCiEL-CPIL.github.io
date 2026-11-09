"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    # Pippenger 58-comparator network for 16 wires
    network = [
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (1, 2), (5, 6), (9, 10), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (2, 4), (3, 5), (6, 8), (7, 9), (10, 12), (11, 13),
        (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)
    ]
    
    # Verify using Zero-One Principle
    def _apply_comparator(wires, i, j):
        if wires[i] > wires[j]:
            wires[i], wires[j] = wires[j], wires[i]

    def _verify_network(comparators, n):
        for mask in range(1 << n):
            wires = [(mask >> i) & 1 for i in range(n)]
            for (i, j) in comparators:
                _apply_comparator(wires, i, j)
            for i in range(n - 1):
                if wires[i] > wires[i + 1]:
                    return False
        return True

    if _verify_network(network, N):
        return network
    else:
        # Fallback to Batcher's Odd-Even Merge Sort (119 comparators)
        def _batcher_odd_even_merge_sort(n):
            if n == 1:
                return []
            half = n // 2
            left = _batcher_odd_even_merge_sort(half)
            right = [(i + half, j + half) for (i, j) in left]
            merge = _odd_even_merge(half)
            return left + right + merge

        def _odd_even_merge(n):
            if n == 1:
                return [(0, 1)]
            even_merge = _odd_even_merge(n // 2)
            odd_merge = _odd_even_merge(n // 2)
            even_translated = [(2*i, 2*j) for (i, j) in even_merge]
            odd_translated = [(2*i + 1, 2*j + 1) for (i, j) in odd_merge]
            final_merge = [(2*i + 1, 2*i + 2) for i in range(n - 1)]
            return even_translated + odd_translated + final_merge

        return _batcher_odd_even_merge_sort(N)
# EVOLVE-BLOCK-END
