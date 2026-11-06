import time

N = 16

def construct():
    # Generate bitonic sorting network for N=16
    comparators = []
    
    def bitonic_sort(k, start, direction):
        if k == 1:
            return
        bitonic_sort(k // 2, start, 1)
        bitonic_sort(k // 2, start + k // 2, -1)
        
        # Merge step
        # The merge step for size k has log2(k) layers
        # Layer i (1 to log2(k)) compares distance 2^(i-1)
        # Wait, for size 4, layers are distance 2 and 1.
        # For size 8, layers are distance 4, 2, 1.
        # So distance is k/2, k/4, ..., 1.
        
        for i in range(int(__import__('math').log2(k)), 0, -1):
            d = 1 << (i - 1)  # distance: 1, 2, 4...
            # Wait, if k=4, log2(4)=2.
            # i=2: d=2.
            # i=1: d=1.
            # This matches.
            
            for j in range(start, start + k, 2 * d):
                for l in range(d):
                    # Compare j+l and j+l+d
                    # But we need to ensure we don't go out of bounds?
                    # The loop range ensures j+l+d < start+k
                    a = j + l
                    b = j + l + d
                    comparators.append((a, b))
                    
    bitonic_sort(N, 0, 1)
    
    # The generated list might have duplicates?
    # If so, the count is len(comparators).
    # If the count is > 53, I might need to optimize.
    # But for now, return the list.
    
    return comparators
