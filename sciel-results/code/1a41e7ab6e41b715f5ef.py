"""Improved sorting network for 18 wires using Batcher's odd-even merge sort."""
N = 18

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a compare-exchange network that sorts 18 inputs using Batcher's odd-even merge sort.
    """
    def odd_even_merge_sort_network(n):
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        comparators = []
        
        def generate_sort(n):
            if n == 1:
                return []
            if n == 2:
                return [(0, 1)]
            
            mid = n // 2
            left = generate_sort(mid)
            right = generate_sort(n - mid)
            right = [(i + mid, j + mid) for (i, j) in right]
            
            result = left + right
            
            # Now merge the two sorted halves
            result += odd_even_merge(mid, n - mid, 0, mid)
            return result
        
        def odd_even_merge(a, b, offset_a, offset_b):
            """Generate comparators to merge two sorted sequences of lengths a and b."""
            if a == 0 or b == 0:
                return []
            if a == 1 and b == 1:
                return [(offset_a, offset_b)]
            if a == 1:
                return [(offset_a, offset_b + i) for i in range(b)]
            if b == 1:
                return [(offset_a + i, offset_b) for i in range(a)]
            
            a1, a2 = a // 2, a - a // 2
            b1, b2 = b // 2, b - b // 2
            
            result = []
            result += odd_even_merge(a1, b1, offset_a, offset_b)
            result += odd_even_merge(a2, b2, offset_a + 1, offset_b + 1)
            
            # Cross comparisons
            for i in range(min(a1, b2)):
                result.append((offset_a + 2 * i + 1, offset_b + 2 * i))
            for i in range(min(a2, b1)):
                result.append((offset_a + 2 * i, offset_b + 2 * i + 1))
            
            return result
        
        return generate_sort(n)
    
    return odd_even_merge_sort_network(N)
# EVOLVE-BLOCK-END
