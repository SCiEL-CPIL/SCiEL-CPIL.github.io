def construct():
    N = 16
    
    def batcher_sort(wires):
        n = len(wires)
        if n <= 1:
            return []
        mid = n // 2
        left = wires[:mid]
        right = wires[mid:]
        comparators = []
        comparators += batcher_sort(left)
        comparators += batcher_sort(right)
        comparators += batcher_merge(left, right)
        return comparators
    
    def batcher_merge(left, right):
        n = len(left)
        if n == 1:
            return [(left[0], right[0])]
        
        comparators = []
        left_odd = left[1::2]
        left_even = left[0::2]
        right_odd = right[1::2]
        right_even = right[0::2]
        
        comparators += batcher_merge(left_odd, right_odd)
        comparators += batcher_merge(left_even, right_even)
        
        for i in range(n // 2):
            comparators.append((left_even[i], left_odd[i]))
        
        return comparators
    
    return batcher_sort(list(range(N)))
