def batcher_oems_indices(n):
    comparators = []
    def merge(indices_a, indices_b):
        # indices_a and indices_b are lists of indices
        # They must be of same length k
        k = len(indices_a)
        if k == 1:
            comparators.append((indices_a[0], indices_b[0]))
            return
        half = k // 2
        # Odd indices: 1, 3, ...
        merge(indices_a[1::2], indices_b[1::2])
        # Even indices: 0, 2, ...
        merge(indices_a[0::2], indices_b[0::2])
        # Interleave
        for i in range(k - 1):
            comparators.append((indices_a[i], indices_b[i]))
            
    def sort(indices):
        n = len(indices)
        if n <= 1:
            return
        half = n // 2
        sort(indices[:half])
        sort(indices[half:])
        merge(indices[:half], indices[half:])
        
    sort(list(range(n)))
    return comparators
