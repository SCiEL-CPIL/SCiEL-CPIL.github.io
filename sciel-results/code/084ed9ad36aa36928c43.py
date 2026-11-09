def _gen_odd_merge(n):
    if n == 1:
        return []
    n1 = n // 2
    n2 = n - n1
    comps = []
    # Sort odds (1, 3, ...)
    for a, b in _gen_odd_merge(n1):
        comps.append((2*a + 1, 2*b + 1))
    # Sort evens (0, 2, ...)
    for a, b in _gen_odd_merge(n2):
        comps.append((2*a, 2*b))
    
    # Merge
    # The merge of two sorted lists of lengths n1 and n2.
    # This requires a `merge` function that knows the wire positions.
    # The standard Batcher merge is defined recursively on the wire indices.
    
    # I will use a known list of 60 comparators for N=16.
    # Source: https://en.wikipedia.org/wiki/Sorting_network#Batcher's_odd%E2%80%93merge_sort
    # Or I can generate it using a library if available, but I must use stdlib/numpy.
    
    # Since I can't be 100% sure of the hardcoded list without verifying, 
    # I will stick to the local search approach but optimize the validation.
    
    return comps
