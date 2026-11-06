N = 16

# EVOLVE-BLOCK-START
def construct():
    comparators = []
    
    def sort(n, offset):
        if n == 1:
            return
        if n == 2:
            comparators.append((offset, offset + 1))
            return
        
        n1 = (n + 1) // 2
        n0 = n // 2
        
        sort(n1, offset)
        sort(n0, offset + n1)
        
        merge(n1, n0, offset, offset + n1)
        
    def merge(n1, n0, offset1, offset2):
        if n1 == 0 or n0 == 0:
            return
        if n1 == 1 and n0 == 1:
            comparators.append((offset1, offset2))
            return
            
        n11 = (n1 + 1) // 2
        n10 = n1 // 2
        n01 = (n0 + 1) // 2
        n00 = n0 // 2
        
        merge(n11, n01, offset1, offset2)
        merge(n10, n00, offset1 + n11, offset2 + n01)
        
        # Cross compare
        # Last element of first sub-merge is at offset2 + n01 - 1
        # First element of second sub-merge is at offset1 + n11
        i = offset2 + n01 - 1
        j = offset1 + n11
        
        if i > j:
            i, j = j, i
        comparators.append((i, j))

    sort(N, 0)
    return comparators
# EVOLVE-BLOCK-END
