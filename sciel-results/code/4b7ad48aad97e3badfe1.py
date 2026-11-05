# EVOLVE-BLOCK-START
def construct():
    best_set = list(range(1, 21))
    best_score = float('inf')
    
    # Search for the best Arithmetic Progression
    # A = {a, a+d, ..., a+19d}
    # |A+A| is always 39 for an AP of 20 distinct integers.
    # We minimize |A*A|.
    
    # We search a small range of a and d.
    # a=1, d=1 is the baseline.
    # We can try a=1..20, d=1..20.
    
    for a in range(1, 21):
        for d in range(1, 21):
            A = [a + i * d for i in range(20)]
            if A[-1] > 10000:
                continue
            
            # Calculate product set size
            prod_set = set()
            for x in A:
                for y in A:
                    prod_set.add(x * y)
            
            size_prod = len(prod_set)
            size_sum = 39
            
            current_max = max(size_sum, size_prod)
            
            if current_max < best_score:
                best_score = current_max
                best_set = A
                
    return best_set
# EVOLVE-BLOCK-END
