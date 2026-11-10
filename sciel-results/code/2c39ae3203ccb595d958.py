def construct():
    best_set = list(range(1, 21))
    best_score = float('inf')
    
    # Check APs starting from 1 to 100
    for start in range(1, 101):
        A = [start + i for i in range(20)]
        if max(A) > 10000:
            continue
        sums = set()
        prods = set()
        for x in A:
            for y in A:
                sums.add(x + y)
                prods.add(x * y)
        score = max(len(sums), len(prods))
        if score < best_score:
            best_score = score
            best_set = A
            
    # Also check a few specific "factor-rich" sets
    # Example: Multiples of 2? {2, 4, ..., 40}
    A = [2 * (i + 1) for i in range(20)]
    sums = set()
    prods = set()
    for x in A:
        for y in A:
            sums.add(x + y)
            prods.add(x * y)
    score = max(len(sums), len(prods))
    if score < best_score:
        best_score = score
        best_set = A
        
    return best_set
