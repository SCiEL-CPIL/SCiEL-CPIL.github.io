def construct():
    import time
    start_time = time.time()
    deadline = start_time + 80
    
    from fractions import Fraction
    
    def all_subset_sums_distinct(elements):
        if not elements:
            return True
        sums = {Fraction(0)}
        for e in elements:
            inv = Fraction(1, e)
            new_sums = {s + inv for s in sums}
            if len(new_sums) != len(sums):
                return False
            sums = sums | new_sums
        return True
    
    all_candidates = list(range(1, 23))
    best = []
    
    # Try greedy with different orderings
    import random
    random.seed(42)
    
    orders = [
        all_candidates,
        sorted(all_candidates, reverse=True),
    ]
    # Add some random orders
    for _ in range(5):
        shuffled = all_candidates[:]
        random.shuffle(shuffled)
        orders.append(shuffled)
    
    for order in orders:
        if time.time() > deadline - 10:
            break
        current = []
        for c in order:
            test = current + [c]
            if all_subset_sums_distinct(test):
                current = test
        if len(current) > len(best):
            best = current
        if len(best) == 22:
            break
    
    if len(best) > 0:
        return best
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
