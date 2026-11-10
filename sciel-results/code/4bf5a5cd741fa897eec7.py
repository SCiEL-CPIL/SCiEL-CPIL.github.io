def construct():
    from math import gcd
    import random
    
    def can_add(selected, x):
        for i in range(len(selected)):
            for j in range(i+1, len(selected)):
                a, b = selected[i], selected[j]
                if gcd(a, b) == gcd(a, x) == gcd(b, x):
                    return False
        return True
    
    best = []
    
    # Deterministic orderings
    orderings = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        sorted(range(1, 129), key=lambda x: -x.bit_count()),
        sorted(range(1, 129), key=lambda x: x.bit_count()),
        list(range(1, 129, 2)) + list(range(2, 129, 2)),
        list(range(2, 129, 2)) + list(range(1, 129, 2)),
    ]
    
    # Add some random orderings
    rng = random.Random(42)
    for _ in range(20):
        order = list(range(1, 129))
        rng.shuffle(order)
        orderings.append(order)
    
    for order in orderings:
        selected = []
        for x in order:
            if can_add(selected, x):
                selected.append(x)
        if len(selected) > len(best):
            best = selected[:]
    
    if not best:
        best = [1]
    return best
